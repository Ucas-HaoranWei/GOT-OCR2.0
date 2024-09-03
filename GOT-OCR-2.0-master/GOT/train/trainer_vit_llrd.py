import os
import torch
import torch.nn as nn
import time
import functools
import re

from transformers import Trainer
from transformers.trainer_pt_utils import (
    get_module_class_from_name,
    get_parameter_names,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_neuroncore_available,
)
from transformers.trainer_utils import (
    FSDPOption,
    ShardedDDPOption,
)
from transformers.training_args import ParallelMode
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from typing import Dict, Optional, Sequence


def lr_scale_func(key):
    if "vision_model.encoder.layers" in key:
        in_pp_layer = int(re.findall(f"layers\.(\d+)\.", key)[0])
        # decay = 0.81 ** (23 - in_pp_layer - 1)
        decay = 0.81 ** (23 - in_pp_layer - 1) * 0.01
        # decay = 0.66 ** (23 - in_pp_layer - 1)
        return decay
        # return 0.01
    elif "vision_model" in key:
        # return 0.01
        return 0.0001
    return 1
                

def get_param_groups(model, no_weight_decay_cond, scale_lr_cond, lr, wd):
    """creates param groups based on weight decay condition (regularized vs non regularized)
    and learning rate scale condition (args.lr vs lr_mult * args.lr)
    scale_lr_cond is used during finetuning where head of the network requires a scaled
    version of the base learning rate.
    """
    wd_no_scale_lr = []
    wd_scale_lr = {}
    no_wd_no_scale_lr = []
    no_wd_scale_lr = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if no_weight_decay_cond is not None:
            no_wd = no_weight_decay_cond(name, param)
        else:
            # do not regularize biases nor Norm parameters
            no_wd = name.endswith(".bias") or len(param.shape) == 1

        if scale_lr_cond is not None:
            lr_mult = scale_lr_cond(name)
            print(name, lr_mult)
            scale_lr = lr_mult != 1
        else:
            scale_lr = False

        if not no_wd and not scale_lr:
            wd_no_scale_lr.append(param)
        elif not no_wd and scale_lr:
            if lr_mult not in wd_scale_lr:
                wd_scale_lr[lr_mult] = [param]
            else:
                wd_scale_lr[lr_mult].append(param)
        elif no_wd and not scale_lr:
            no_wd_no_scale_lr.append(param)
        else:
            if lr_mult not in no_wd_scale_lr:
                no_wd_scale_lr[lr_mult] = [param]
            else:
                no_wd_scale_lr[lr_mult].append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({"params": wd_no_scale_lr, "weight_decay": wd, "lr": lr})
    if len(wd_scale_lr):
        for lr_mult, params in wd_scale_lr.items():
            param_groups.append({"params": params, "weight_decay": wd, "lr": lr * lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append(
            {"params": no_wd_no_scale_lr, "weight_decay": 0.0, "lr": lr}
        )
    if len(no_wd_scale_lr):
        for lr_mult, params in no_wd_scale_lr.items():
            param_groups.append({"params": params, "weight_decay": 0.0, "lr": lr * lr_mult})

    return param_groups


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class GOTTrainer(Trainer):

    def _safe_save(self, output_dir: str):
        """Collects the state dict and dump to disk."""
        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            self._save(output_dir, state_dict=cpu_state_dict)  # noqa


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(GOTTrainer, self)._save(output_dir, state_dict)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            # decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # optimizer_grouped_parameters = [
            #     {
            #         "params": [
            #             p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
            #         ],
            #         "weight_decay": self.args.weight_decay,
            #     },
            #     {
            #         "params": [
            #             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            #         ],
            #         "weight_decay": 0.0,
            #     },
            # ]

            optimizer_grouped_parameters = get_param_groups(opt_model, None, lr_scale_func, self.args.learning_rate, self.args.weight_decay)
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)
        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            from apex import amp
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        if self.args.jit_mode_eval:
            start_time = time.time()
            model = self.torch_jit_model_eval(model, dataloader, training)
            self.jit_compilation_time = round(time.time() - start_time, 4)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
            from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
            from fairscale.nn.wrap import auto_wrap
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)
        # Distributed training using PyTorch FSDP
        elif self.fsdp is not None:
            if not self.args.fsdp_config["xla"]:
                # PyTorch FSDP!
                from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

                if FSDPOption.OFFLOAD in self.args.fsdp:
                    cpu_offload = CPUOffload(offload_params=True)
                else:
                    cpu_offload = CPUOffload(offload_params=False)

                auto_wrap_policy = None

                if FSDPOption.AUTO_WRAP in self.args.fsdp:
                    if self.args.fsdp_config["fsdp_min_num_params"] > 0:
                        auto_wrap_policy = functools.partial(
                            size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["fsdp_min_num_params"]
                        )
                    elif self.args.fsdp_config.get("fsdp_transformer_layer_cls_to_wrap", None) is not None:
                        transformer_cls_to_wrap = set()
                        for layer_class in self.args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"]:
                            transformer_cls = get_module_class_from_name(model, layer_class)
                            if transformer_cls is None:
                                raise Exception("Could not find the transformer layer class to wrap in the model.")
                            else:
                                transformer_cls_to_wrap.add(transformer_cls)
                        auto_wrap_policy = functools.partial(
                            transformer_auto_wrap_policy,
                            # Transformer layer class to wrap
                            transformer_layer_cls=transformer_cls_to_wrap,
                        )
                mixed_precision_policy = None
                dtype = None
                if self.args.fp16:
                    dtype = torch.float16
                elif self.args.bf16:
                    dtype = torch.bfloat16
                if dtype is not None:
                    mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
                if type(model) != FSDP:
                    # XXX: Breaking the self.model convention but I see no way around it for now.
                    self.model = model = FSDP(
                        model,
                        sharding_strategy=self.fsdp,
                        cpu_offload=cpu_offload,
                        auto_wrap_policy=auto_wrap_policy,
                        mixed_precision=mixed_precision_policy,
                        device_id=self.args.device,
                        backward_prefetch=self.backward_prefetch,
                        forward_prefetch=self.forword_prefetch,
                        limit_all_gathers=self.limit_all_gathers,
                        use_orig_params=True,
                    )
            else:
                try:
                    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
                    from torch_xla.distributed.fsdp import checkpoint_module
                    from torch_xla.distributed.fsdp.wrap import (
                        size_based_auto_wrap_policy,
                        transformer_auto_wrap_policy,
                    )
                except ImportError:
                    raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
                auto_wrap_policy = None
                auto_wrapper_callable = None
                if self.args.fsdp_config["fsdp_min_num_params"] > 0:
                    auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["fsdp_min_num_params"]
                    )
                elif self.args.fsdp_config.get("fsdp_transformer_layer_cls_to_wrap", None) is not None:
                    transformer_cls_to_wrap = set()
                    for layer_class in self.args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"]:
                        transformer_cls = get_module_class_from_name(model, layer_class)
                        if transformer_cls is None:
                            raise Exception("Could not find the transformer layer class to wrap in the model.")
                        else:
                            transformer_cls_to_wrap.add(transformer_cls)
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        # Transformer layer class to wrap
                        transformer_layer_cls=transformer_cls_to_wrap,
                    )
                fsdp_kwargs = self.args.xla_fsdp_config
                if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
                    # Apply gradient checkpointing to auto-wrapped sub-modules if specified
                    def auto_wrapper_callable(m, *args, **kwargs):
                        return FSDP(checkpoint_module(m), *args, **kwargs)

                # Wrap the base model with an outer FSDP wrapper
                self.model = model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    auto_wrapper_callable=auto_wrapper_callable,
                    **fsdp_kwargs,
                )

                import torch_xla.core.xla_model as xm
                # Patch `xm.optimizer_step` should not reduce gradients in this case,
                # as FSDP does not need gradient reduction over sharded parameters.
                def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
                    loss = optimizer.step(**optimizer_args)
                    if barrier:
                        xm.mark_step()
                    return loss

                xm.optimizer_step = patched_optimizer_step
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                **kwargs,
            )

        # torch.compile() needs to be called after wrapping the model with FSDP or DDP
        # to ensure that it accounts for the graph breaks required by those wrappers
        if self.args.torch_compile:
            model = torch.compile(model, backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode)

        return model

