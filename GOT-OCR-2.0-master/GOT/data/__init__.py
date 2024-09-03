
import torch
import transformers
from dataclasses import dataclass, field

from GOT.utils.constants import *


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # print(instances)
        # exit()
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        images = [torch.stack(instance['image']) for instance in instances]

        # if 'flattened_patches' in instances[0]['image_high'][0].keys():
        #     images_high = [torch.stack([instance['image_high'][0]['flattened_patches']]) for instance in instances]
        # else:
        images_high = [torch.stack(instance['image_high']) for instance in instances]

        images = list(zip(images, images_high))


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        return batch
    

def make_supervised_data_module(interleave, with_box, tokenizer, data_args):

    if data_args.conversation_version == 'mpt':
        from GOT.data.conversation_dataset_qwen import ConversationDataset
        dataset_cls = ConversationDataset
        
    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        datasets=data_args.datasets,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high = data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)