export SERVER_HOST=$(hostname -I | awk '{print $1}')

docker compose down
docker compose up -d
docker compose logs -f --no-log-prefix
