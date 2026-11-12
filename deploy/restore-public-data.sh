#!/bin/bash
# Runs ON the EC2 host. Replaces everything the public instance serves with
# one exported tenant (see deploy/export-public-data.py), and moves the stack
# to the public read-only overlay on a new app image.
#
#   IMAGE=<ecr-uri>:<tag> BUNDLE_URL=<https url of the .tgz> bash restore-public-data.sh
#
# The bundle holds: the compose files and deploy/ configs to run, the
# tenant's Postgres rows as CSV with a restore.sql, its ClickHouse spans in
# Native format, and the Alembic revision they were exported at. Postgres is
# restored in one transaction; a schema-revision mismatch aborts before any
# data is touched.
set -euo pipefail
cd /opt/agentlens

REGION=us-east-1
: "${IMAGE:?IMAGE is required}"
: "${BUNDLE_URL:?BUNDLE_URL is required}"
param() { aws ssm get-parameter --name "/agentlens/prod/$1" --with-decryption --query Parameter.Value --output text --region "$REGION"; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
curl -fsS "$BUNDLE_URL" -o "$WORK/bundle.tgz"
tar -xzf "$WORK/bundle.tgz" -C "$WORK"
echo "== bundle: $(cat "$WORK/manifest.txt")"

echo "== 1. compose files and configs"
cp "$WORK"/files/docker-compose*.yml .
mkdir -p deploy && cp "$WORK"/files/deploy/* deploy/
rm -f docker-compose.demo.yml

export AGENTLENS_IMAGE="$IMAGE"
export AGENTLENS_POSTGRES_PASSWORD=$(param postgres_password)
export AGENTLENS_CLICKHOUSE_PASSWORD=$(param clickhouse_password)
export AGENTLENS_APP_ROLE_PASSWORD=$(param app_role_password)
DC="docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.public.yml"
$DC config -q

echo "== 2. pull ${IMAGE##*/}"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${IMAGE%%/*}" >/dev/null
docker pull -q "$IMAGE"

echo "== 3. data stores on the low-memory overlay"
# The old overlay ran a traffic generator; it and its data go away here.
docker rm -f agentlens-demo-traffic-1 >/dev/null 2>&1 || true
$DC up -d --remove-orphans postgres redis clickhouse
for i in $(seq 1 40); do
  pg=$(docker inspect -f '{{.State.Health.Status}}' agentlens-postgres-1 2>/dev/null || true)
  ch=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:58123/ping || true)
  [ "$pg" = healthy ] && [ "$ch" = 200 ] && break
  sleep 3
done
echo "postgres=$pg clickhouse=$ch"

echo "== 4. schema revision check"
server_rev=$(docker exec agentlens-postgres-1 psql -U agentlens -d agentlens -Atc "select version_num from alembic_version")
bundle_rev=$(tr -d '\r\n' < "$WORK/pg/alembic_version.txt")
if [ "$server_rev" != "$bundle_rev" ]; then
  echo "FATAL: server schema $server_rev, bundle exported at $bundle_rev; not restoring"; exit 1
fi

echo "== 5. app on the new image (creates the ClickHouse schema if absent)"
$DC up -d app
for i in $(seq 1 40); do curl -s -m 3 http://localhost:8100/health | grep -q '"status"' && break; sleep 3; done

echo "== 6. Postgres: replace all tenant data in one transaction"
docker exec agentlens-postgres-1 rm -rf /tmp/restore
docker cp "$WORK/pg" agentlens-postgres-1:/tmp/restore
docker exec agentlens-postgres-1 psql -U agentlens -d agentlens -v ON_ERROR_STOP=1 -1 -q -f /tmp/restore/restore.sql
docker exec agentlens-postgres-1 rm -rf /tmp/restore

echo "== 7. ClickHouse: replace spans, keep them a year"
CH="docker exec -i agentlens-clickhouse-1 clickhouse-client --password $AGENTLENS_CLICKHOUSE_PASSWORD"
$CH -q "TRUNCATE TABLE agentlens.spans"
$CH -q "INSERT INTO agentlens.spans FORMAT Native" < "$WORK/spans.native"
$CH -q "ALTER TABLE agentlens.spans MODIFY TTL toDateTime(start_time) + INTERVAL 365 DAY"

echo "== 8. restart the app so no cache predates the restore"
$DC restart app >/dev/null
for i in $(seq 1 40); do curl -s -m 3 http://localhost:8100/health | grep -q '"status":"ok"' && break; sleep 3; done

echo "== 9. verify"
echo "health: $(curl -s -m 5 http://localhost:8100/health | cut -c1-200)"
echo "runs:   $(curl -s -m 10 'http://localhost:8100/v1/evals/runs?limit=100' | grep -o '"count":[0-9]*')"
echo "spans:  $($CH -q 'SELECT count() FROM agentlens.spans')"
echo "write without key -> $(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:8100/v1/flush)"
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}'
