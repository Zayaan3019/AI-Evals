#!/bin/bash
# Publish the local stack's public tenant to the EC2 host, on a freshly built
# app image. Run from the repository root with the local docker stack up and
# AWS credentials for the account:
#
#   bash deploy/publish-public-instance.sh
#
# 1. build platform/Dockerfile, push it to ECR tagged with the commit
# 2. export the tenant (deploy/export-public-data.py)
# 3. upload the bundle to a private S3 key; hand the host a 15-minute URL
# 4. run deploy/restore-public-data.sh on the host through SSM Run Command
# 5. delete the uploaded bundle
set -euo pipefail

REGION=us-east-1
TENANT=${TENANT:-public}
INSTANCE_ID=${INSTANCE_ID:-$(aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Name,Values=ai-evals-host Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)}
REPO=$(aws ecr describe-repositories --region $REGION --repository-names ai-evals-platform \
  --query 'repositories[0].repositoryUri' --output text)
BUCKET=${BUCKET:-ai-evals-frontend-$(aws sts get-caller-identity --query Account --output text)}
TAG=$(git rev-parse --short=12 HEAD)
[ -z "$(git status --porcelain -- platform sdk)" ] || { echo "platform/ or sdk/ has uncommitted changes; commit first so the tag means something"; exit 1; }

echo "== 1. image $REPO:$TAG"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "${REPO%%/*}" >/dev/null
docker build -q -f platform/Dockerfile -t "$REPO:$TAG" -t "$REPO:latest" .
docker push -q "$REPO:$TAG" && docker push -q "$REPO:latest"

echo "== 2. export tenant $TENANT"
BUNDLE=$(mktemp -d)/public-bundle.tgz
python deploy/export-public-data.py --tenant "$TENANT" --out "$BUNDLE"

echo "== 3. upload"
KEY="transfer/public-bundle-$TAG-$(date +%s).tgz"
aws s3 cp --only-show-errors "$BUNDLE" "s3://$BUCKET/$KEY"
URL=$(aws s3 presign "s3://$BUCKET/$KEY" --expires-in 900 --region $REGION)
trap 'aws s3 rm --only-show-errors "s3://$BUCKET/$KEY" || true' EXIT

echo "== 4. restore on $INSTANCE_ID"
SCRIPT=$(base64 -w0 deploy/restore-public-data.sh)
PARAMS=$(python -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": ["900"]}))' \
  "echo $SCRIPT | base64 -d > /tmp/restore-public-data.sh && IMAGE='$REPO:$TAG' BUNDLE_URL='$URL' bash /tmp/restore-public-data.sh 2>&1")
CID=$(aws ssm send-command --region $REGION --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --comment "agentlens public restore $TAG" --parameters "$PARAMS" --query Command.CommandId --output text)
echo "command $CID"
for i in $(seq 1 120); do
  STATUS=$(aws ssm get-command-invocation --region $REGION --command-id "$CID" --instance-id "$INSTANCE_ID" \
    --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
  sleep 5
done
aws ssm get-command-invocation --region $REGION --command-id "$CID" --instance-id "$INSTANCE_ID" \
  --query StandardOutputContent --output text
echo "status: $STATUS"
[ "$STATUS" = Success ]
