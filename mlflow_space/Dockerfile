# =============================================================================
# Hugging Face Space — standalone MLflow tracking server
# =============================================================================
# Push this directory (this Dockerfile + README.md) as the root of a new
# Docker-SDK Space (huggingface.co/spaces/<user>/<space>). HF builds it,
# binds it to https://<user>-<space>.hf.space, and — as long as the Space
# visibility is set to Private — gates every request behind an HF auth
# token. No basic-auth app or extra code needed on the mlflow side for that.
#
# Required Space secrets (Settings → Variables and secrets):
#   BACKEND_STORE_URI     postgresql://user:pass@host/db  (e.g. a free Neon/
#                          Supabase Postgres — HF Spaces have no durable disk
#                          on the free tier, so SQLite would be wiped on rebuild)
#   S3_BUCKET              same bucket the EC2 pipeline already writes to
#   AWS_ACCESS_KEY_ID       a narrow IAM user (s3:GetObject/PutObject/List on
#   AWS_SECRET_ACCESS_KEY   that bucket only) — no instance role exists here
#   AWS_REGION
#
# EC2 side then points MLFLOW_TRACKING_URI at this Space's URL and sets
# MLFLOW_TRACKING_TOKEN to an HF access token with read access to the Space
# (mlflow's client picks it up and sends `Authorization: Bearer <token>`
# automatically — see docker-compose.ec2.yml).
# =============================================================================
FROM ghcr.io/mlflow/mlflow:v2.13.0

RUN pip install --no-cache-dir psycopg2-binary boto3

ENV MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com \
    GUNICORN_CMD_ARGS="--forwarded-allow-ips=*"

EXPOSE 7860

# Shell form so $BACKEND_STORE_URI / $S3_BUCKET expand at container start —
# these come from Space secrets, not build args, so they must stay runtime.
CMD mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "s3://$S3_BUCKET/mlflow" \
    --host 0.0.0.0 \
    --port 7860
