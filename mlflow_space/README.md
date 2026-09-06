---
title: Nyc Taxi Mlflow
emoji: 🚕
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Standalone MLflow tracking server for the NYC Taxi MLOps pipeline.
Backend store: external Postgres (see Dockerfile header). Artifact store: S3.

**Set this Space's visibility to Private** — it holds experiment data and
needs the AWS/DB secrets below. Access from the EC2 pipeline is via an HF
token sent as a Bearer header (mlflow's `MLFLOW_TRACKING_TOKEN` does this
automatically), not a public login.

Required secrets: `BACKEND_STORE_URI`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.
