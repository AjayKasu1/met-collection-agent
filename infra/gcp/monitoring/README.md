# Cloud Run monitoring

This Terraform root installs three production alerts around the existing Cloud Run API:

- public `/ready` failure for two minutes
- a 5xx ratio above 5 percent for five minutes
- p95 request latency above 10 seconds for five minutes

It does not create or modify the Cloud Run service. The readiness route checks Qdrant and the configured audit store, so the uptime incident covers dependency outages as well as container failures.

Create and verify an email, SMS, PagerDuty, or webhook notification channel in Cloud Monitoring first. Then authenticate with Google Application Default Credentials and pass the real resource name. Keep local variable files out of Git.

```sh
gcloud auth application-default login
gcloud monitoring channels list --format='table(name,displayName,type,enabled)'
export MET_GCP_PROJECT="$(gcloud config get-value project)"
export MET_API_HOST="met-collection-agent-api-584674541487.us-east1.run.app"
read -r -p 'Verified notification channel resource name: ' MET_NOTIFICATION_CHANNEL
terraform init
terraform plan \
  -var="project_id=${MET_GCP_PROJECT}" \
  -var="api_hostname=${MET_API_HOST}" \
  -var="notification_channel_ids=[\"${MET_NOTIFICATION_CHANNEL}\"]"
terraform apply
unset MET_NOTIFICATION_CHANNEL
```

Do not commit a `.tfvars` file containing account identifiers or operational routing details. Run `terraform plan` after every change and retain the plan output in the change record used by the operator.
