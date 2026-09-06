output "uptime_check_id" {
  description = "Cloud Monitoring uptime-check identifier."
  value       = google_monitoring_uptime_check_config.api_readiness.uptime_check_id
}

output "alert_policy_names" {
  description = "Created alert policy resource names."
  value = {
    latency      = google_monitoring_alert_policy.api_latency.name
    server_error = google_monitoring_alert_policy.api_server_errors.name
    unavailable  = google_monitoring_alert_policy.api_unavailable.name
  }
}
