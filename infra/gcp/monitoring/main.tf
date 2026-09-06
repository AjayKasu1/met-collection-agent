resource "google_project_service" "monitoring" {
  project            = var.project_id
  service            = "monitoring.googleapis.com"
  disable_on_destroy = false
}

resource "google_monitoring_uptime_check_config" "api_readiness" {
  project      = var.project_id
  display_name = "Met collection API readiness"
  period       = "60s"
  timeout      = "10s"
  checker_type = "STATIC_IP_CHECKERS"

  http_check {
    path           = "/ready"
    port           = 443
    request_method = "GET"
    use_ssl        = true
    validate_ssl   = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      host       = var.api_hostname
      project_id = var.project_id
    }
  }

  depends_on = [google_project_service.monitoring]
}

resource "google_monitoring_alert_policy" "api_unavailable" {
  project      = var.project_id
  display_name = "Met collection API is unavailable"
  combiner     = "OR"
  enabled      = true

  documentation {
    content = "The public readiness check failed. Follow docs/operations.md to identify whether Cloud Run, Qdrant, or PostgreSQL is unavailable."
  }

  conditions {
    display_name = "Readiness failed for two minutes"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\"",
        "resource.type=\"uptime_url\"",
        "metric.label.check_id=\"${google_monitoring_uptime_check_config.api_readiness.uptime_check_id}\"",
      ])
      comparison      = "COMPARISON_LT"
      threshold_value = 1
      duration        = "120s"

      aggregations {
        alignment_period   = "120s"
        per_series_aligner = "ALIGN_NEXT_OLDER"
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channel_ids
  depends_on            = [google_project_service.monitoring]
}

resource "google_monitoring_alert_policy" "api_server_errors" {
  project      = var.project_id
  display_name = "Met collection API elevated server errors"
  combiner     = "OR"
  enabled      = true

  documentation {
    content = "More than the configured fraction of Cloud Run requests returned 5xx for five minutes. Follow docs/operations.md and correlate by X-Request-ID."
  }

  conditions {
    display_name = "Five-minute 5xx ratio"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"run.googleapis.com/request_count\"",
        "resource.type=\"cloud_run_revision\"",
        "resource.label.service_name=\"${var.service_name}\"",
        "metric.label.response_code_class=\"5xx\"",
      ])
      denominator_filter = join(" AND ", [
        "metric.type=\"run.googleapis.com/request_count\"",
        "resource.type=\"cloud_run_revision\"",
        "resource.label.service_name=\"${var.service_name}\"",
      ])
      comparison      = "COMPARISON_GT"
      threshold_value = var.server_error_ratio
      duration        = "300s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }

      denominator_aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channel_ids
  depends_on            = [google_project_service.monitoring]
}

resource "google_monitoring_alert_policy" "api_latency" {
  project      = var.project_id
  display_name = "Met collection API elevated latency"
  combiner     = "OR"
  enabled      = true

  documentation {
    content = "Cloud Run p95 request latency exceeded the configured threshold for five minutes. Inspect model retries, provider rate limits, and dependency latency in docs/operations.md."
  }

  conditions {
    display_name = "Five-minute p95 latency"

    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"run.googleapis.com/request_latencies\"",
        "resource.type=\"cloud_run_revision\"",
        "resource.label.service_name=\"${var.service_name}\"",
      ])
      comparison      = "COMPARISON_GT"
      threshold_value = var.p95_latency_ms
      duration        = "300s"

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MAX"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }

  notification_channels = var.notification_channel_ids
  depends_on            = [google_project_service.monitoring]
}
