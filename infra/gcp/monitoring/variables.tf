variable "project_id" {
  description = "Google Cloud project containing the Cloud Run service."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "project_id cannot be empty."
  }
}

variable "region" {
  description = "Cloud Run service region."
  type        = string
  default     = "us-east1"
}

variable "service_name" {
  description = "Cloud Run service monitored by metric alerts."
  type        = string
  default     = "met-collection-agent-api"
}

variable "api_hostname" {
  description = "Public Cloud Run hostname without a scheme or path."
  type        = string

  validation {
    condition = (
      length(trimspace(var.api_hostname)) > 0 &&
      !strcontains(var.api_hostname, "://") &&
      !strcontains(var.api_hostname, "/")
    )
    error_message = "api_hostname must be a hostname without a scheme or path."
  }
}

variable "notification_channel_ids" {
  description = "Verified Cloud Monitoring channel resource names that receive incidents."
  type        = list(string)

  validation {
    condition = (
      length(var.notification_channel_ids) > 0 &&
      alltrue([
        for id in var.notification_channel_ids :
        startswith(id, "projects/${var.project_id}/notificationChannels/")
      ])
    )
    error_message = "Provide at least one notification channel from the selected project."
  }
}

variable "server_error_ratio" {
  description = "Five-minute 5xx ratio that opens an incident."
  type        = number
  default     = 0.05

  validation {
    condition     = var.server_error_ratio > 0 && var.server_error_ratio < 1
    error_message = "server_error_ratio must be between zero and one."
  }
}

variable "p95_latency_ms" {
  description = "Five-minute p95 request latency that opens an incident."
  type        = number
  default     = 10000

  validation {
    condition     = var.p95_latency_ms >= 1000
    error_message = "p95_latency_ms must be at least 1000 milliseconds."
  }
}
