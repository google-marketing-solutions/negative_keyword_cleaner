variable "project_id" {
  type = string
}

variable "location" {
  type = string
  default = "europe-west1"
}

variable "google_oauth_client_id" {
  description = "Google OAuth Client Id"
}

variable "google_oauth_client_secret" {
  description = "Google OAuth Client Secret"
}
