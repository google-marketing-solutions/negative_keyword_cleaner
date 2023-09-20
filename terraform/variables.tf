variable "project_id" {
  type = string
}

variable "region" {
  type = string
  default = "europe-west1"
}

variable "appengine_location" {
  type = string
  default = "europe-west"
}

variable "google_oauth_client_id" {
  description = "Google OAuth Client Id"
}

variable "google_oauth_client_secret" {
  description = "Google OAuth Client Secret"
}
