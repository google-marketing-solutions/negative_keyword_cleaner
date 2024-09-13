variable "project_id" {
  type = string
}

variable "location" {
  type    = string
  default = "europe-west1"
}

variable "google_oauth_client_id" {
  description = "Google OAuth Client Id"
}

variable "google_oauth_client_secret" {
  description = "Google OAuth Client Secret"
}

variable "mcc_id" {
  description = "Google Ads MCC account ID. Set it without hyphens XXXXXXXXXX"
}

variable "google_ads_api_token" {
  description = "The developer token from Google Ads"
}

variable "openai_api_key" {
  description = "The API Key from OpenAI (optional)"
}


