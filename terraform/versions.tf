terraform {
  # backend {
  #   gcs = "gs://"
  # }

  required_providers {
    archive = {
      source = "hashicorp/archive"
      version = "2.4.0"
    }

    google = {
      source = "hashicorp/google"
      version = "4.82.0"
    }

    random = {
      source = "hashicorp/random"
      version = "3.5.1"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
