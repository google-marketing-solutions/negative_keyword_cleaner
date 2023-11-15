resource "google_project_service" "cloudresourcemanager" {
  service = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

##
# Custom Service Account
#

resource "google_project_service" "iam" {
  service            = "iam.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

resource "google_service_account" "main" {
  account_id   = "neg-keywords-cleaner"
  display_name = "Negative Keywords Cleaner Service Account"

  depends_on = [google_project_service.iam]
}

resource "google_project_iam_member" "logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.main.email}"
}

resource "google_project_iam_member" "aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.main.email}"
}

##
# Vertex AI
#

resource "google_project_service" "aiplatform" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

resource "google_project_service" "apikeys" {
  service            = "apikeys.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

resource "random_id" "vertexai_apikey_suffix" {
  byte_length = 8
}

resource "google_apikeys_key" "vertexai" {
  name         = "negcleaner-palm2-${random_id.vertexai_apikey_suffix.hex}"
  display_name = "Negative Keywords Cleaner - PALM 2"
  project      = var.project_id

  restrictions {
    api_targets {
      service = "language.googleapis.com"
    }
  }

  depends_on = [google_project_service.apikeys]
}

##
# Google Ads
#

resource "google_project_service" "googleads" {
  service            = "googleads.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

##
# Cloud Run Deployment
#

resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

resource "google_cloud_run_v2_service" "default" {
  name     = "neg-kws-cleaner"
  location = "europe-west1"
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "gcr.io/${var.project_id}/negatives:v2"

      env {
        name = "port"
        value = "8080"
      }

      env {
        name = "OAUTH_CLIENT_ID"
        value = var.google_oauth_client_id
      }

      env {
        name = "OAUTH_CLIENT_SECRET"
        value = var.google_oauth_client_secret
      }

      env {
        name = "GOOGLE_VERTEXAI_API_KEY"
        value = google_apikeys_key.vertexai.key_string
      }
    }
  }

  traffic {
    type = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "null_resource" "env_var_update" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = "gcloud run services update ${google_cloud_run_v2_service.default.name} --update-env-vars=OAUTH_REDIRECT_URI=${google_cloud_run_v2_service.default.uri} --region=${google_cloud_run_v2_service.default.location} --platform=managed"
  }
  
  depends_on = [google_cloud_run_v2_service.default]
}
