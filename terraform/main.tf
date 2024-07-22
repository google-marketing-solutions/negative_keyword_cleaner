resource "null_resource" "enable_cloud_apis" {
  provisioner "local-exec" {
    command = "gcloud services enable serviceusage.googleapis.com cloudresourcemanager.googleapis.com iam.googleapis.com --project ${var.project_id}"
  }
}

##
# Custom Service Account
#

resource "google_service_account" "main" {
  account_id   = "neg-keywords-cleaner"
  display_name = "Negative Keywords Cleaner Service Account"

  depends_on = [null_resource.enable_cloud_apis]
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
# Cloud Storage
#
resource "random_id" "bucket_main_suffix" {
  keepers = {
    # Generate a new id each time we switch to a new Project ID
    ami_id = var.project_id
  }
  byte_length = 8
}

resource "google_storage_bucket" "main" {
  name                        = "neg-kws-cleaner-${random_id.bucket_main_suffix.hex}"
  location                    = var.location
  storage_class               = "STANDARD"
  force_destroy               = true
  uniform_bucket_level_access = true
  depends_on = [null_resource.enable_cloud_apis]
}
resource "google_storage_bucket_iam_member" "member" {
  bucket = google_storage_bucket.main.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.main.email}"
}

##
# Vertex AI
#

resource "google_project_service" "aiplatform" {
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
  depends_on = [null_resource.enable_cloud_apis]
}

resource "google_project_service" "apikeys" {
  service            = "apikeys.googleapis.com"
  disable_on_destroy = false
  depends_on = [null_resource.enable_cloud_apis]
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
      service = "generativelanguage.googleapis.com"
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
  depends_on = [null_resource.enable_cloud_apis]
}

##
# Cloud Run Deployment
#

resource "google_project_service" "cloud_run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
  depends_on = [null_resource.enable_cloud_apis]
}

resource "google_cloud_run_v2_service" "default" {
  name     = "neg-kws-cleaner"
  location = var.location
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "gcr.io/${var.project_id}/negatives:v1"

      env {
        name  = "port"
        value = "8080"
      }

      env {
        name  = "OAUTH_CLIENT_ID"
        value = var.google_oauth_client_id
      }

      env {
        name  = "OAUTH_CLIENT_SECRET"
        value = var.google_oauth_client_secret
      }

      env {
        name  = "GOOGLE_VERTEXAI_API_KEY"
        value = google_apikeys_key.vertexai.key_string
      }

      env {
        name  = "DEFAULT_BUCKET_NAME"
        value = google_storage_bucket.main.name
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "8Gi"
        }
      }
    }
    timeout          = "1800s"
    service_account  = google_service_account.main.email
    session_affinity = true
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

resource "google_cloud_run_v2_service_iam_policy" "policy" {
  project  = var.project_id
  location = var.location
  name     = google_cloud_run_v2_service.default.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

resource "null_resource" "env_var_update" {
  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = "gcloud run services update ${google_cloud_run_v2_service.default.name} --update-env-vars=OAUTH_REDIRECT_URI=${google_cloud_run_v2_service.default.uri} --region=${google_cloud_run_v2_service.default.location} --platform=managed"
  }

  depends_on = [google_cloud_run_v2_service.default]
}