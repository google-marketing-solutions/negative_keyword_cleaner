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

resource "google_project_iam_member" "gae_api" {
  project = var.project_id
  role    = "roles/compute.networkUser"
  member  = "serviceAccount:${google_service_account.main.email}"
}

resource "google_project_iam_member" "logs_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.main.email}"
}

resource "google_project_iam_member" "storage_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
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
# App Engine Deployment
#

resource "google_project_service" "appengine" {
  service = "appengine.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

resource "google_project_service" "appengineflex" {
  service = "appengineflex.googleapis.com"
  disable_on_destroy = false
  depends_on         = [google_project_service.cloudresourcemanager]
}

data "google_app_engine_default_service_account" "default" {
    depends_on = [google_app_engine_flexible_app_version.negcleaner]
}

resource "google_project_iam_member" "appspot_storage_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${data.google_app_engine_default_service_account.default.email}"
}

resource "random_id" "bucket_main_suffix" {
  keepers = {
    # Generate a new id each time we switch to a new Project ID
    ami_id = var.project_id
  }

  byte_length = 8
}

resource "google_storage_bucket" "main" {
  name          = "neg-kws-cleaner-${random_id.bucket_main_suffix.hex}"
  location      = "EU"
  storage_class = "STANDARD"
  force_destroy = true

  uniform_bucket_level_access = true

  depends_on = [google_project_service.cloudresourcemanager]
}

resource "google_storage_bucket_iam_member" "member" {
  bucket = google_storage_bucket.main.name
  role = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.main.email}"
}

locals {
  app_hostname = "${google_app_engine_application.main.default_hostname}"
}

data "archive_file" "app" {
  type             = "zip"
  source_dir       = "${path.module}/../"
  output_file_mode = "0666"
  output_path      = "${path.module}/files/app.zip"
  excludes         = [
    ".venv",
    ".vscode",
    "terraform",
    "tests",
    "app_config.yaml",
  ]
}

resource "google_storage_bucket" "deployments" {
 name          = "neg-kws-cleaner-deployment-${random_id.bucket_main_suffix.hex}"
 location      = "EU"
 storage_class = "STANDARD"

 uniform_bucket_level_access = true
}

# Uploads the `app.zip` file.
resource "google_storage_bucket_object" "app_zip" {
 name         = "app.zip"
 source       = data.archive_file.app.output_path
 content_type = "text/plain"
 bucket       = google_storage_bucket.deployments.id
}

resource "google_app_engine_application" "main" {
  project     = var.project_id
  location_id = var.appengine_location

  depends_on = [google_project_service.appengine]
}

resource "google_app_engine_flexible_app_version" "negcleaner" {
  version_id      = "v1"
  service         = "default"
  service_account = google_service_account.main.email

  # See: https://cloud.google.com/appengine/docs/flexible/python/runtime#newversions
  runtime         = "custom"

  deployment {
    container {
      image = ""  # HACK: to workaround the bug https://github.com/hashicorp/terraform-provider-google/issues/10185
    }
    zip {
      source_url = "https://storage.googleapis.com/${google_storage_bucket_object.app_zip.bucket}/${google_storage_bucket_object.app_zip.name}"
    }

    cloud_build_options {
      app_yaml_path       = "./app.yaml"
      cloud_build_timeout = "1200s"  # 20min
    }
  }

  liveness_check {
    path = "/"
  }

  readiness_check {
    path = "/"
  }

  env_variables = {
    port = "8080"
    OAUTH_CLIENT_ID = var.google_oauth_client_id
    OAUTH_CLIENT_SECRET = var.google_oauth_client_secret
    OAUTH_REDIRECT_URI = "https://${local.app_hostname}/"
    GOOGLE_VERTEXAI_API_KEY = google_apikeys_key.vertexai.key_string
  }

  # handlers {
  #   url_regex        = ".*"
  #   security_level   = "SECURE_ALWAYS"
  #   login            = "LOGIN_REQUIRED"
  #   auth_fail_action = "AUTH_FAIL_ACTION_REDIRECT"

  #   # static_files {
  #   #   path = "my-other-path"
  #   #   upload_path_regex = ".*\\/my-path\\/*"
  #   # }
  # }

  automatic_scaling {
    cool_down_period = "120s"
    min_total_instances = 1
    max_total_instances = 1

    cpu_utilization {
      target_utilization = 0.5
    }
  }

  noop_on_destroy = true
  depends_on = [google_project_service.appengineflex]
}

# resource "google_app_engine_service_split_traffic" "main" {
#   service = google_app_engine_flexible_app_version.negcleaner.service

#   migrate_traffic = false
#   split {
#     shard_by = "UNSPECIFIED"
#     allocations = {
#       (google_app_engine_flexible_app_version.negcleaner.version_id) = 1.0
#     }
#   }
# }
