output "app_url" {
  value = google_cloud_run_v2_service.default.uri
  description = "The URL of the deployed Cloud Run service"
}
