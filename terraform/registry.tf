# ---------------------------------------------------------------------------
# Artifact Registry and Cloud Build
#
# masseuse-video-tee/cloudbuild.tee.yaml builds the producer image and the
# TEE image (workload/tee/Dockerfile.tee) into this repository by
# digest. The WIF condition requires the attested image_reference to be
# under this repository, so an image pushed anywhere else cannot read the
# weights even if it were somehow launched with the right digest.
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = "masseuse-video-tee"
  format        = "DOCKER"
  description   = "masseuse.ai video pipeline Confidential Space images"

  docker_config {
    immutable_tags = false
  }

  depends_on = [google_project_service.apis]
}

# Cloud Build runs as the project's Compute Engine default service account
# (the default for projects created after 2024-04); it needs to push here
# and to write its logs.
locals {
  cloud_build_sa = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"
}

resource "google_artifact_registry_repository_iam_member" "cloud_build_writes" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = local.cloud_build_sa
}

resource "google_project_iam_member" "cloud_build_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = local.cloud_build_sa
}

resource "google_project_iam_member" "cloud_build_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = local.cloud_build_sa
}

# Operators submit builds.
resource "google_project_iam_member" "operator_cloud_build" {
  for_each = toset(var.operator_members)
  project  = var.project_id
  role     = "roles/cloudbuild.builds.editor"
  member   = each.value
}
