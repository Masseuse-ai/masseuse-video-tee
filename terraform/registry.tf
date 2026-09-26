# ---------------------------------------------------------------------------
# Artifact Registry
#
# The release workflow (.github/workflows/release.yml) builds the TEE image
# (workload/tee/Dockerfile.tee) on GitHub, publishes it to ghcr.io with
# SLSA provenance, and its promote job copies it by digest into this
# repository and signs it there (github-release.tf, signing.tf). Nothing
# else writes here: the only writer is the github-release service account.
# The WIF condition requires the attested image_reference to be under this
# repository, so an image pushed anywhere else cannot read the weights even
# if it were somehow launched with the right digest.
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

# The Compute Engine default service account, which Cloud Build ran as
# while it built the images (before release.yml), holds no grant here any
# more: not writer on this repository, not signer on the key, no project
# roles. The VM runs as its own account (main.tf) with reader only.
