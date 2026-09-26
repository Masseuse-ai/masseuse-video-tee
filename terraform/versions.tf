terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    # sweeper.tf: google_project_service_identity (the Workflows service
    # agent has to exist before the first workflow is created).
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
    # signing.tf: the signing key's fingerprint (sha256 over DER) via
    # openssl, which Terraform's string functions cannot do on binary.
    external = {
      source  = "hashicorp/external"
      version = "~> 2.3"
    }
  }

  # Bootstrapped once, by hand (README.md):
  #   gcloud storage buckets create gs://prod-masseuse-video-tee-tf-state \
  #       --project prod-masseuse-video-tee --location us-central1 \
  #       --uniform-bucket-level-access --public-access-prevention
  #   gcloud storage buckets update gs://prod-masseuse-video-tee-tf-state --versioning
  backend "gcs" {
    bucket = "prod-masseuse-video-tee-tf-state"
    prefix = "masseuse-video-tee"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}
