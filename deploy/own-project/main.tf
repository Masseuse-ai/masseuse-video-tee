terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
}

# From a checkout of this repository at a release tag. From anywhere else:
#   source = "github.com/Masseuse-ai/masseuse-video-tee//terraform/modules/slot?ref=<the release tag>"
module "slots" {
  source = "../../terraform/modules/slot"

  project_id  = var.project_id
  slots       = var.slots
  network     = var.network
  subnetworks = var.subnetworks

  container_image            = var.container_image
  signature_repo             = var.signature_repo
  controller_url             = var.controller_url
  controller_service_account = var.controller_service_account
  models_bucket              = var.models_bucket
  wif_audience               = var.wif_audience
  hosted_principals          = var.hosted_principals
  acme_contact_email         = var.acme_contact_email
  idle_exit_s                = var.idle_exit_s
}

output "slots" {
  description = "Point each slot's DNS name at its address."
  value       = module.slots.slots
}

output "vm_service_account" {
  value = module.slots.vm_service_account
}

output "verifier" {
  description = "tee-verify's values for these slots."
  value       = module.slots.verifier
}
