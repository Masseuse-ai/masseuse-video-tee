output "slot_hosts" {
  description = "Slot hostname -> static IP; link-router/terraform/tee-dns.tf publishes these as DNS-only A records."
  value       = { for name in local.slot_names : local.slot_hosts[name] => google_compute_address.slot[name].address }
}

output "slot_origins" {
  description = "The slot origins the trainer (masseuse.ai's Cloud Run service) is configured with (POSE_PRODUCER_URLS)."
  value       = [for name in local.slot_names : "https://${local.slot_hosts[name]}"]
}

output "slot_vms" {
  description = "Each slot the trainer can start and stop on demand (the trainer's TEE_SLOT_VMS_JSON): its public origin keyed to the project, zone and instance name the trainer drives through the Compute API."
  value = [for name in local.slot_names : {
    origin   = "https://${local.slot_hosts[name]}"
    project  = var.project_id
    zone     = var.zone
    instance = google_compute_instance.slot[name].name
  }]
}

output "vm_service_account" {
  description = "Goes into the trainer's POSE_SIGNALS_INVOKER_SERVICE_ACCOUNT list: the slot posts readings as this identity."
  value       = google_service_account.vm.email
}

output "wif_audience" {
  description = "The external_account audience the entrypoint renders into the credential file."
  value       = local.wif_audience
}

output "wif_pool" {
  value = local.wif_pool_name
}

output "models_bucket" {
  value = google_storage_bucket.models.name
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "debug_mode" {
  value = var.debug_mode
}

output "acme_directory_url" {
  value = local.acme_directory_url
}

output "soak_service_account" {
  description = "Debug phase only: the identity the operator mints ID tokens for to drive a slot's control routes from a laptop (README, soak). null in production."
  value       = var.debug_mode ? google_service_account.soak[0].email : null
}
