output "slots" {
  description = "Each slot's zone, the address its DNS name must resolve to, and its origin."
  value = {
    for name, slot in var.slots : name => {
      zone     = slot.zone
      address  = google_compute_address.slot[name].address
      origin   = "https://${slot.host}"
      instance = google_compute_instance.slot[name].self_link
    }
  }
}

output "vm_service_account" {
  description = "The account the slots run as: the operator approves it with the project (external_slot_service_accounts)."
  value       = google_service_account.vm.email
}

output "verifier" {
  description = "What these slots attest about where they run: tee-verify's values for them (VERIFY.md, \"Which values to verify against\"), as a -policy-url document takes them."
  value = {
    projectIds       = [var.project_id]
    imageRefPrefixes = [local.image_ref_prefix]
    readingsUrl      = var.controller_url
    invokers         = [var.controller_service_account]
    captureBucket    = ""
    requestSeal      = "v1"
  }
}
