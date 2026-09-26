# ---------------------------------------------------------------------------
# The trainer's power over a slot
#
# A slot VM is started when a masseuse.ai visitor taps Enable camera and
# stopped when the session ends (the trainer, masseuse.ai's Cloud Run
# service, drives the Compute API). The
# trainer's runtime service account is the one identity that may do so, and
# only that: a custom role with get, start and stop, bound per instance -
# not project-wide - so the trainer can turn the slots on and off and touch
# nothing else in the project (it cannot read the models bucket, cannot
# reach the enclave, cannot change the VM). Starting a VM that runs as a
# service account also needs actAs on that account, granted on the VM's SA
# alone. The trainer polls instances.get for status rather than the
# zone operation, so no zoneOperations.get is needed.
# ---------------------------------------------------------------------------
resource "google_project_iam_custom_role" "slot_operator" {
  project     = var.project_id
  role_id     = "masseuseVideoTeeSlotOperator"
  title       = "masseuse-video-tee slot operator"
  description = "Get, start and stop a slot VM; nothing else. Bound per instance to the trainer."
  permissions = [
    "compute.instances.get",
    "compute.instances.start",
    "compute.instances.stop",
  ]
}

resource "google_compute_instance_iam_member" "trainer_operates_slot" {
  for_each      = google_compute_instance.slot
  project       = var.project_id
  zone          = var.zone
  instance_name = each.value.name
  role          = google_project_iam_custom_role.slot_operator.id
  member        = "serviceAccount:${var.trainer_service_account}"
}

# actAs on the VM's own service account: starting an instance that runs as
# a service account requires it (iam/docs/service-accounts-actas). Scoped
# to this one SA, so it is not permission to run anything else as it.
resource "google_service_account_iam_member" "trainer_acts_as_vm_sa" {
  service_account_id = google_service_account.vm.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.trainer_service_account}"
}
