# ---------------------------------------------------------------------------
# Slots hosted in other projects
#
# A released image can boot in someone else's Google Cloud project
# (modules/slot, deploy/own-project) and be leased by a controller other
# than the trainer. What it needs from this project is what a slot of this
# project gets: the weights and the analysis bundle from the models bucket,
# and the image and its release signature from Artifact Registry. This file
# grants them through a pool of its own, separately revocable, only for the
# projects an operator approved (external_slot_projects, in the git-ignored
# terraform.tfvars, with the controller each was approved for), and only to
# a slot that attests what one of this project's does: production posture
# on the STABLE image, TDX with secure boot, the H100 in CC mode, the
# release signature; plus an approved project and VM account, that
# controller's address and identity, and no record bucket. Nothing here
# exists until the first approval.
#
# Removing a project from the list refuses its next exchange; a credential
# already exchanged lives out its hour. The pool and its provider outlive
# the list (hosted_slots_pool): a deleted pool or provider keeps its name
# for 30 days, so deleting them with the last project would fail the next
# approval with a 409. With no project listed, the provider admits no slot
# and every grant below is gone. The weights a hosted slot reads are
# the ones models.lock and analysis.lock pin, like any slot's; what keeps a
# session from the host project's owners is the same as here: the clients
# hold the slot to the release's public record, pin the attested TLS key,
# seal to the attested seal key, and hold every reading to the attested
# evidence key (X-Masseuse-Signature).
# ---------------------------------------------------------------------------
locals {
  hosted_enabled = length(var.external_slot_projects) > 0
  hosted_pool_on = var.hosted_slots_pool || local.hosted_enabled

  hosted_pool_name = local.hosted_pool_on ? "projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.hosted[0].workload_identity_pool_id}" : ""

  # The digests a hosted slot may run: the fleet's, and releases approved
  # for hosted slots before the fleet rolls to them.
  hosted_digests = distinct(concat([var.container_image_digest], var.candidate_image_digests, var.external_image_digests))

  # The image a hosted deployment is given: the last release approved for
  # hosted slots, else the fleet's.
  hosted_container_image = length(var.external_image_digests) > 0 ? "${local.image_signature_repo}@${var.external_image_digests[length(var.external_image_digests) - 1]}" : var.container_image
  hosted_principals = local.hosted_enabled ? {
    for digest in local.hosted_digests :
    digest => "principalSet://iam.googleapis.com/${local.hosted_pool_name}/attribute.image_digest/${digest}"
  } : {}
}

resource "google_iam_workload_identity_pool" "hosted" {
  count                     = local.hosted_pool_on ? 1 : 0
  project                   = var.project_id
  workload_identity_pool_id = "hosted-slots"
  display_name              = "Slots hosted elsewhere"
  description               = "Confidential Space attestation for released slots in approved projects"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "hosted" {
  count                              = local.hosted_pool_on ? 1 : 0
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.hosted[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "attestation-hosted"
  display_name                       = "CS attestation: hosted elsewhere"

  oidc {
    issuer_uri        = "https://confidentialcomputing.googleapis.com/"
    allowed_audiences = ["https://sts.googleapis.com"]
  }

  attribute_mapping = local.wif_attribute_mapping

  attribute_condition = join(" && ", [
    "assertion.swname == 'CONFIDENTIAL_SPACE'",
    "assertion.hwmodel == 'GCP_INTEL_TDX'",
    "assertion.secboot == true",
    "'STABLE' in assertion.submods.confidential_space.support_attributes",
    "assertion.dbgstat == 'disabled-since-boot'",
    "assertion.submods.nvidia_gpu.cc_mode == 'ON'",
    "assertion.submods.nvidia_gpu.gpus.exists(gpu, gpu.hwmodel == 'GCP_NVIDIA_H100')",
    local.wif_signature_condition,
    "assertion.submods.gce.project_id in ${jsonencode(var.external_slot_projects)}",
    "assertion.google_service_accounts.exists(sa, sa in ${jsonencode(var.external_slot_service_accounts)})",
    "assertion.submods.container.env.TRAINER_URL == '${var.external_controller_url}'",
    "assertion.submods.container.env.TRAINER_INVOKER_SERVICE_ACCOUNT == '${var.external_controller_service_account}'",
    "(!('TEE_CAPTURE_BUCKET' in assertion.submods.container.env) || assertion.submods.container.env.TEE_CAPTURE_BUCKET == '')",
  ])

  lifecycle {
    precondition {
      condition     = !local.hosted_enabled || (var.external_controller_url != "" && var.external_controller_service_account != "" && length(var.external_slot_service_accounts) > 0)
      error_message = "Approving a project needs the controller it was approved for (external_controller_url, external_controller_service_account) and its slots' VM accounts (external_slot_service_accounts)."
    }
  }
}

resource "google_storage_bucket_iam_member" "models_hosted_reader" {
  for_each = local.hosted_principals
  bucket   = google_storage_bucket.models.name
  role     = "roles/storage.objectViewer"
  member   = each.value
}

# The image and its signature, by digest, for the launcher of a hosted slot.
resource "google_artifact_registry_repository_iam_member" "hosted_pulls" {
  for_each   = toset(var.external_slot_service_accounts)
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${each.value}"
}

output "hosted_slots" {
  description = "What a hosted slot's deployment (modules/slot) takes from this project; null while no project is approved."
  value = local.hosted_enabled ? {
    wif_audience      = "//iam.googleapis.com/${local.hosted_pool_name}/providers/${google_iam_workload_identity_pool_provider.hosted[0].workload_identity_pool_provider_id}"
    models_bucket     = google_storage_bucket.models.name
    container_image   = local.hosted_container_image
    signature_repo    = local.image_signature_repo
    digests           = local.hosted_digests
    hosted_principals = values(local.hosted_principals)
    signer_key_ids    = [for s in local.accepted_image_signers : split(":", s)[1]]
  } : null
}
