# ---------------------------------------------------------------------------
# Workload Identity Federation: the attestation token becomes a credential
#
# The enclave's launcher writes a Google Cloud Attestation OIDC token
# (audience https://sts.googleapis.com) to
# /run/container_launcher/attestation_verifier_claims_token; the workload's
# external_account credential (workload/tee/wif-credential.json.tmpl)
# exchanges it here. Only tokens that satisfy a provider's condition are
# exchanged, and the models bucket grants read to the pool principal whose
# attribute.image_digest is the deployed digest - so the weights can be
# read by exactly one image, on exactly this hardware, and by nobody at
# the console.
#
# Claim paths below were read off a real token on the Phase 0 spike VM
# (masseuse-video-tee/README.md, "Phase 0 findings"):
#   hwmodel                          "GCP_INTEL_TDX"
#   dbgstat                          "enabled" (debug image) / "disabled-since-boot" (production)
#   submods.nvidia_gpu.cc_mode       "ON"
#   submods.nvidia_gpu.gpus[].hwmodel "GCP_NVIDIA_H100"
#   submods.confidential_space.support_attributes  absent on the debug image,
#                                    ["LATEST","STABLE","USABLE"] on production
# ---------------------------------------------------------------------------
resource "google_iam_workload_identity_pool" "tee" {
  project                   = var.project_id
  workload_identity_pool_id = "masseuse-video-tee"
  display_name              = "masseuse video TEE"
  description               = "Confidential Space attestation for the video pipeline slots"

  depends_on = [google_project_service.apis]
}

locals {
  wif_attribute_mapping = {
    "google.subject"            = "\"gcpcs::\"+assertion.submods.container.image_digest+\"::\"+assertion.submods.gce.project_number+\"::\"+assertion.submods.gce.instance_id"
    "attribute.image_digest"    = "assertion.submods.container.image_digest"
    "attribute.instance_id"     = "assertion.submods.gce.instance_id"
    "attribute.gpu_cc_mode"     = "assertion.submods.nvidia_gpu.cc_mode"
    "attribute.debug_status"    = "assertion.dbgstat"
    "attribute.image_reference" = "assertion.submods.container.image_reference"
  }

  wif_base_conditions = [
    "assertion.swname == 'CONFIDENTIAL_SPACE'",
    "assertion.hwmodel == 'GCP_INTEL_TDX'",
    "assertion.secboot == true",
    "assertion.submods.gce.project_id == '${var.project_id}'",
    "'${google_service_account.vm.email}' in assertion.google_service_accounts",
    "assertion.submods.nvidia_gpu.cc_mode == 'ON'",
    "assertion.submods.nvidia_gpu.gpus.exists(gpu, gpu.hwmodel == 'GCP_NVIDIA_H100')",
    "assertion.submods.container.image_reference.startsWith('${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/')",
  ]
}

# Debug phase: the confidential-space-debug image (dbgstat == 'enabled',
# no support_attributes claim). Exists only while debug_mode is true; the
# production flip destroys it, after which no debug enclave can read the
# weights.
resource "google_iam_workload_identity_pool_provider" "attestation_debug" {
  count                              = var.debug_mode ? 1 : 0
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.tee.workload_identity_pool_id
  workload_identity_pool_provider_id = "attestation-debug"
  display_name                       = "CS attestation: debug image"

  oidc {
    issuer_uri        = "https://confidentialcomputing.googleapis.com/"
    allowed_audiences = ["https://sts.googleapis.com"]
  }

  attribute_mapping = local.wif_attribute_mapping

  attribute_condition = join(" && ", concat(local.wif_base_conditions, [
    "assertion.dbgstat == 'enabled'",
  ]))
}

# Production: the STABLE confidential-space image with debugging disabled
# since boot - the launcher enforces that no SSH, no log redirect and no
# monitoring were ever possible on this VM.
resource "google_iam_workload_identity_pool_provider" "attestation_prod" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.tee.workload_identity_pool_id
  workload_identity_pool_provider_id = "attestation-prod"
  display_name                       = "CS attestation: production"

  oidc {
    issuer_uri        = "https://confidentialcomputing.googleapis.com/"
    allowed_audiences = ["https://sts.googleapis.com"]
  }

  attribute_mapping = local.wif_attribute_mapping

  attribute_condition = join(" && ", concat(local.wif_base_conditions, [
    "'STABLE' in assertion.submods.confidential_space.support_attributes",
    "assertion.dbgstat == 'disabled-since-boot'",
    ], var.require_signed_image ? [
    # signing.tf: some accepted key signed the running image.
    local.wif_signature_condition,
  ] : []))
}

locals {
  wif_pool_name = "projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.tee.workload_identity_pool_id}"

  # The provider the VM's credential file names: debug while debugging.
  wif_provider_id = var.debug_mode ? google_iam_workload_identity_pool_provider.attestation_debug[0].workload_identity_pool_provider_id : google_iam_workload_identity_pool_provider.attestation_prod.workload_identity_pool_provider_id
  wif_audience    = "//iam.googleapis.com/${local.wif_pool_name}/providers/${local.wif_provider_id}"

  # Pool principals by image digest: the deployed one and the candidates.
  wif_principals = {
    for digest in distinct(concat([var.container_image_digest], var.candidate_image_digests)) :
    digest => "principalSet://iam.googleapis.com/${local.wif_pool_name}/attribute.image_digest/${digest}"
  }
}
