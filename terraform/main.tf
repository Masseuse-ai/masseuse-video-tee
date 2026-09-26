# ============================================================================
# masseuse-video-tee: the Confidential Space slot project
# ============================================================================
#
# prod-masseuse-video-tee holds exactly what the video pipeline's enclave
# needs and nothing that can read it: the VM service account (can pull the
# image, ask for attestation, and - debug only - write logs), the WIF pool
# whose providers admit this image on TDX with the GPU in CC mode, the
# models bucket only that federated identity may read, the registry, the
# static IP, the firewall, and the a3-highgpu-1g Spot VM itself.
#
# The trainer and Cloudflare stay in prod-femled-couple-router; the phone
# talks to this project's VM directly (tee-dns.tf in link-router/terraform
# points slot-N.tee.masseuse.ai at the static IP, DNS-only).
# ============================================================================

data "google_project" "current" {
  project_id = var.project_id
}

locals {
  slot_names = [for i in range(var.slot_count) : "slot-${i}"]
  slot_hosts = { for name in local.slot_names : name => "${name}.${var.slot_domain}" }

  # Google Trust Services through Cloud Public CA, in both modes: the slot
  # boots once per session and mints a fresh certificate each time, which
  # Let's Encrypt's five-a-week-per-name limit cannot take and GTS's
  # per-project request quotas can (README "Certificate authority"). The
  # production directory even while debugging, so the phone sees a publicly
  # trusted chain; the enclave mints the EAB the directory demands at boot
  # with its attested identity (acme.tf, tee/entrypoint.sh). The staging
  # directory (https://dv.acme-v02.test-api.pki.goog/directory) is a plain
  # override; the entrypoint picks the matching EAB endpoint from the host.
  acme_directory_url = var.acme_directory_url != "" ? var.acme_directory_url : "https://dv.acme-v02.api.pki.goog/directory"

  cs_image_family = var.debug_mode ? "confidential-space-debug" : "confidential-space"
}

# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "confidentialcomputing.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "cloudkms.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "serviceusage.googleapis.com",
    # acme.tf: the enclave mints its ACME External Account Binding here.
    "publicca.googleapis.com",
    # sweeper.tf
    "workflows.googleapis.com",
    "workflowexecutions.googleapis.com",
    "cloudscheduler.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# The VM's service account
#
# Attached to the VM. It can pull the image, obtain attestation tokens and
# (debug phase only) write the container's redirected stdout to Cloud
# Logging. It has no access to the models bucket: that goes to the WIF
# federated identity below, which exists only for an attested image digest.
# ---------------------------------------------------------------------------
resource "google_service_account" "vm" {
  project      = var.project_id
  account_id   = "masseuse-video-tee-vm"
  display_name = "masseuse video TEE slot VM (Confidential Space)"
}

resource "google_project_iam_member" "vm_confidential_workload" {
  project = var.project_id
  role    = "roles/confidentialcomputing.workloadUser"
  member  = "serviceAccount:${google_service_account.vm.email}"
}

resource "google_artifact_registry_repository_iam_member" "vm_pulls" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.vm.email}"
}

# Debug phase only: with log_redirect=debugonly on the image, the production
# image never emits anything for this role to write.
resource "google_project_iam_member" "vm_log_writer" {
  count   = var.debug_mode ? 1 : 0
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.vm.email}"
}

resource "google_project_iam_member" "vm_metric_writer" {
  count   = var.debug_mode ? 1 : 0
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.vm.email}"
}

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
resource "google_project_iam_member" "operator_iap_ssh" {
  for_each = var.debug_mode ? toset(var.operator_members) : toset([])
  project  = var.project_id
  role     = "roles/iap.tunnelResourceAccessor"
  member   = each.value
}

# Debug phase only: an identity the operator can mint ID tokens for, listed
# as a trainer invoker on the slot, so the soak can drive the control
# routes (/lease, /produce, /status) from a laptop through the IAP tunnel
# before the trainer can reach the slot. Gone with debug_mode, along with
# its place in TRAINER_INVOKER_SERVICE_ACCOUNT (vm.tf), so in production
# only the trainer's own service account is heard.
resource "google_service_account" "soak" {
  count        = var.debug_mode ? 1 : 0
  project      = var.project_id
  account_id   = "masseuse-video-tee-soak"
  display_name = "masseuse video TEE soak driver (debug phase only)"
}

resource "google_service_account_iam_member" "operator_mints_soak_tokens" {
  for_each           = var.debug_mode ? toset(var.operator_members) : toset([])
  service_account_id = google_service_account.soak[0].name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = each.value
}

# ---------------------------------------------------------------------------
# Audit: every STS exchange (the weights read) and IAM change is logged.
# ---------------------------------------------------------------------------
resource "google_project_iam_audit_config" "sts_audit" {
  project = var.project_id
  service = "sts.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

resource "google_project_iam_audit_config" "iam_audit" {
  project = var.project_id
  service = "iam.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}

resource "google_project_iam_audit_config" "storage_audit" {
  project = var.project_id
  service = "storage.googleapis.com"

  audit_log_config {
    log_type = "DATA_READ"
  }
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}
