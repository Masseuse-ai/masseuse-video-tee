# ---------------------------------------------------------------------------
# Released slots in a project of your own
#
# The same machine the operator's fleet runs (../../vm.tf): an
# a3-highgpu-1g on Intel TDX with its H100 in confidential-computing mode,
# Spot (Confidential Space GPU VMs must be Spot or Flex-start), booting the
# production Confidential Space image (debugging disabled since boot) and
# the released image by digest, with the deployment variables the launch
# policy allows. Here they name your controller and no record bucket, and
# the weights and the analysis bundle come from the operator's bucket
# through the hosted-slots pool (../../hosted.tf), which admits your
# project once the operator has approved it.
#
# Owning this project lets you start, stop and name the slots. It does not
# give you a session: devices and controllers hold each slot to the
# release's public record, the TLS key and the seal key its attestation
# names, and every reading to its evidence key (VERIFY.md).
# ---------------------------------------------------------------------------
locals {
  slot_region      = { for name, slot in var.slots : name => regex("^(.+)-[a-z]$", slot.zone)[0] }
  signature_repo   = var.signature_repo != "" ? var.signature_repo : split("@", var.container_image)[0]
  image_ref_prefix = "${join("/", slice(split("/", local.signature_repo), 0, length(split("/", local.signature_repo)) - 1))}/"
  tag              = "${var.name_prefix}-slot"

  metadata_common = merge(
    {
      "tee-image-reference"    = var.container_image
      "tee-install-gpu-driver" = "true"
      "tee-mount" = join(";", [
        "type=tmpfs,source=tmpfs,destination=/models,size=${var.models_tmpfs_gib * 1073741824}",
        "type=tmpfs,source=tmpfs,destination=/run/tee,size=${var.run_tmpfs_gib * 1073741824}",
      ])
      "tee-env-TRAINER_URL"                     = var.controller_url
      "tee-env-TRAINER_INVOKER_SERVICE_ACCOUNT" = var.controller_service_account
      "tee-env-TEE_ALLOWED_ORIGINS"             = join(",", var.allowed_origins)
      "tee-env-MODELS_BUCKET"                   = var.models_bucket
      "tee-env-WIF_AUDIENCE"                    = var.wif_audience
      "tee-env-GOOGLE_CLOUD_PROJECT"            = var.project_id
      "tee-env-ACME_DIRECTORY_URL"              = var.acme_directory_url
      "tee-env-ACME_CONTACT_EMAIL"              = var.acme_contact_email
      # A clean exit or a crash powers the VM off; a guest reboot would
      # leave the H100's attestation binding broken until a stop and start.
      "tee-restart-policy"     = "Never"
      "tee-signed-image-repos" = local.signature_repo
    },
    var.idle_exit_s != null ? { "tee-env-TEE_IDLE_EXIT_S" = tostring(var.idle_exit_s) } : {},
    var.boot_idle_s != null ? { "tee-env-TEE_BOOT_IDLE_S" = tostring(var.boot_idle_s) } : {},
  )
}

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "confidentialcomputing.googleapis.com",
    "publicca.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_service_account" "vm" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-slot-vm"
  display_name = "Released enclave slot VM (Confidential Space)"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "vm_confidential_workload" {
  project = var.project_id
  role    = "roles/confidentialcomputing.workloadUser"
  member  = "serviceAccount:${google_service_account.vm.email}"
}

# The slot's certificate: each boot registers a fresh ACME account with
# Google Trust Services, whose binding the enclave mints in this project
# with its attested identity, the operator's pool principal for its digest.
resource "google_project_iam_member" "slot_mints_eab" {
  for_each = toset(var.hosted_principals)
  project  = var.project_id
  role     = "roles/publicca.externalAccountKeyCreator"
  member   = each.value

  depends_on = [google_project_service.apis]
}

resource "google_compute_address" "slot" {
  for_each     = var.slots
  project      = var.project_id
  name         = "${var.name_prefix}-${each.key}"
  region       = local.slot_region[each.key]
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
  description  = "Static IP for ${each.value.host}"

  depends_on = [google_project_service.apis]
}

resource "google_compute_firewall" "signalling" {
  project = var.project_id
  name    = "${var.name_prefix}-slot-signalling"
  network = var.network

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = var.media_source_ranges
  target_tags   = [local.tag]
}

resource "google_compute_firewall" "media" {
  project = var.project_id
  name    = "${var.name_prefix}-slot-media"
  network = var.network

  allow {
    protocol = "udp"
    ports    = ["8189"]
  }
  allow {
    protocol = "tcp"
    ports    = ["8189"]
  }

  source_ranges = var.media_source_ranges
  target_tags   = [local.tag]
}

resource "google_compute_instance" "slot" {
  for_each     = var.slots
  project      = var.project_id
  name         = "${var.name_prefix}-${each.key}"
  machine_type = "a3-highgpu-1g"
  zone         = each.value.zone

  allow_stopping_for_update = true

  confidential_instance_config {
    confidential_instance_type = "TDX"
  }

  scheduling {
    provisioning_model          = "SPOT"
    preemptible                 = true
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    instance_termination_action = "STOP"
  }

  guest_accelerator {
    type  = "nvidia-h100-80gb"
    count = 1
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  boot_disk {
    initialize_params {
      image = "projects/confidential-space-images/global/images/family/confidential-space"
      size  = 100
      type  = "pd-ssd"
    }
  }

  # The machine type's two local NVMe SSDs, declared so a plan does not
  # replace the VM; the workload sees only its tmpfs mounts.
  scratch_disk {
    interface = "NVME"
  }
  scratch_disk {
    interface = "NVME"
  }

  network_interface {
    subnetwork = var.subnetworks[local.slot_region[each.key]]

    access_config {
      nat_ip       = google_compute_address.slot[each.key].address
      network_tier = "PREMIUM"
    }
  }

  metadata = merge(local.metadata_common, {
    "tee-env-TEE_PUBLIC_HOST" = each.value.host
    "tee-env-TEE_PUBLIC_IP"   = google_compute_address.slot[each.key].address
    "tee-env-SLOT_NAME"       = "tee-${each.key}"
  })

  service_account {
    email  = google_service_account.vm.email
    scopes = ["cloud-platform"]
  }

  tags = [local.tag]

  lifecycle {
    # Whether a slot is up is its controller's and its own idle exit's
    # business, not Terraform's.
    ignore_changes = [desired_status]

    precondition {
      condition     = contains(keys(var.subnetworks), local.slot_region[each.key])
      error_message = "${each.key} is in ${local.slot_region[each.key]}, which has no entry in subnetworks."
    }
  }

  depends_on = [
    google_project_iam_member.vm_confidential_workload,
    google_project_iam_member.slot_mints_eab,
    google_compute_firewall.signalling,
    google_compute_firewall.media,
  ]
}
