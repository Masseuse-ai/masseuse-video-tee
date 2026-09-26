# ---------------------------------------------------------------------------
# The slot VMs
#
# One a3-highgpu-1g (Intel TDX + one H100 in confidential-computing mode)
# per slot, Spot - Confidential Space GPU VMs must be Spot or Flex-start -
# booting the Confidential Space image family debug_mode selects, running
# the TEE workload image by digest with the tee-env-* the entrypoint reads.
# The slots are spread over slot_zones by index (evens in us-central1-a,
# odds in us-east5-a with the defaults), each in its region's subnet with
# its region's static IP, so that a Spot stockout in one zone leaves the
# trainer the other zone's slots to start (masseuse-trainer slot-pool.js).
#
# The VM's running state is not Terraform's: the trainer starts a slot when
# a masseuse.ai visitor taps Enable camera (the trainer calls the Compute
# API) and the slot ends about a minute after the session: the
# enclave exits 0 (tee_mode.IdleExit), the trainer's reaper stops the VM
# as soon as the enclave is gone, and on the production image the
# launcher's own `shutdown --poweroff +2` follows anyway. So this resource
# ignores desired_status, and the sweeper (sweeper.tf) only stops a VM
# wedged RUNNING far too long. Preemption STOPs the VM rather than deleting
# it: the static IP, the disk and this resource survive, and the next tap
# starts it again.
#
# A guest *reboot* is the one thing to avoid: on H100 + TDX the GPU's
# attestation binding is lost until a full stop/start (README, "stop/start
# rule"). What the launcher does after the workload exits is decided by
# go-tpm-tools launcher/launcher/main.go getExitCode and the image's
# exit_script.sh: the debug image always *holds* (exit 4, "VM remains
# running": nothing stops it but the trainer or the sweeper); the
# production image powers off after a clean exit under Never or OnFailure
# (exit 0) and after a crash under Never (exit 1), but *reboots the guest*
# (exit 3) after a crash under OnFailure and after any exit under Always.
# Hence Never below.
# ---------------------------------------------------------------------------
locals {
  slot_metadata_common = merge(
    {
      "tee-image-reference"    = var.container_image
      "tee-install-gpu-driver" = "true"
      # Weights and the boot's ephemeral state live in memory only. size= is
      # bytes: the launcher parses it with ParseUint and fails the whole
      # launch spec on "8G" (the first boot, 2026-09-08).
      "tee-mount" = join(";", [
        "type=tmpfs,source=tmpfs,destination=/models,size=${var.models_tmpfs_gib * 1073741824}",
        "type=tmpfs,source=tmpfs,destination=/run/tee,size=${var.run_tmpfs_gib * 1073741824}",
      ])
      "tee-env-TRAINER_URL"                     = var.trainer_url
      "tee-env-TRAINER_INVOKER_SERVICE_ACCOUNT" = join(",", concat(var.trainer_invoker_service_accounts, var.debug_mode ? [google_service_account.soak[0].email] : []))
      "tee-env-TEE_ALLOWED_ORIGINS"             = join(",", distinct(concat(var.phone_origins, [var.trainer_url])))
      "tee-env-MODELS_BUCKET"                   = google_storage_bucket.models.name
      "tee-env-WIF_AUDIENCE"                    = local.wif_audience
      "tee-env-ACME_DIRECTORY_URL"              = local.acme_directory_url
      "tee-env-ACME_CONTACT_EMAIL"              = var.acme_contact_email
    },
    var.debug_mode ? {
      # Debug: stdout to Cloud Logging + serial and memory metrics. No
      # tee-restart-policy (production image only). On this image the
      # launcher holds the VM after the workload exits (exit 4, observed
      # 2026-09-08: "TEE container launcher exiting exit_code=4 exit_msg=VM
      # remains running"), so the idle exit alone stops nothing here; the
      # trainer's reaper does (it sees the enclave gone and calls
      # instances.stop), and the sweeper is the backstop.
      "tee-container-log-redirect"   = "true"
      "tee-monitoring-memory-enable" = "true"
      # SSH (IAP, debug only) through OS Login, so `gcloud compute ssh` does
      # not park a key in instance metadata for Terraform to fight over.
      "enable-oslogin" = "TRUE"
      } : {
      # Never (the default, written down because the choice matters): a
      # clean exit 0 powers the VM off (the idle exit), and so does a
      # crash, which the next tap answers with a fresh stop/start.
      # OnFailure would *reboot the guest* on a crash and Always on every
      # exit, and a guest reboot leaves the H100's attestation binding
      # broken until a full stop/start: a slot up but refused by every
      # verifier. Not the in-place container restart the name suggests.
      "tee-restart-policy" = "Never"
    },
    # The launcher fetches and verifies the cosign signature from here at
    # boot and reports it in the token (signing.tf).
    var.require_signed_image ? {
      "tee-signed-image-repos" = local.image_signature_repo
    } : {},
  )
}

resource "google_compute_instance" "slot" {
  for_each     = toset(local.slot_names)
  project      = var.project_id
  name         = "masseuse-video-tee-${each.value}"
  machine_type = "a3-highgpu-1g"
  zone         = local.slot_zone[each.value]

  # Never "TERMINATED": stopping a VM with local SSDs needs the API's
  # discard-local-ssd flag, which the provider does not send, so a create
  # (or replace) with vm_running=false fails after the VM is up and leaves
  # the resource tainted. A fresh VM therefore runs until the enclave's idle
  # exit and the trainer's reaper stop it, as after any session; to park by
  # hand: gcloud compute instances stop ... --discard-local-ssd=true.
  desired_status            = var.vm_running ? "RUNNING" : null
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

  # pd-ssd, not pd-balanced, for the image pull: the launcher fsyncs every
  # layer blob into the (dm-crypt, dm-integrity) stateful partition before
  # it unpacks it, and the harness on the debug slot (README "Boot time")
  # showed that commit pinned at pd-balanced's 100 GiB ceiling, 168 MiB/s,
  # for 17 of the pull's 29 s. pd-ssd's baseline is 240 MiB/s + 0.48/GiB,
  # 288 MiB/s here, for $7/month more. Changing the type replaces the VM.
  boot_disk {
    initialize_params {
      image = "projects/confidential-space-images/global/images/family/${local.cs_image_family}"
      size  = 100
      type  = "pd-ssd"
    }
  }

  # a3-highgpu-1g comes with two 375 GB local NVMe SSDs whether asked for or
  # not; declaring them keeps Terraform from planning their removal, which
  # would replace the VM. The workload never sees them: the launcher mounts
  # only the tee-mount tmpfs into the container, so nothing lands on them.
  scratch_disk {
    interface = "NVME"
  }
  scratch_disk {
    interface = "NVME"
  }

  network_interface {
    subnetwork = google_compute_subnetwork.tee[local.slot_region[each.value]].id

    access_config {
      nat_ip       = google_compute_address.slot[each.value].address
      network_tier = "PREMIUM"
    }
  }

  metadata = merge(local.slot_metadata_common, {
    "tee-env-TEE_PUBLIC_HOST" = local.slot_hosts[each.value]
    "tee-env-TEE_PUBLIC_IP"   = google_compute_address.slot[each.value].address
    "tee-env-SLOT_NAME"       = "tee-${each.value}"
  })

  service_account {
    email  = google_service_account.vm.email
    scopes = ["cloud-platform"]
  }

  tags = ["masseuse-video-tee"]

  # Changing debug_mode changes the boot image family, which the provider
  # does not see as a change (the debug image's name contains the
  # production family's): apply with
  #   -replace='google_compute_instance.slot["slot-0"]'
  # while the slot is TERMINATED and unleased. A replaced instance comes
  # back without its instance-level IAM (trainer-iam.tf), so plan and apply
  # once more right after; until then the trainer cannot start or stop it.
  lifecycle {
    # The trainer starts the VM and the enclave stops it (see the header):
    # its power state is theirs, not Terraform's, so a plan never fights a
    # slot that is up for a session or down between them. desired_status
    # still sets the state at create.
    ignore_changes = [desired_status]

    precondition {
      condition     = endswith(var.container_image, "@${var.container_image_digest}")
      error_message = "container_image must be pinned to container_image_digest (the WIF principal that may read the weights)."
    }
  }

  depends_on = [
    google_iam_workload_identity_pool_provider.attestation_prod,
    google_storage_bucket_iam_member.models_wif_reader,
    google_artifact_registry_repository_iam_member.vm_pulls,
    google_project_iam_member.vm_confidential_workload,
    google_compute_firewall.allow_signalling,
    google_compute_firewall.allow_media,
  ]
}
