variable "project_id" {
  description = "The dedicated project for the video TEE (created in Phase 0)."
  type        = string
  default     = "prod-masseuse-video-tee"
}

variable "project_number" {
  description = "Its project number (WIF principal strings need it)."
  type        = string
  default     = "181371164640"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  description = "a3-highgpu-1g with TDX is offered in us-central1-a/b/c; the static IP is regional so any of them works."
  type        = string
  default     = "us-central1-a"
}

# ---------------------------------------------------------------------------
# Debug vs production
# ---------------------------------------------------------------------------

variable "debug_mode" {
  description = <<-EOT
    true (Phase 4 soak): the confidential-space-debug image, container stdout
    to Cloud Logging and the serial console, memory metrics, SSH via IAP,
    the attestation-debug WIF provider (dbgstat == 'enabled'), and the
    launcher holding the VM after any exit for the trainer's reaper to stop.
    false (Phase 5): the production image family, no log redirect, no
    monitoring, no SSH rule, no debug provider, tee-restart-policy=Never
    (the launcher powers the VM off two minutes after any exit). Both modes
    issue from Google Trust Services. The trainer's TEE_POLICY_JSON must
    agree (allowDebug / requireStable).
  EOT
  type        = bool
  default     = true
}

variable "slot_count" {
  description = "How many slot VMs (slot-0..N-1). Each is one a3-highgpu-1g Spot VM with its own static IP and hostname."
  type        = number
  default     = 1
}

variable "vm_running" {
  description = "true asks Terraform to start every slot VM; false leaves the running state alone (the trainer and the enclave's idle exit stop a slot; a stop needs the discard-local-ssd flag the provider cannot send, see vm.tf). The static IP and disk stay either way."
  type        = bool
  default     = true
}

# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------

variable "container_image" {
  description = "The TEE workload image by digest: <region>-docker.pkg.dev/<project>/masseuse-video-tee/masseuse-video-tee@sha256:... (cloudbuild.tee.yaml)."
  type        = string
}

variable "container_image_digest" {
  description = "The sha256:... of container_image; the WIF principal that may read the models bucket."
  type        = string

  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.container_image_digest))
    error_message = "container_image_digest must be sha256:<64 hex>."
  }
}

variable "candidate_image_digests" {
  description = "Digests being rolled in (or out): they may read the models bucket too, so a new image boots before the VM metadata flips to it."
  type        = list(string)
  default     = []
}

variable "require_signed_image" {
  description = <<-EOT
    Phase 5: the launcher verifies the image's cosign signature at boot
    (tee-signed-image-repos), the production WIF provider demands a
    signature by the KMS key in signing.tf, and the trainer's tee_policy
    should list image_signer_fingerprint. Off while the debug images
    (built before the key) are in use.
  EOT
  type        = bool
  default     = false
}

variable "sweeper_enabled" {
  description = "Cloud Scheduler + Workflows stop a slot wedged RUNNING past slot_max_uptime_hours (sweeper.tf): the backstop for a trainer that died mid-session or a boot the enclave watchdog never reached. On by default; it never starts a VM."
  type        = bool
  default     = true
}

variable "sweeper_schedule" {
  description = "When the sweeper looks (cron, sweeper_time_zone). Every fifteen minutes, all day: a wedged a3-highgpu-1g bills around the clock, so there are no off hours to skip."
  type        = string
  default     = "*/15 * * * *"
}

variable "sweeper_time_zone" {
  description = "Time zone for sweeper_schedule (the schedule is uniform, so it barely matters)."
  type        = string
  default     = "America/Los_Angeles"
}

variable "slot_max_uptime_hours" {
  description = "The sweeper stops a slot that has been RUNNING longer than this. Above the producer's six-hour lease cap so it only ever catches a genuinely wedged VM, not a long legitimate session."
  type        = number
  default     = 7
}

variable "image_signer_fingerprints_extra" {
  description = "Additional accepted signing-key fingerprints (hex sha256 of DER public key) during a key rotation; the current primary version's is always accepted."
  type        = list(string)
  default     = []
}

# ---------------------------------------------------------------------------
# What the entrypoint reads (tee-env-*)
# ---------------------------------------------------------------------------

variable "slot_domain" {
  description = "Slot hostnames are slot-<n>.<slot_domain>; the A records live in link-router/terraform/tee-dns.tf."
  type        = string
  default     = "tee.masseuse.ai"
}

variable "trainer_url" {
  description = "Where the slot posts readings: the trainer's Cloud Run origin (its ID-token audience), not the Cloudflare front door."
  type        = string
  default     = "https://masseuse-trainer-125139120897.us-central1.run.app"
}

variable "phone_origins" {
  description = "Page origins the phone signals from (CORS allow-list for the slot's WHIP/WHEP and /attestation, TEE_ALLOWED_ORIGINS). The Cloudflare front door(s); trainer_url is added for the direct-to-Cloud-Run path used during the soak."
  type        = list(string)
  default     = ["https://masseuse.ai", "https://www.masseuse.ai"]
}

variable "trainer_invoker_service_accounts" {
  description = "The trainer's runtime service account(s): the only callers of the slot's control routes."
  type        = list(string)
  default     = ["masseuse-trainer@prod-femled-couple-router.iam.gserviceaccount.com"]
}

variable "trainer_service_account" {
  description = "The trainer's runtime service account, the one identity allowed to start and stop a slot VM on demand (trainer-iam.tf). Normally the same account that appears in trainer_invoker_service_accounts."
  type        = string
  default     = "masseuse-trainer@prod-femled-couple-router.iam.gserviceaccount.com"
}

variable "acme_directory_url" {
  description = "Set explicitly to override the default, Google Trust Services production (https://dv.acme-v02.api.pki.goog/directory). The GTS staging directory is https://dv.acme-v02.test-api.pki.goog/directory; the entrypoint mints the EAB from the matching Public CA endpoint (preprod- for staging)."
  type        = string
  default     = ""
}

variable "acme_contact_email" {
  type    = string
  default = "ops@femled.ai"
}

variable "models_tmpfs_gib" {
  description = "The /models tmpfs, in GiB: hf-bf16 (3.0 GB) + detectors (0.25 GB) + artifacts, with headroom. (tee-mount's size= is plain bytes; the launcher refuses a unit suffix.)"
  type        = number
  default     = 8
}

variable "run_tmpfs_gib" {
  description = "The /run/tee tmpfs, in GiB: Caddy's storage, the rendered credential, a session's capture rows."
  type        = number
  default     = 2
}

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

variable "operator_members" {
  description = "Who may run Terraform, Cloud Build and (debug only) SSH: the VPC-SC ingress access level is built from these. Set in terraform.tfvars (IAM members, e.g. user:... or group:...); never committed."
  type        = list(string)
  default     = []
}

variable "source_models_bucket" {
  description = "The current producer bucket the weights are copied from (README: copy step); informational."
  type        = string
  default     = "prod-femled-couple-router-pose-producer"
}

# ---------------------------------------------------------------------------
# Phase 5: VPC Service Controls (needs an organization; see vpc-sc.tf)
# ---------------------------------------------------------------------------

variable "vpc_sc_enabled" {
  type    = bool
  default = false
}

variable "vpc_sc_dry_run" {
  description = "Apply the perimeter in dry-run mode first and read the audit logs before enforcing."
  type        = bool
  default     = true
}

variable "access_policy_id" {
  description = "The organization's Access Context Manager policy id (accessPolicies/<id>); empty until the project has an organization parent."
  type        = string
  default     = ""
}
