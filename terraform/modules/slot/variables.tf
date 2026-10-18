variable "project_id" {
  description = "The project the slots run in: yours."
  type        = string
}

variable "slots" {
  description = "Each slot: the zone it runs in (one that offers a3-highgpu-1g with Confidential Computing) and the DNS name devices reach it at, which must resolve to its address (output slots)."
  type = map(object({
    zone = string
    host = string
  }))

  validation {
    condition     = length(var.slots) > 0 && alltrue([for name, _ in var.slots : can(regex("^[a-z][a-z0-9-]{0,30}$", name))])
    error_message = "Name at least one slot, each lowercase letters, digits and hyphens."
  }
}

variable "network" {
  description = "The VPC network the slots join (its self link or id); the firewall rules below go on it."
  type        = string
}

variable "subnetworks" {
  description = "A subnetwork of that network for each region a slot's zone is in, by region."
  type        = map(string)
}

variable "container_image" {
  description = "The released image, by digest, in the registry the launcher verifies its signature in (the operator's hosted_slots output, container_image)."
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.container_image))
    error_message = "container_image is pinned by digest: <repository>@sha256:<64 hex>."
  }
}

variable "signature_repo" {
  description = "Where the launcher reads the release signature (hosted_slots.signature_repo); the image's repository when empty."
  type        = string
  default     = ""
}

variable "controller_url" {
  description = "The controller the slots post their readings to, attested as TRAINER_URL."
  type        = string
}

variable "controller_service_account" {
  description = "The one identity the slots' control routes take a token from, attested as TRAINER_INVOKER_SERVICE_ACCOUNT."
  type        = string
}

variable "allowed_origins" {
  description = "Page origins the slots answer (TEE_ALLOWED_ORIGINS); \"*\" for any page, every request under the device's capability being sealed regardless."
  type        = list(string)
  default     = ["*"]
}

variable "models_bucket" {
  description = "The bucket the weights and the analysis bundle are read from (hosted_slots.models_bucket)."
  type        = string
}

variable "wif_audience" {
  description = "The identity pool provider that exchanges the slot's attestation for a credential (hosted_slots.wif_audience)."
  type        = string
}

variable "hosted_principals" {
  description = "The pool's principals for the approved digests (hosted_slots.hosted_principals): granted, in this project, the ACME account binding the slot mints for its certificate at boot."
  type        = list(string)
}

variable "acme_directory_url" {
  description = "The ACME directory the slot's certificate comes from: Google Trust Services, whose account binding the slot mints in this project."
  type        = string
  default     = "https://dv.acme-v02.api.pki.goog/directory"
}

variable "acme_contact_email" {
  description = "The ACME account's contact."
  type        = string
}

variable "idle_exit_s" {
  description = "How long a slot stays up with no lease before the enclave exits and the VM powers off (TEE_IDLE_EXIT_S); null keeps the image's 60 s."
  type        = number
  default     = null
}

variable "boot_idle_s" {
  description = "The same for a boot that never gets a lease (TEE_BOOT_IDLE_S); null keeps the image's 300 s."
  type        = number
  default     = null
}

variable "models_tmpfs_gib" {
  description = "The /models tmpfs, in GiB: the weights and the analysis bundle, in memory only."
  type        = number
  default     = 8
}

variable "run_tmpfs_gib" {
  description = "The /run/tee tmpfs, in GiB: the certificate, the rendered credential."
  type        = number
  default     = 2
}

variable "name_prefix" {
  description = "Prefix of every resource name this module creates."
  type        = string
  default     = "enclave"
}

variable "media_source_ranges" {
  description = "Who may reach the slots' signalling (443) and media (8189) ports: the devices, so the internet by default."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
