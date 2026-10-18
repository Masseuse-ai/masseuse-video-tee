variable "project_id" {
  description = "Your project."
  type        = string
}

variable "slots" {
  description = "Each slot's zone and DNS name."
  type = map(object({
    zone = string
    host = string
  }))
}

variable "network" {
  description = "Your VPC network (self link or id)."
  type        = string
}

variable "subnetworks" {
  description = "A subnetwork of it per region the slots' zones are in."
  type        = map(string)
}

variable "container_image" {
  description = "The released image by digest (the operator's hosted_slots.container_image, or a release approved for hosted slots)."
  type        = string
}

variable "signature_repo" {
  description = "Where the launcher reads the release signature (hosted_slots.signature_repo)."
  type        = string
  default     = ""
}

variable "controller_url" {
  description = "The controller your project was approved for: where readings go."
  type        = string
}

variable "controller_service_account" {
  description = "The identity that controller presents to the slots."
  type        = string
}

variable "models_bucket" {
  description = "hosted_slots.models_bucket"
  type        = string
}

variable "wif_audience" {
  description = "hosted_slots.wif_audience"
  type        = string
}

variable "hosted_principals" {
  description = "hosted_slots.hosted_principals"
  type        = list(string)
}

variable "acme_contact_email" {
  description = "Your contact for the slots' certificates."
  type        = string
}

variable "idle_exit_s" {
  description = "How long an idle slot stays up; null keeps the image's 60 s."
  type        = number
  default     = null
}
