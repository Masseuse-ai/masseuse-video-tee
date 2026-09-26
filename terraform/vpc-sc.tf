# ---------------------------------------------------------------------------
# Phase 5: a VPC Service Controls perimeter around the project
#
# Restricts storage, compute, confidentialcomputing, sts and
# artifactregistry to callers inside the perimeter or on the operator
# access level, with the egress rules Confidential Space needs
# (https://docs.cloud.google.com/confidential-computing/confidential-space/docs/vpc-service-controls):
#   - google.storage.objects.get to projects 870449385679 and 180376494128
#     (the Confidential Space images and the GPU driver bucket),
#   - InstancesService.Insert to project 30229352718 (the launcher's
#     attestation flow),
#   - google.storage.objects.create to the project holding the session
#     records bucket (capture_bucket_project_number; the enclave writes a
#     leased session's record there and nothing else, README "Session
#     records").
#
# Access Context Manager policies are organization resources. The project
# currently has no organization parent (Phase 0: `gcloud organizations
# list` is empty), so this file is inert until vpc_sc_enabled is true with
# an access_policy_id from an organization the project has been migrated
# into. Apply in dry-run first (vpc_sc_dry_run = true), read the
# `dry_run` audit logs for a week, then enforce.
# ---------------------------------------------------------------------------
locals {
  vpc_sc_capture_egress = var.capture_bucket != "" && var.capture_bucket_project_number != ""
  vpc_sc_on             = var.vpc_sc_enabled && var.access_policy_id != ""

  vpc_sc_services = [
    "storage.googleapis.com",
    "compute.googleapis.com",
    "confidentialcomputing.googleapis.com",
    "sts.googleapis.com",
    "artifactregistry.googleapis.com",
  ]
}

resource "google_access_context_manager_access_level" "operators" {
  count  = local.vpc_sc_on ? 1 : 0
  parent = "accessPolicies/${var.access_policy_id}"
  name   = "accessPolicies/${var.access_policy_id}/accessLevels/masseuse_video_tee_operators"
  title  = "masseuse video TEE operators"

  basic {
    conditions {
      members = var.operator_members
    }
  }
}

resource "google_access_context_manager_service_perimeter" "tee" {
  count                     = local.vpc_sc_on ? 1 : 0
  parent                    = "accessPolicies/${var.access_policy_id}"
  name                      = "accessPolicies/${var.access_policy_id}/servicePerimeters/masseuse_video_tee"
  title                     = "masseuse video TEE"
  perimeter_type            = "PERIMETER_TYPE_REGULAR"
  use_explicit_dry_run_spec = var.vpc_sc_dry_run

  dynamic "spec" {
    for_each = var.vpc_sc_dry_run ? [1] : []
    content {
      resources           = ["projects/${var.project_number}"]
      restricted_services = local.vpc_sc_services
      access_levels       = [google_access_context_manager_access_level.operators[0].name]

      egress_policies {
        egress_from {
          identity_type = "ANY_IDENTITY"
        }
        egress_to {
          resources = ["projects/870449385679", "projects/180376494128"]
          operations {
            service_name = "storage.googleapis.com"
            method_selectors {
              method = "google.storage.objects.get"
            }
          }
        }
      }

      egress_policies {
        egress_from {
          identity_type = "ANY_IDENTITY"
        }
        egress_to {
          resources = ["projects/30229352718"]
          operations {
            service_name = "compute.googleapis.com"
            method_selectors {
              method = "InstancesService.Insert"
            }
          }
        }
      }

      # The session record (workload/producer/record.py): the enclave
      # creates objects in the capture bucket, which lives in the trainer's
      # project. Create only; nothing in the perimeter reads or deletes
      # there.
      dynamic "egress_policies" {
        for_each = local.vpc_sc_capture_egress ? [1] : []
        content {
          egress_from {
            identity_type = "ANY_IDENTITY"
          }
          egress_to {
            resources = ["projects/${var.capture_bucket_project_number}"]
            operations {
              service_name = "storage.googleapis.com"
              method_selectors {
                method = "google.storage.objects.create"
              }
            }
          }
        }
      }
    }
  }

  dynamic "status" {
    for_each = var.vpc_sc_dry_run ? [] : [1]
    content {
      resources           = ["projects/${var.project_number}"]
      restricted_services = local.vpc_sc_services
      access_levels       = [google_access_context_manager_access_level.operators[0].name]

      egress_policies {
        egress_from {
          identity_type = "ANY_IDENTITY"
        }
        egress_to {
          resources = ["projects/870449385679", "projects/180376494128"]
          operations {
            service_name = "storage.googleapis.com"
            method_selectors {
              method = "google.storage.objects.get"
            }
          }
        }
      }

      egress_policies {
        egress_from {
          identity_type = "ANY_IDENTITY"
        }
        egress_to {
          resources = ["projects/30229352718"]
          operations {
            service_name = "compute.googleapis.com"
            method_selectors {
              method = "InstancesService.Insert"
            }
          }
        }
      }

      # The session record (workload/producer/record.py): the enclave
      # creates objects in the capture bucket, which lives in the trainer's
      # project. Create only; nothing in the perimeter reads or deletes
      # there.
      dynamic "egress_policies" {
        for_each = local.vpc_sc_capture_egress ? [1] : []
        content {
          egress_from {
            identity_type = "ANY_IDENTITY"
          }
          egress_to {
            resources = ["projects/${var.capture_bucket_project_number}"]
            operations {
              service_name = "storage.googleapis.com"
              method_selectors {
                method = "google.storage.objects.create"
              }
            }
          }
        }
      }
    }
  }
}
