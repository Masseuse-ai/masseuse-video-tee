# ---------------------------------------------------------------------------
# The sweeper: a backstop that stops a wedged slot
#
# A slot ends itself: the enclave exits about a minute after the session
# and the launcher shuts the VM down (tee_mode.IdleExit, vm.tf), and the
# trainer stops it on release too. The sweeper exists for the case where
# neither happens - the trainer died mid-session, a boot the watchdog never
# reached - so an a3-highgpu-1g does not bill indefinitely. Every so often
# it stops any slot that has been RUNNING far longer than a session could
# be (slot_max_uptime_hours; the producer caps a lease at six). It never
# starts anything: only a masseuse.ai visitor does that, through the
# trainer. Nothing here can read enclave state.
#
# On by default. The stop passes discardLocalSsd (the a3 has local NVMe and
# a plain stop is refused, hashicorp/terraform-provider-google#26173), and
# a stop/start is anyway the only correct recovery for the GPU attestation
# binding (README, "the stop/start rule").
# ---------------------------------------------------------------------------
resource "google_service_account" "sweeper" {
  count        = var.sweeper_enabled ? 1 : 0
  project      = var.project_id
  account_id   = "masseuse-video-tee-sweeper"
  display_name = "Stops wedged masseuse.ai video slots"
}

resource "google_project_iam_custom_role" "slot_sweeper" {
  count       = var.sweeper_enabled ? 1 : 0
  project     = var.project_id
  role_id     = "masseuseVideoTeeSlotSweeper"
  title       = "masseuse-video-tee slot sweeper"
  description = "Read instance status and stop a running instance; never start one."
  permissions = [
    "compute.instances.get",
    "compute.instances.stop",
    "compute.zoneOperations.get",
  ]
}

# Project-wide because the connector polls the zone operation, which is not
# an instance-level resource; the project holds nothing but the slots.
resource "google_project_iam_member" "sweeper_stops_slots" {
  count   = var.sweeper_enabled ? 1 : 0
  project = var.project_id
  role    = google_project_iam_custom_role.slot_sweeper[0].id
  member  = "serviceAccount:${google_service_account.sweeper[0].email}"
}

# The Workflows service agent is created lazily; creating a workflow before
# it exists fails ("Workflows service agent does not exist").
resource "google_project_service_identity" "workflows" {
  count    = var.sweeper_enabled ? 1 : 0
  provider = google-beta
  project  = var.project_id
  service  = "workflows.googleapis.com"

  depends_on = [google_project_service.apis]
}

resource "google_workflows_workflow" "sweeper" {
  count           = var.sweeper_enabled ? 1 : 0
  project         = var.project_id
  region          = var.region
  name            = "masseuse-video-tee-sweeper"
  description     = "Stop slots wedged RUNNING past slot_max_uptime_hours."
  service_account = google_service_account.sweeper[0].id
  # The sweeper holds no state: it is torn down and recreated with the
  # sweeper_enabled flag, so the provider's default deletion protection
  # would only block that.
  deletion_protection = false

  # For each slot (each in its own zone, slot_zones): get status; if
  # RUNNING and up longer than the cap, stop it (discarding the local
  # SSDs). A stop that fails is logged and tried again on the next run.
  # Anything not RUNNING is left alone.
  source_contents = <<-EOT
    main:
      params: [args]
      steps:
        - init:
            assign:
              - project: "${var.project_id}"
              - slots: ${jsonencode([for name in local.slot_names : { zone = local.slot_zone[name], instance = "masseuse-video-tee-${name}" }])}
              - max_uptime_s: ${var.slot_max_uptime_hours * 3600}
              - stopped: []
        - each:
            for:
              value: slot
              in: $${slots}
              steps:
                - status:
                    call: googleapis.compute.v1.instances.get
                    args:
                      project: $${project}
                      zone: $${slot.zone}
                      instance: $${slot.instance}
                    result: vm
                - decide:
                    switch:
                      - condition: $${vm.status == "RUNNING"}
                        steps:
                          - age:
                              assign:
                                - started_at: $${default(map.get(vm, "lastStartTimestamp"), vm.creationTimestamp)}
                                - uptime_s: $${sys.now() - time.parse(started_at)}
                          - maybe_stop:
                              switch:
                                - condition: $${uptime_s > max_uptime_s}
                                  steps:
                                    - stop:
                                        try:
                                          call: googleapis.compute.v1.instances.stop
                                          args:
                                            project: $${project}
                                            zone: $${slot.zone}
                                            instance: $${slot.instance}
                                            discardLocalSsd: true
                                        except:
                                          as: e
                                          steps:
                                            - log_failure:
                                                call: sys.log
                                                args:
                                                  severity: WARNING
                                                  text: '$${"stop failed for " + slot.instance + " in " + slot.zone + ", " + json.encode_to_string(e)}'
                                            - next_instance:
                                                next: continue
                                    - note:
                                        assign:
                                          - stopped: $${list.concat(stopped, slot.zone + "/" + slot.instance)}
        - done:
            return: $${stopped}
  EOT

  depends_on = [google_project_service.apis, google_project_service_identity.workflows[0]]
}

resource "google_project_iam_member" "sweeper_invokes_workflows" {
  count   = var.sweeper_enabled ? 1 : 0
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.sweeper[0].email}"
}

resource "google_cloud_scheduler_job" "sweeper" {
  count       = var.sweeper_enabled ? 1 : 0
  project     = var.project_id
  region      = var.region
  name        = "masseuse-video-tee-sweeper"
  description = "Stop wedged video slots."
  schedule    = var.sweeper_schedule
  time_zone   = var.sweeper_time_zone

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/${google_workflows_workflow.sweeper[0].id}/executions"
    body        = base64encode(jsonencode({ argument = "{}" }))
    headers     = { "Content-Type" = "application/json" }

    oauth_token {
      service_account_email = google_service_account.sweeper[0].email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [google_project_iam_member.sweeper_invokes_workflows]
}
