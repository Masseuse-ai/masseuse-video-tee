# ---------------------------------------------------------------------------
# Network: one VPC, a subnet in every region the slots span (slot_zones),
# a static IP per slot in its own region, attached directly to the VM (no
# load balancer between the phone and the enclave's TLS), and a firewall
# that admits only what the workload serves. The firewall rules are
# network-wide, so a slot in a new region needs only its subnet and IP.
#
#   tcp 443        Caddy: WHIP/WHEP signalling, /attestation, the trainer's
#                  control routes, and the TLS-ALPN-01 challenge itself.
#   udp/tcp 8189   MediaMTX: ICE/DTLS/SRTP - the camera and the overlay.
#   tcp 22         debug phase only, from IAP's TCP-forwarding range.
#
# No port 80: Caddy's HTTP challenge is disabled and nothing redirects.
# ---------------------------------------------------------------------------
resource "google_compute_network" "tee" {
  project                 = var.project_id
  name                    = "masseuse-video-tee"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "tee" {
  for_each                 = local.slot_regions
  project                  = var.project_id
  name                     = "masseuse-video-tee-${each.key}"
  region                   = each.key
  network                  = google_compute_network.tee.id
  ip_cidr_range            = var.subnet_cidrs[each.key]
  private_ip_google_access = true

  lifecycle {
    precondition {
      condition     = contains(keys(var.subnet_cidrs), each.key)
      error_message = "slot_zones puts a slot in ${each.key}, which has no subnet_cidrs entry."
    }
  }
}

# The subnet predates slot_zones and was a single resource; it keeps its
# name, range and state under the new key.
moved {
  from = google_compute_subnetwork.tee
  to   = google_compute_subnetwork.tee["us-central1"]
}

resource "google_compute_address" "slot" {
  for_each     = toset(local.slot_names)
  project      = var.project_id
  name         = "masseuse-video-tee-${each.value}"
  region       = local.slot_region[each.value]
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
  # description is immutable on an address (a change would replace it, which
  # prevent_destroy refuses), so it does not name the zone.
  description = "Direct static IP for ${local.slot_hosts[each.value]}"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_compute_firewall" "allow_signalling" {
  project = var.project_id
  name    = "masseuse-video-tee-allow-https"
  network = google_compute_network.tee.name

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["masseuse-video-tee"]
}

resource "google_compute_firewall" "allow_media" {
  project = var.project_id
  name    = "masseuse-video-tee-allow-media"
  network = google_compute_network.tee.name

  allow {
    protocol = "udp"
    ports    = ["8189"]
  }
  allow {
    protocol = "tcp"
    ports    = ["8189"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["masseuse-video-tee"]
}

resource "google_compute_firewall" "allow_iap_ssh" {
  count   = var.debug_mode ? 1 : 0
  project = var.project_id
  name    = "masseuse-video-tee-allow-iap-ssh"
  network = google_compute_network.tee.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["masseuse-video-tee"]
}
