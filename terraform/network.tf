# ---------------------------------------------------------------------------
# Network: a static IP per slot, attached directly to the VM (no load
# balancer between the phone and the enclave's TLS), and a firewall that
# admits only what the workload serves.
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
  project                  = var.project_id
  name                     = "masseuse-video-tee-${var.region}"
  region                   = var.region
  network                  = google_compute_network.tee.id
  ip_cidr_range            = "10.90.0.0/24"
  private_ip_google_access = true
}

resource "google_compute_address" "slot" {
  for_each     = toset(local.slot_names)
  project      = var.project_id
  name         = "masseuse-video-tee-${each.value}"
  region       = var.region
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
  description  = "Direct static IP for ${local.slot_hosts[each.value]}"

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
