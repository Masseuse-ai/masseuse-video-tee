# ---------------------------------------------------------------------------
# Image signing (Phase 5): a Cloud KMS signing key that Cloud Build uses
# with cosign after every push (cloudbuild.tee*.yaml, step "sign"). The
# launcher verifies the signature at boot (tee-signed-image-repos) and
# lists it in the attestation token as
# submods.container.image_signatures[] = {key_id, signature_algorithm}
# where key_id is the hex SHA-256 of the DER-encoded public key. The WIF
# production provider and the trainer/phone policy pin that key_id, so a
# rebuilt image nobody signed cannot read the weights or take a session,
# whatever its digest. Same shape as auth-broker-tee/terraform/kms.tf.
#
# Signing runs on every build; *requiring* the signature is the
# require_signed_image switch (production flip), because the debug images
# built before the key existed carry none.
# ---------------------------------------------------------------------------
resource "google_kms_key_ring" "image_signing" {
  project  = var.project_id
  name     = "masseuse-video-tee-cosign"
  location = var.region

  depends_on = [google_project_service.apis]
}

resource "google_kms_crypto_key" "image_signer" {
  name     = "image-signer"
  key_ring = google_kms_key_ring.image_signing.id
  purpose  = "ASYMMETRIC_SIGN"

  # Manual rotation: promote a new version to primary, sign the next
  # build, list both fingerprints (image_signer_fingerprints_extra) until
  # the last image signed by the old one is gone.
  version_template {
    algorithm        = "EC_SIGN_P256_SHA256"
    protection_level = "SOFTWARE"
  }

  # Losing the key invalidates every signature in flight and the pinned
  # key_id everywhere; deleting it has to be a deliberate edit here.
  lifecycle {
    prevent_destroy = true
  }
}

# Cloud Build signs (asymmetricSign) and reads the public key for the
# dev.cosignproject.cosign/pub annotation the launcher insists on.
resource "google_kms_crypto_key_iam_member" "cloud_build_signs" {
  crypto_key_id = google_kms_crypto_key.image_signer.id
  role          = "roles/cloudkms.signerVerifier"
  member        = local.cloud_build_sa
}

resource "google_kms_crypto_key_iam_member" "cloud_build_reads_key" {
  crypto_key_id = google_kms_crypto_key.image_signer.id
  role          = "roles/cloudkms.viewer"
  member        = local.cloud_build_sa
}

# The primary version's public key: PEM for VERIFICATION.md readers, and
# its fingerprint (what the token reports as key_id) computed the way the
# launcher does, sha256 over the DER SubjectPublicKeyInfo.
data "google_kms_crypto_key_version" "image_signer" {
  crypto_key = google_kms_crypto_key.image_signer.id
}

data "external" "image_signer_fingerprint" {
  program = ["sh", "-c", <<-EOT
    set -eu
    pem=$(cat | python3 -c 'import json,sys; print(json.load(sys.stdin)["pem"])')
    fp=$(printf '%s\n' "$pem" | openssl pkey -pubin -outform DER | openssl dgst -sha256 | awk '{print $NF}')
    printf '{"fingerprint":"%s"}' "$fp"
  EOT
  ]
  query = {
    pem = data.google_kms_crypto_key_version.image_signer.public_key[0].pem
  }
}

locals {
  image_signer_kms_key_uri = "gcpkms://${google_kms_crypto_key.image_signer.id}"
  image_signer_fingerprint = data.external.image_signer_fingerprint.result.fingerprint
  image_signer_algorithm   = "ECDSA_P256_SHA256"

  # Accepted "<algorithm>:<key_id>" pairs, one per live key version.
  accepted_image_signers = [
    for fp in distinct(concat([local.image_signer_fingerprint], var.image_signer_fingerprints_extra)) :
    "${local.image_signer_algorithm}:${fp}"
  ]
  accepted_image_signers_cel = "[${join(", ", [for s in local.accepted_image_signers : "'${s}'"])}]"

  # The CEL clause the production WIF provider adds once signatures are
  # required: some listed signer signed the running image.
  wif_signature_condition = "${local.accepted_image_signers_cel}.exists(s, s in assertion.submods.container.image_signatures.map(sig, sig.signature_algorithm + ':' + sig.key_id))"

  # Where the launcher looks for the signature: cosign stores it as a tag
  # on the image's own repository (<registry>/<project>/<ar repo>/<image>
  # :sha256-<digest>.sig), so tee-signed-image-repos names the image
  # repository, not the Artifact Registry repository above it (the codelab's
  # value ends in the image name for the same reason). Derived from
  # container_image so the two cannot drift.
  image_signature_repo = split("@", var.container_image)[0]
}

output "image_signer_kms_key_uri" {
  description = "cosign sign --key <this> (the build passes it as _COSIGN_KEY)."
  value       = local.image_signer_kms_key_uri
}

output "image_signer_fingerprint" {
  description = "Hex SHA-256 of the signing key's DER public key: image_signatures[].key_id in the attestation, and the value for the trainer's tee_policy.image_signatures."
  value       = local.image_signer_fingerprint
}

output "image_signer_public_key_pem" {
  description = "The signing public key, for VERIFICATION.md and external verifiers."
  value       = data.google_kms_crypto_key_version.image_signer.public_key[0].pem
}
