# ---------------------------------------------------------------------------
# The release workflow's identity: GitHub Actions OIDC, no long-lived key
#
# .github/workflows/release.yml builds the enclave image on GitHub-hosted
# runners, pushes it to ghcr.io with SLSA provenance and a keyless cosign
# signature, and then *promotes* it: `crane copy` by digest into the
# Artifact Registry repository the WIF condition requires
# (attribute.image_reference must be under it), and `cosign sign` with the
# KMS key in signing.tf so the launcher's signature check
# (tee-signed-image-repos) and the production provider's key_id pin are
# unchanged. The promote job needs write on the repository and sign on the
# key, nothing else; it gets them through this pool, and only from runs of
# release.yml in the named repository on a release tag. Nothing else can
# sign with the key (signing.tf grants it to no other principal), so a
# signature by it, which the launcher verifies at boot and the attestation
# token reports as image_signatures[].key_id, means: built by release.yml
# from a tag of this repository. That is what the trainer, the phone and
# the camera connector pin, instead of a list of digests that would always
# be one release behind the image (VERIFY.md, "What identifies the image").
#
# A second pool rather than a second provider in `tee`: the enclave pool's
# principals are image digests and its bucket grant is per digest; mixing
# a CI identity into it would widen what "a principal of that pool" means.
# ---------------------------------------------------------------------------
variable "github_repository" {
  description = "The GitHub repository whose release workflow may promote images into Artifact Registry and sign them (owner/name)."
  type        = string
  default     = "FemLed/masseuse-video-tee"
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-release"
  display_name              = "GitHub release workflow"
  description               = "OIDC from GitHub Actions for the enclave image release"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  attribute_mapping = {
    "google.subject"         = "assertion.sub"
    "attribute.repository"   = "assertion.repository"
    "attribute.ref"          = "assertion.ref"
    "attribute.ref_type"     = "assertion.ref_type"
    "attribute.workflow"     = "assertion.workflow_ref"
    "attribute.job_workflow" = "assertion.job_workflow_ref"
  }

  # Only this repository, only a release tag, only release.yml: a pull
  # request, a branch build, or another workflow file added to the
  # repository (even on a tag) gets no token exchange. job_workflow_ref is
  # the workflow file the running job belongs to, at the ref that ran it,
  # so the signing job is the release.yml of the tag's own commit. The tag
  # itself cannot be moved or deleted afterwards: the repository's ruleset
  # "Tags are immutable" (all tags; no update, deletion or force push; no
  # bypass actor) keeps the commit a provenance names the commit the tag
  # named when it was released.
  attribute_condition = join(" && ", [
    "assertion.repository == '${var.github_repository}'",
    "assertion.ref_type == 'tag'",
    "assertion.ref.startsWith('refs/tags/v')",
    "assertion.job_workflow_ref.startsWith('${var.github_repository}/.github/workflows/release.yml@refs/tags/v')",
  ])
}

resource "google_service_account" "github_release" {
  project      = var.project_id
  account_id   = "github-release"
  display_name = "GitHub release workflow (image promote + sign)"
}

# The pool principal for the repository may impersonate the account.
resource "google_service_account_iam_member" "github_release_impersonation" {
  service_account_id = google_service_account.github_release.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/${google_iam_workload_identity_pool.github.workload_identity_pool_id}/attribute.repository/${var.github_repository}"
}

# Push the promoted image (and its cosign signature manifest) by digest.
resource "google_artifact_registry_repository_iam_member" "github_release_writes" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_release.email}"
}

# Sign with the image signing key and read its public key (cosign's
# gcpkms:// provider fetches the public key before signing).
resource "google_kms_crypto_key_iam_member" "github_release_signs" {
  crypto_key_id = google_kms_crypto_key.image_signer.id
  role          = "roles/cloudkms.signerVerifier"
  member        = "serviceAccount:${google_service_account.github_release.email}"
}

resource "google_kms_crypto_key_iam_member" "github_release_reads_key" {
  crypto_key_id = google_kms_crypto_key.image_signer.id
  role          = "roles/cloudkms.viewer"
  member        = "serviceAccount:${google_service_account.github_release.email}"
}

output "github_release_workload_identity_provider" {
  description = "For google-github-actions/auth: the full provider resource name."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "github_release_service_account" {
  description = "For google-github-actions/auth: the account the workflow impersonates."
  value       = google_service_account.github_release.email
}
