# ---------------------------------------------------------------------------
# The models bucket
#
# A copy of the weights prefixes of the current producer bucket (README:
# `gcloud storage cp -r` of hf-bf16/, detectors/ and the artifact file).
# Readable by the attested image digest(s) through the WIF pool and by no
# service account: the operator can upload weights, and can grant nothing
# a running enclave would not already prove about itself.
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "models" {
  project                     = var.project_id
  name                        = "${var.project_id}-models"
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = false
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "models_wif_reader" {
  for_each = local.wif_principals
  bucket   = google_storage_bucket.models.name
  role     = "roles/storage.objectViewer"
  member   = each.value
}

# Operators upload the weights; nothing else in the project needs to.
resource "google_storage_bucket_iam_member" "models_operator_admin" {
  for_each = toset(var.operator_members)
  bucket   = google_storage_bucket.models.name
  role     = "roles/storage.objectAdmin"
  member   = each.value
}
