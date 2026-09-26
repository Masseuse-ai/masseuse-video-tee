# ---------------------------------------------------------------------------
# The certificate authority: Google Trust Services via Cloud Public CA
#
# GTS's ACME directory admits only accounts that present an External
# Account Binding, and an EAB is minted with externalAccountKeys.create on
# this project. Each boot of a slot is a new ACME account (the key and the
# certificate live on the enclave's tmpfs and die with it), so the mint
# happens inside the enclave at boot, with the same federated identity that
# reads the models bucket: the attested image digest on TDX with the GPU in
# CC mode. No service account holds this role. An operator can therefore
# not obtain an EAB and register an account that impersonates a slot, and
# the WIF provider's condition (wif.tf) decides who is a slot.
#
# EAB keys are single-use and expire unused after seven days; the project's
# quota is 120 requests a minute on this API and 100 new ACME accounts an
# hour on the directory (README "Certificate authority"), against one boot
# per session.
# ---------------------------------------------------------------------------
resource "google_project_iam_member" "wif_mints_eab" {
  for_each = local.wif_principals
  project  = var.project_id
  role     = "roles/publicca.externalAccountKeyCreator"
  member   = each.value

  depends_on = [google_project_service.apis]
}
