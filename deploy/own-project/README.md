# Released slots in your own project

Runs the released enclave image on your own `a3-highgpu-1g` machines, in your VPC, with [`terraform/modules/slot`](../../terraform/modules/slot). The image is the release, by digest; what this deployment sets is what the launch policy leaves to any deployment: the slots' names and addresses, the controller they report to, and where they read the weights. All of it is attested, and the slots keep no record of a session.

## Before you apply

1. **Quota and the image.** `a3-highgpu-1g` Spot capacity with Confidential Computing (one H100 each) in the zones you choose, and the Confidential Space image (`confidential-space-images`, family `confidential-space`).
2. **Approval.** The weights and the analysis bundle are released only to a slot in a project the operator approved, running as an approved VM account, attesting the controller the approval names (`terraform/hosted.tf`). The account is `enclave-slot-vm@<your project>.iam.gserviceaccount.com` (the module's `name_prefix` is `enclave`). Until the approval is applied, a slot boots and stops: it cannot read the weights.
3. **The values.** Copy `terraform.tfvars.example` to `terraform.tfvars`. The controller's values come from the service your project was approved for, and the rest from the operator's `hosted_slots` output.

## Apply

```sh
terraform init
terraform apply
```

Then point each slot's DNS name at its address (`terraform output slots`): each boot obtains its own certificate for that name from Google Trust Services, through an account binding the enclave mints in your project with its attested identity.

## Verify

`terraform output verifier` gives the values these slots attest about where they run. With a slot up, run [`tee-verify`](../../verifier) with them, or with your controller's published policy (`-policy-url`). It passes only for a production enclave of a released image, signed, with its SLSA provenance, on this hardware ([`VERIFY.md`](../../VERIFY.md), "Which values to verify against").

## What owning the project gives you

Starting, stopping and naming the machines, and the variables above. Nothing of a session: the devices and the controller hold each slot to the release's public record, to the TLS key and the seal key its attestation names, and to its evidence key, which signs every reading it sends (`X-Masseuse-Signature`).

## Leaving

`terraform destroy` removes the machines, their addresses and their account. The operator removing your project from `hosted.tf` stops every slot of yours from reading the weights at its next exchange.
