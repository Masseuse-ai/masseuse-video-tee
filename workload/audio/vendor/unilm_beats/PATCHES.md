# BEATs vendor

Source: <https://github.com/microsoft/unilm/tree/master/beats>

The exact source revision is recorded in `UPSTREAM_COMMIT`; `LICENSE` is the
upstream MIT license (the repository root's). `BEATs.py`, `backbone.py` and
`modules.py` are the model code the fine-tuned AudioSet checkpoint needs at
inference; the tokenizer and quantizer files, used only in pre-training, are
not vendored. Upstream publishes no package release.

FemLed carries two import-only patches, so that the three files load as a
package (`vendor.unilm_beats`) without torchaudio, which the image does not
ship:

1. `BEATs.py` imports its filterbank from `..kaldi_fbank` (torchaudio's
   `compliance/kaldi.py`, vendored beside this folder) instead of
   `torchaudio.compliance.kaldi`, and `TransformerEncoder` from `.backbone`
   instead of `backbone`.
2. `backbone.py` imports from `.modules` instead of `modules`.

`__init__.py` is new. Nothing else differs from upstream: the model, its
preprocessing and its numerics are upstream's.

The checkpoint is not committed. It is fetched by
`workload/tee/Dockerfile.tee` (the `beats` stage) at a pinned revision of a
byte-identical Hugging Face upload and checked against its SHA-256:

```
BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
sha256 e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c
```
