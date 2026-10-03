# Kaldi filterbank vendor

Source: <https://github.com/pytorch/audio/blob/v2.11.0/src/torchaudio/compliance/kaldi.py>

The exact source revision (the v2.11.0 tag) is recorded in `UPSTREAM_COMMIT`;
`LICENSE` is the upstream BSD 2-Clause license. BEATs computes its input
features with this module's `fbank`, and the image ships no torchaudio, so
the file is carried here: pure torch, no compiled extension.

FemLed's patch removes, and changes nothing that remains:

1. `kaldi.py` ends after `fbank`: `_get_dct_matrix`, `_get_lifter_coeffs` and
   `mfcc` (the only code that needed `torchaudio` itself) are dropped, with
   the `import torchaudio` line and `"mfcc"` in `__all__`.

`__init__.py` is new.
