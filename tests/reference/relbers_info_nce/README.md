# RElbers InfoNCE reference (test only)

`info_nce.py` is an unmodified copy of `info_nce/__init__.py` from
[RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch/blob/a2a9c126260acb655472e4836c5d96805fae3749/info_nce/__init__.py),
commit `a2a9c126260acb655472e4836c5d96805fae3749`, retrieved 2026-09-24.
The upstream MIT license is retained in `LICENSE`.

SHA-256 of the original downloaded bytes:

- `info_nce.py`: `63403d7ea99ddbda9812cec482acd531033f50f8a8efc0c33ef2adb76616b297`
- `LICENSE`: `9775b93b5414cb2cbb97da7d2967e689c94eb7bde6ab25f393ec777baa3c2d60`

This fixture is used for offline loss-value and query-gradient comparisons.
Production code does not import it. Tests explicitly detach source keys and
filter negatives before calling upstream when comparing the global helper.
PatchNCE comparisons call upstream once per image to preserve within-image
negatives. CUT's norm-plus-epsilon normalization and finite diagonal mask
are documented numerical differences, not changes made to this fixture.
