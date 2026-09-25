# Official InfoOT audit reference

- Repository: https://github.com/chingyaoc/InfoOT
- Revision: `352efd202f5b475dc170a8d08a99049689d5ee1a`
- Audited: 2026-09-25
- Source: https://github.com/chingyaoc/InfoOT/blob/352efd202f5b475dc170a8d08a99049689d5ee1a/infoot.py
- License: MIT; original notice retained in `LICENSE`.
- SHA-256 of `infoot.py`, normalizing CRLF to LF:
  `0d636765545256a0062ec697cc2e88ca17cc9ca40f6e185999eaf1b93ff2253b`

The solver source is unchanged. This copy is imported only by audit tests; it
is not a runtime dependency of training. Tests require the already declared
NumPy, SciPy, scikit-learn, tqdm and POT dependencies. No test downloads source.

The full upstream tree has five files: `LICENSE`, `README.md`, `infoot.py`,
`domain_adapt.py`, and `retrieval.py`. All were reviewed. The experiment scripts
were not executed because they need Office-Caltech data; their fixed-feature,
label-based evaluations are not image-translation training.

The out-of-sample `project` branch contains undefined/unbound names. The audit
records that failure and checks our readout against the working upstream
`conditional_score` followed by `projection` instead. Tests also exercise the
working in-sample `project` branches and both solver classes.
