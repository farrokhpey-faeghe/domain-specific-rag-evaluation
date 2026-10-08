# Corpus (WS1, feat/data)

A frozen snapshot of the university's public help pages. Everything here is committed (the repo is private).

- `sources.csv`: one row per page, with columns `doc_id,url,section`. The `doc_id` is a short stable slug such as `it-password-reset`.
- `docs/<doc_id>.md`: one file per page, written by `deskrag data snapshot` and then cleaned by hand. Each file starts with front-matter:

  ```markdown
  ---
  doc_id: it-password-reset
  title: Resetting your password
  url: https://...
  section: IT
  snapshot_date: 2026-10-10
  ---
  # Resetting your password
  ...
  ```

- `MANIFEST`: the corpus fingerprint (`corpus_sha256` in `deskrag.core.corpus`), written once the snapshot is frozen.

Rules:
- Snapshot once, politely: 1 request per second, and skip any URL that robots.txt disallows.
- After the freeze, every corpus change goes through a PR that also re-runs and re-promotes the baselines. The dataset and corpus fingerprints change, so old runs stop being comparable.

See `data/samples/corpus/` for five small fictional examples in the same format.
