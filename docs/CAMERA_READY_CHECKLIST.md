# Camera-Ready Repository Checklist

Use this checklist before sharing the repository in poster slides, the workshop page, or the camera-ready PDF.

## Metadata and Links

- [ ] Confirm the paper title in `README.md` and `CITATION.cff` matches the final camera-ready PDF.
- [ ] Confirm author order, affiliations, and corresponding author email in `CITATION.cff`.
- [ ] Verify OpenReview, workshop, and project links are active and public.

## Reproducibility

- [ ] Confirm all commands in `README.md` run successfully in a clean environment.
- [ ] Check that `requirements.txt` and `setup.py` remain in sync after dependency changes.
- [ ] Ensure generated demo assets are reproducible from source scripts and manifest.

## Repository Hygiene

- [ ] Remove macOS artifact files before pushing:
  - Dry run: `bash scripts/cleanup_macos_artifacts.sh --dry-run`
  - Apply cleanup: `bash scripts/cleanup_macos_artifacts.sh --apply`
- [ ] Ensure no private data, credentials, or local absolute paths are tracked.
- [ ] Confirm large files are intentional and documented.

## Release Steps

- [ ] Tag a release version for the camera-ready snapshot.
- [ ] Add final citation details (venue metadata, author list, DOI/arXiv when available).
- [ ] Create a concise release note summarizing accepted-poster scope and limitations.
