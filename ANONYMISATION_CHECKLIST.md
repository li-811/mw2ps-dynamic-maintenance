# Double-blind release checklist

Before creating the anonymous review repository:

- [ ] Upload the files from this package into a new repository with no prior Git history.
- [ ] Do not reuse commits from a personal repository.
- [ ] Keep author names, affiliations and email addresses out of README/issues/releases.
- [ ] Do not upload local IDE settings, shell history, `.git/`, caches or virtual environments.
- [ ] If using Git locally, configure an anonymous review-only user name/email before the first commit.
- [ ] Add the original `pilot_results.csv` if available.
- [ ] Run `python make_paper_outputs.py` once after upload and compare the seven generated figures.
- [ ] Replace the manuscript placeholder with the anonymous review URL only after the repository passes this audit.
- [ ] After acceptance, replace anonymous authorship/licence metadata and archive a permanent release.
