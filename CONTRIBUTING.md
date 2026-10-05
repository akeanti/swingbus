# Contributing

Thanks for taking the time. The most useful contributions are bug reports with a case that reproduces them, comparisons against other tools, and new analyses that come with a way to validate them.

## Setup

```bash
git clone https://github.com/akeanti/swingbus
cd swingbus
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Checks

```bash
python -m pytest
ruff check .
ruff format --check .
mypy
```

`pytest -m oracle` runs only the comparisons with PYPOWER, and `pytest -m network` runs the tests that download from PGLib-OPF. CI runs everything on Linux, macOS and Windows with Python 3.10 to 3.14, against the oldest supported NumPy and SciPy, and against their pre-releases.

## Guidelines

- Every numerical change needs a test that compares against an independent result: PYPOWER, MATPOWER, a published case solution, or a physical identity such as power balance.
- Runtime dependencies stay limited to NumPy and SciPy. Optional features go behind an extra.
- Data tables use MATPOWER's column layout, and `swingbus.idx` names the columns. New results should keep that layout so solved cases can be written back to `.m` files.
- The source does not use comments or docstrings. Names carry the meaning, the mathematics lives in [docs/theory.md](docs/theory.md), and the tests show how each function is meant to be used. If a piece of code needs an explanation, it usually needs a better name or a section in the theory page.
- Keep pull requests focused, and add a line to `CHANGELOG.md` for anything a user would notice.

## Releasing

1. Update the version in `src/swingbus/__init__.py`, `CITATION.cff` and `CHANGELOG.md`.
2. Tag the commit as `vX.Y.Z` and publish a GitHub release from the tag.
3. The `Release` workflow checks that the tag matches the version, builds the sdist and wheel, publishes them to PyPI through trusted publishing, and attaches them to the release.

Publishing to PyPI needs a one-time setup: on pypi.org, add a trusted publisher for the `akeanti/swingbus` repository with workflow `release.yml` and environment `pypi`, then set the repository variable `PUBLISH_TO_PYPI` to `true`. Until then, releases still build and attach the distributions but skip the upload.

Archiving releases on Zenodo gives every version a DOI. Enable the repository once at zenodo.org under GitHub integration; Zenodo reads the metadata from `CITATION.cff` when the next release is published.
