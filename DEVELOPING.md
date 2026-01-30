# Developing dbos-openai

## Setup

```bash
uv sync --dev
```

## Running Tests

```bash
uv run pytest tests/
```

## Versioning

Versions are automatically generated from git tags and branches:

| Branch | Version Format | Example |
|--------|---------------|---------|
| `release/v*` | Release | `0.2.0` |
| `main` | Alpha | `0.2.0a6` |
| Feature branch | Alpha + hash | `0.2.0a3+g1a2b3c4` |

The version logic is in `publish/hatch_build.py`.

## Releasing

1. Ensure you're on `main` and up-to-date:

```bash
git checkout main
git pull
```

2. Run the release script:

```bash
python publish/make_release.py                 # Auto-generate next version
python publish/make_release.py --version 0.2.0 # Or specify version
```

This creates a git tag and `release/v{version}` branch, then pushes both.

3. Go to GitHub Actions and run the **Publish to PyPI** workflow from the release branch.

## Testing a Pre-release

Publish from `main` to get an alpha version, then install with:

```bash
pip install --pre dbos-openai
```
