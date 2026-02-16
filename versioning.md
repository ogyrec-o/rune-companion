# Versioning policy

This project uses Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`.

- **PATCH**: bug fixes, internal refactors, no API breakage.
- **MINOR**: new features, backwards compatible changes.
- **MAJOR**: breaking changes in public APIs or behavior.

## Release process

1. Update `CHANGELOG.md` (move items from Unreleased to a new version section).
2. Bump `version` in `pyproject.toml`.
3. Create a Git tag: `vMAJOR.MINOR.PATCH`.
4. Push the tag to GitHub.

GitHub Actions will build and attach `dist/*` artifacts to the release.
