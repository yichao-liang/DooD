"""Smoke tests for the `dood` package."""

from dood import __version__


def test_version_is_semver_string() -> None:
    """The package exposes a MAJOR.MINOR.PATCH version string."""
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
