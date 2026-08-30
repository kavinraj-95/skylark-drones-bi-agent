"""Deployment manifests must not drift apart.

`pyproject.toml` drives local development; `requirements.txt` drives Streamlit
Community Cloud. If they disagree, the hosted app runs different code from the one
that was tested - the kind of failure that only shows up in front of an evaluator.
"""

from __future__ import annotations

import re
import tomllib

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _requirements() -> dict[str, str]:
    specs = {}
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, spec = re.split(r"([<>=!~]+)", line, maxsplit=1)[0], "", line
        specs[name.strip().lower()] = spec.strip()
    return specs


def _pyproject_dependencies() -> dict[str, str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    specs = {}
    for entry in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~\[]", entry, maxsplit=1)[0]
        specs[name.strip().lower()] = entry.strip()
    return specs


def test_runtime_dependencies_match():
    assert _requirements() == _pyproject_dependencies()


def test_app_entrypoint_exists():
    assert (REPO / "app.py").is_file()


def test_package_sits_beside_the_entrypoint():
    """Streamlit Cloud runs from the repo root, so the package must import with no
    path manipulation and no install step."""
    assert (REPO / "skylark_bi" / "__init__.py").is_file()
    app = (REPO / "app.py").read_text(encoding="utf-8")
    assert "sys.path" not in app, "app.py should not need to patch sys.path"


def test_python_version_is_pinned_to_one_version():
    version = (REPO / ".python-version").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"3\.\d+", version), f"expected a single minor version, got {version!r}"


def test_no_secrets_committed():
    """Checks git, not the filesystem.

    A developer running locally *should* have a .env, and may keep a
    .streamlit/secrets.toml. Neither may ever be tracked. Asserting they do not exist
    on disk would fail on any working machine while catching nothing that matters.
    """
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git repository")

    tracked = set(result.stdout.splitlines())
    assert ".env" not in tracked
    assert ".streamlit/secrets.toml" not in tracked
