"""Proof that no business data is baked into the application.

The assignment is explicit: the agent must query monday.com dynamically and must not
hardcode the CSV data. These tests assert that mechanically rather than asking anyone
to take it on trust.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "skylark_bi"
REPO = Path(__file__).resolve().parents[1]

#: Constants the application is allowed to contain. Each is a declared modelling
#: choice, documented at its definition - never an observation about this dataset.
ALLOWED_NUMERIC_CONSTANTS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 100,          # indexes, months, small bounds
    0.0, 1.0, 0.2, 0.5, 0.8,                                 # probability weights
    0.86,                                                     # fuzzy-match cutoff
    0.15,                                                     # concentration threshold
    15, 20, 24, 25, 30, 40, 45, 60, 45.0,                    # timeouts, retries, limits
    64, 120, 200, 400, 401, 403, 429, 500, 800, 900,         # HTTP, budgets, lengths
    1024, 2048, 4096, 8192, 1200, 1e5, 1e7, 3, 2.0,
}


def python_files():
    return sorted(PACKAGE.rglob("*.py"))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids() of string constants that are docstrings."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


def code_strings(path: Path) -> list[str]:
    """Every string literal the code actually *uses*.

    Docstrings are excluded, and comments never reach the AST at all. That is the
    right boundary for this check: explaining in prose why `Sakura` appears 27 times
    is documentation of a design decision, whereas the same text in a live string
    literal would be business data baked into the program.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


class TestNoBusinessDataInSource:
    def test_no_masked_identifiers_appear_in_code(self):
        """Client codes, owner codes and work order serials are data, not constants."""
        patterns = [
            re.compile(r"\bCOMPANY\d{3}\b"),
            re.compile(r"\bWOCOMPANY_\d{3}\b"),
            re.compile(r"\bOWNER_\d{3}\b"),
            re.compile(r"\bSDPLDEAL-\d{3}\b"),
            re.compile(r"SDPL/FY\d{2}-\d{2}/\d{3}"),
        ]
        offenders = []
        for path in python_files():
            for literal in code_strings(path):
                for pattern in patterns:
                    if pattern.search(literal):
                        offenders.append(f"{path.relative_to(REPO)}: {literal!r}")
        assert not offenders, f"Business identifiers embedded in source: {offenders}"

    def test_no_masked_deal_names_in_code(self):
        """The masked deal names are cartoon characters; none belong in source."""
        names = ("Sakura", "Naruto", "Sasuke", "Alphonse", "Scooby-Doo", "Mojo Jojo",
                 "Stewie Griffin", "Ben Tennyson", "Bugs Bunny", "Tanjiro", "Subaru")
        offenders = [
            f"{path.relative_to(REPO)}: {literal!r}"
            for path in python_files()
            for literal in code_strings(path)
            for name in names
            if name in literal
        ]
        assert not offenders, f"Deal names embedded in source: {offenders}"

    def test_no_large_numeric_literals(self):
        """A number like 732487329 could only have come from this dataset."""
        offenders = []
        for path in python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    value = node.value
                    if isinstance(value, bool) or value in ALLOWED_NUMERIC_CONSTANTS:
                        continue
                    if abs(value) > 2100:  # above a plausible year
                        offenders.append(
                            f"{path.relative_to(REPO)}:{node.lineno}: {value}"
                        )
        assert not offenders, f"Unexplained large literals: {offenders}"

    def test_application_never_reads_the_import_csvs(self):
        """`setup/import/` holds one-time import artifacts. The app must not read them."""
        markers = ("setup/import", "monday_deals.csv", "monday_work_orders.csv")
        offenders = [
            f"{path.relative_to(REPO)}: {literal!r}"
            for path in python_files()
            for literal in code_strings(path)
            if any(marker in literal for marker in markers)
        ]
        assert not offenders, f"Application reads import artifacts: {offenders}"

    def test_no_csv_parsing_in_the_application(self):
        """Data arrives over the monday.com API, never from a file."""
        offenders = [
            str(path.relative_to(REPO))
            for path in python_files()
            if re.search(r"^\s*import csv\b|^\s*from csv\b", path.read_text(encoding="utf-8"), re.M)
        ]
        assert not offenders, f"CSV parsing in application code: {offenders}"


class TestSnapshotIsNotCommitted:
    """The outage fallback must never look like hardcoded data."""

    def test_no_snapshot_file_is_committed(self):
        """Tracked by git - not merely absent from disk.

        Running the app locally *should* leave a snapshot in `.cache/`; that is the
        outage fallback working. What must never happen is one being committed, where
        an evaluator could mistake it for hardcoded business data. So the check is
        against git's index, not the filesystem.
        """
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:  # not a git checkout (e.g. extracted from a ZIP)
            pytest.skip("not a git repository")

        offenders = [
            line for line in result.stdout.splitlines()
            if line.endswith(".snapshot.json") or line.startswith(".cache/")
        ]
        assert not offenders, f"A runtime snapshot is committed or untracked-but-visible: {offenders}"

    def test_cache_directory_is_gitignored(self):
        ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
        assert ".cache/" in ignored
        assert "*.snapshot.json" in ignored

    def test_snapshot_is_only_written_from_a_live_response(self):
        """There is no code path that seeds a snapshot from anything but the API."""
        source = (PACKAGE / "agent" / "service.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        writers = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_write_snapshot"
        ]
        assert len(writers) == 1

        callers = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_write_snapshot"
        ]
        assert len(callers) == 1, "snapshot written from more than one place"
        # ...and that one place is the live fetch.
        assert "_write_snapshot" in source.split("def _fetch_live")[1].split("def ")[0]


class TestFixturesAreTestOnly:
    def test_fixtures_live_under_tests(self):
        """Deleting the fixtures must not change production behaviour."""
        conftest = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
        assert "setup" in conftest  # the fixtures read the CSVs...
        for path in python_files():
            assert "conftest" not in path.read_text(encoding="utf-8"), (
                f"{path} imports test fixtures"
            )

    def test_package_imports_without_test_data_present(self):
        """The package must not require any data file at import time."""
        import importlib

        for module in (
            "skylark_bi.config", "skylark_bi.monday.client", "skylark_bi.ingest.builder",
            "skylark_bi.analytics.metrics", "skylark_bi.analytics.engine",
            "skylark_bi.agent.resolver", "skylark_bi.agent.responder",
            "skylark_bi.agent.service",
        ):
            assert importlib.import_module(module) is not None
