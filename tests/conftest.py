"""Shared pytest configuration for the deepseek-tools test suite.

**Why this file exists.** pytest creates every ``tmp_path`` with
``mode=0o700`` and, more importantly, creates its base temporary directory
during ``pytest_configure`` -- before a confined file sandbox has granted
workspace write access. On Windows that early directory receives a
restrictive ACL, so it can never be listed again and every ``tmp_path``
fixture then fails with ``PermissionError``.

This module makes pytest create its base temporary directory lazily, on first
actual use (by which time the sandbox grant is in place), and with inherited
permissions rather than ``mode=0o700``. ``tmp_path`` and ``tmp_path_factory``
otherwise behave exactly as documented.
"""

from __future__ import annotations

import itertools
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest


def _probe(path: Path) -> bool:
    """Return True when *path* can be created, listed, written to and removed."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        list(path.iterdir())
        probe.unlink()
        return True
    except OSError:
        return False


def _candidate_roots() -> list[Path]:
    """Writable temporary roots to try, most specific first.

    A fresh, uniquely named root per session keeps a stale or
    permission-locked directory from an earlier run from poisoning this one.
    """
    suffix = f"{os.getpid()}-{int(time.time() * 1000) % 1_000_000}"
    roots: list[Path] = []
    workspace = os.environ.get("DSH_WORKSPACE")
    if workspace:
        roots.append(Path(workspace) / f".pytest-tmp-{suffix}")
    roots.append(Path.cwd() / f".pytest-tmp-{suffix}")
    profile = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if profile:
        roots.append(Path(profile) / f".deepseek-tools-pytest-{suffix}")
    return roots


def _patch_pytest_tempdir() -> None:
    """Make pytest's temporary directories usable under a file sandbox.

    The stock ``make_numbered_dir`` creates directories with ``mode=0o700``
    and derives uniqueness by listing the root. Both are replaced: the new
    implementation uses inherited permissions and a monotonic counter, so it
    neither produces an unlistable directory nor depends on listing one.
    """
    from _pytest import pathlib as _pathlib
    from _pytest import tmpdir as _tmpdir

    counter = itertools.count()

    def make_numbered_dir(root, prefix, mode=0o700):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            candidate = root / f"{prefix}{next(counter)}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate
        raise OSError(f"could not create numbered dir with prefix {prefix} in {root}")

    _pathlib.make_numbered_dir = make_numbered_dir
    _tmpdir.make_numbered_dir = make_numbered_dir

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        basename = self._ensure_relative_to_basetemp(basename)
        if not numbered:
            path = self.getbasetemp().joinpath(basename)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = make_numbered_dir(root=self.getbasetemp(), prefix=basename)
            self._trace("mktemp", path)
        return path

    _tmpdir.TempPathFactory.mktemp = mktemp


def _patch_cleanup() -> None:
    """Make pytest's teardown fast and tolerant of unreadable temp trees.

    Two hazards are neutralised. ``cleanup_dead_symlinks`` re-lists the base
    temp directory, which a confined process may not be allowed to do; and
    ``rm_rf`` can block for a long time walking a tree with restricted
    entries. Both are made best-effort so teardown never hangs or masks
    results.
    """
    from _pytest import pathlib as _pathlib
    from _pytest import tmpdir as _tmpdir

    def safe_cleanup(root) -> None:
        try:
            list(Path(root).iterdir())
        except OSError:
            # Unreadable in this security context: nothing safe to sweep.
            return

    def safe_rm_rf(path, *args, **kwargs) -> None:
        """Remove a tree best-effort, never raising and never hanging.

        ``shutil.rmtree(ignore_errors=True)`` retries fewer times than pytest's
        implementation and gives up instead of blocking on a restricted entry.
        """
        shutil.rmtree(path, ignore_errors=True)

    for module in (_pathlib, _tmpdir):
        if hasattr(module, "cleanup_dead_symlinks"):
            module.cleanup_dead_symlinks = safe_cleanup
        if hasattr(module, "rm_rf"):
            module.rm_rf = safe_rm_rf


def _patch_temp_factory() -> None:
    """Replace pytest's base-temp creation with a sandbox-safe version.

    The stock ``getbasetemp`` runs ``basetemp.mkdir(mode=0o700)`` the first
    time it is called, and for the implicit path it also builds a predictable
    ``pytest-of-<user>`` directory under the system temp root. On Windows the
    ``0o700`` mode yields a directory a confined process can never re-list, and
    a stale directory of that name from an earlier security context blocks
    every later run.

    This replacement creates the base temp directory under the workspace with
    inherited permissions, and reuses it for the whole session.
    """
    from _pytest import tmpdir as _tmpdir

    def getbasetemp(self) -> Path:
        if self._basetemp is not None:
            return self._basetemp

        given = self._given_basetemp
        if given is not None:
            root = Path(given)
        else:
            # Plain mkdir with a unique name; avoid tempfile.mkdtemp, which
            # routes back through pytest's own numbered-dir machinery.
            parent = _root_parent()
            stamp = f"{os.getpid()}-{int(time.time() * 1000) % 1_000_000}"
            root = parent / f"pytest-{stamp}"
        root.mkdir(parents=True, exist_ok=True)

        try:
            root = root.resolve()
        except OSError:
            pass
        self._basetemp = root
        self._trace("new basetemp", root)
        return root

    _tmpdir.TempPathFactory.getbasetemp = getbasetemp


def _root_parent() -> Path:
    """Directory in which lazily created temporary roots are placed."""
    workspace = os.environ.get("DSH_WORKSPACE")
    parent = Path(workspace) if workspace else Path.cwd()
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def pytest_configure(config: pytest.Config) -> None:
    """Install sandbox-safe temporary directory handling.

    The base temporary directory is deliberately *not* created here: this hook
    runs before a confined sandbox has granted workspace write access, and a
    directory created at that moment is born unlistable. Creation happens
    lazily on first use instead.
    """
    _patch_pytest_tempdir()
    _patch_cleanup()
    _patch_temp_factory()
