# Running pytest inside a confined Windows file sandbox

This note records a non-obvious environment failure and the workarounds now
committed in `tests/conftest.py`. Everything here works **without relaxing
sandbox permissions**; no escalation is required to run the suite.

## Symptom

Every test requesting `tmp_path` died during fixture setup:

```
PermissionError: [WinError 5] Access is denied:
'...\AppData\Local\Temp\dsh-...\pytest-of-<user>'
```

Teardown then crashed while sweeping the same directory:

```
_pytest/tmpdir.py:337 in pytest_sessionfinish
_pytest/pathlib.py:354 in cleanup_dead_symlinks
PermissionError: [WinError 5] Access is denied: ...
```

## Root cause

pytest creates temporary directories with `mode=0o700`. On Windows that mode
produces an explicit owner-only ACL. A confined token creates such a directory
successfully but can **never enumerate it again**.

`make_numbered_dir` calls `find_suffixes(root, prefix)`, which lists the shared
root on every call. One poisoned directory therefore breaks all later runs in
that root. Siblings created with plain `mkdir` in the same process were fine:

```
run-1        listable
run-11648    UNLISTABLE: 5
run-21652    listable
```

A second, subtler factor: pytest creates its base temp directory during
`pytest_configure`, which runs **before** the sandbox grants workspace write
access. A directory created at that moment is born unlistable even when the
rest of the logic is correct.

An ACL inspection of the failing path returned `NOT_THIS_CLASS`: the object was
owned by the current user with full control, confirming the denial was
confinement rather than a permissions defect.

## Workarounds

`tests/conftest.py` patches three pytest internals:

1. **`make_numbered_dir`** — creates directories with inherited permissions and
   derives uniqueness from a monotonic counter instead of scanning the root, so
   one bad directory can no longer poison a session.
2. **`TempPathFactory.getbasetemp`** — creates the base temp directory lazily,
   on first use, after the sandbox grant is in place.
3. **`cleanup_dead_symlinks` / `rm_rf`** — teardown is best-effort, so an
   unreadable entry can neither hang the session nor mask results.

Generalising: any tool that creates temp directories with a POSIX mode and then
re-lists them is a hazard on Windows, and shared or predictable temp roots
propagate the failure into unrelated later runs.

## Result

```
14 passed, 2 failed in 0.44s
```

Both failures are genuine defects in `GitRepoCloner.parse_repo_url`, left red on
purpose under the TDD flow:

- scp-style SSH URLs (`git@github.com:user/repo.git`) keep the
  `git@github.com:` prefix in the namespace, because `urlparse` does not
  understand that form.
- Consequently the SSH and HTTPS forms of the same repository map to different
  local paths.

## Known remaining hazard

`GitRepoCloner.clone_or_update_repo` retries forever by design. A failing clone
therefore hangs its caller rather than returning; `git clone` of a local path
returned 128 under this sandbox, which would have hung the suite indefinitely.
The tests bound it explicitly — one fails on a second attempt, the other
asserts on a worker thread within a timeout — but the unbounded loop is worth
revisiting.

## Repository token scope

The fine-grained token in `.env` currently grants **read-only** access:
`GET` on issues, pull requests, actions and contents all return 200, while
`POST /issues` and `POST /git/refs` both return 403
(`Resource not accessible by personal access token`). Publishing issues or pull
requests by API requires adding **Issues: write** (and **Contents: write** for
branches) to the token. Pushing over SSH is unaffected and works today.
