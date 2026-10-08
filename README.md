# deepseek-tools

[![CI](https://github.com/fufurobot/deepseek-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/fufurobot/deepseek-tools/actions/workflows/ci.yml)

My collection of deepseek-tools, built as a Python wheel.

## Tools

| Console script | Module | Purpose |
| --- | --- | --- |
| `clip-commit` | `clip_commit` | Commit using the clipboard as the message; amends when nothing is staged |
| `concat-repo` | `concat_repo` | Concatenate files by extension into one output file |
| `auto-commit` | `git_auto_commit` | Generate a commit message from the staged diff with a local Ollama model |
| `git-cloner` | `git_cloner` | Clone/update many repositories with namespace isolation |
| `ollama-finetuner` | `ollama_finetuner` | Fine-tune Ollama models via TRL (SFT / DPO / GRPO) |
| `rwkv7-trainer` | `rwkv7_trainer` | RWKV-7 pretraining plus GRPO fine-tuning |
| `pwcodegen` | `pwcodegen` | Playwright codegen in an isolated environment |
| `deepseek-chat-export` | `deepseek_chat_export` | Export a chat.deepseek.com share page |

## Setup

```sh
uv sync --group dev          # runtime + test dependencies
uv sync --group dev --group gui   # adds pyautogui (needs a desktop session)
```

## Testing

```sh
uv run pytest
```

The suite runs on Linux and Windows in CI across Python 3.10 and 3.12.

`tests/conftest.py` contains Windows-only workarounds for running inside a
confined file sandbox: pytest's default `mode=0o700` temporary directories
become unlistable to a restricted token, which previously made every
`tmp_path` test fail. See [docs/sandbox-notes.md](docs/sandbox-notes.md) for
the full mechanism. On other platforms pytest's stock behaviour is used.

## Configuration

Copy `.env.example` to `.env` and fill in the values. `.env` is git-ignored
and must never be committed.

| Variable | Purpose |
| --- | --- |
| `GITHUB_TOKEN` | Fine-grained PAT for publishing issues/PRs and reading CI status |
| `GITHUB_REPOSITORY` | `owner/name` the tools publish to |
| `OLLAMA_API_BASE` | Ollama HTTP endpoint (default `http://localhost:11434`) |

Check the configured token's identity and effective permissions without ever
printing the secret:

```sh
uv run python scripts/check_github_token.py
```
