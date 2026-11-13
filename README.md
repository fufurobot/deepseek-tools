# deepseek-tools

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/fufurobot/deepseek-tools)

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
| `tmux-manager` | `tmux_manager` | HTML / RST / Table Renderer DSL tables -> tmux panes (plus a Streamlit UI) |

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

## tmux-manager

`python -m deepseek_tools.tmux_manager table.html -o setup.sh` turns a table
into a tmux layout; `python -m deepseek_tools.tmux_manager web` opens the
Streamlit editor (input pickers, size editors, DSL / script / pixel-box
output and a download). Every input (HTML, RST, DSL) is first converted to
Table Renderer DSL text, then laid out; panes and pixel boxes use the same
resolved integer sizes.

Size hints are either **relative** (`30 percentage` / `width="30%"`: a share
of the whole pane; unconstrained tracks split the rest; over 100% is an
error) or **absolute** (`12 pixels`, `3 em`, HTML `width="12"`: literal
character cells; `-` marks the flexible track taking what is left; absolute
tables shrink to their sizes). Mixing is allowed: a percentage next to
absolute sizes wins, growing the table so it can hold its share.

Known gaps, marked `#FIXME`/`#TODO` in the code: the assumed 80x24 pane for
HTML tables is not configurable yet, a pane side with zero cells shares its
parent pane instead of being reported, and the tablet size editor handles
one-row tables.
