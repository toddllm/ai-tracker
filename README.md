# AI Tracker (Terminal)

Terminal-first AI tracker that aggregates updates from:
- X (optional, requires `X_BEARER_TOKEN`)
- arXiv
- AI newsletters

It runs as a live terminal dashboard and:
- starts immediately with an active TUI, then refreshes in the background
- refreshes feed data hourly (default)
- keeps a real-time Ollama model list (size-sorted)
- analyzes top feed items using local Ollama model `nemotron-3-nano` by default
- generates a long-form Ollama markdown briefing with scroll support in the TUI
- renders briefing in two columns (main brief + snapshot sidebar fallback)
- shows a local Ollama chat sidebar in the brief second column (toggleable)
- allows optional compact ASCII/TUI-style mini charts in the briefing when useful
- grounds the snapshot editor with live feed context (ranked/latest items, source mix, links, recency)
- shows warnings in a compact bottom status bar to preserve feed/brief space
- ranks by newest or trending
- supports keyboard navigation and opening story links directly from the TUI

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional env config:

```bash
cp .env.example .env
```

## Run

Default (hourly refresh, trending sort, Ollama analysis with `nemotron-3-nano`):

```bash
python app.py
```

One-shot run (no hourly loop):

```bash
python app.py --once
```

Run newsletters-only, strict topic mode:

```bash
python app.py --sources newsletters --strict-topics --topics "agents, multimodal"
```

## Key Flags

- `--refresh-minutes 60` update interval (default `60`)
- `--ollama-model nemotron-3-nano` analysis model (default `nemotron-3-nano`)
- `--analysis-limit 15` number of items sent to Ollama each cycle
- `--model-poll-seconds 30` refresh interval for live Ollama model list
- `--sources newsletters,arxiv,x`
- `--sort newest|trending`
- `--lookback-days 15`
- `--max-items 25`
- `--newsletters all` or comma-separated names
- `--quick-filter "open-source"`

## Keyboard + Commands

Normal mode hotkeys:

- `/` enter command mode
- bottom line shows slash-command hint/input at all times
- `Tab` cycle focused section (`status`, `brief`, `feed`, `story`, `models`, `sources`, `menu`, `commands`)
- `Shift+Tab` cycle focused section backwards
- `j`/`k` next/previous story
- `Down`/`Up` next/previous story (or brief scroll when `brief` is focused)
- `PgDn`/`PgUp` scroll long Ollama brief
- `+` / `-` grow or shrink the focused row height (`status`, `brief`, or `body`)
- `Enter` open selected story link
- `o` open selected story link
- `b` toggle brief focus mode (full-width brief reader)
- `c` toggle local chat sidebar in the brief panel
- `[` / `]` resize right-hand panel narrower/wider
- `m` toggle menu panel
- `q` quit immediately

Command mode hotkeys:

- `Enter` submit typed slash command
- `Esc` cancel command mode
- `Backspace` edit command buffer
- printable keys append to command buffer

Slash commands:

Commands:

- `/help` show command list
- `/config` show active runtime config
- `/topics <csv>` set topics quickly
- `/sort <newest|trending>` change sorting quickly
- `/lookback <days>` set lookback window quickly
- `/max <items>` set item cap quickly
- `/model [list|index|name]` pick an Ollama model from local tags or set by name
- `/strict <on|off>` toggle strict topic mode quickly
- `/limit <n>` set analysis limit quickly
- `/set <key> <value>` update settings live
- `/sources <newsletters,arxiv,x>` change sources
- `/newsletters <all|csv names>` change newsletter set
- `/refresh` trigger immediate refresh now
- `/open [index]` open selected story (or a specific 1-based index)
- `/brief [focus|normal|toggle|top]` change brief focus mode or jump brief to top
- `/focus <section>` set focused section directly
- `/size <status|brief|body> <1..12>` set row height ratios directly
- `/layout right <2..6>` resize right panel via command
- `/export [md|json|csv] [path]` export current feed snapshot
- `/menu [on|off|toggle]` show/hide menu panel
- `/ask <question>` ask the local chat assistant using current feed context
- `/chat [on|off|toggle|model <name|auto>|clear]` control chat visibility/model/history
- `/clearlog` clear command log
- `/quit` or `/exit` stop the app

`/set` keys:
- `topics`
- `strict_topics`
- `lookback_days`
- `max_items`
- `sort` (`newest` or `trending`)
- `quick_filter`
- `refresh_minutes`
- `ollama_host`
- `ollama_model`
- `analysis_limit`
- `ollama_timeout_seconds`
- `model_list_size`
- `model_poll_seconds`

## Ollama Notes

- Start Ollama server:

```bash
ollama serve
```

- Ensure analysis model exists locally:

```bash
ollama pull nemotron-3-nano
```

The dashboard shows model availability warnings if the selected model is missing.

## Environment Variables

- `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
- `X_BEARER_TOKEN` (optional)

## Included Newsletter Feeds

- Import AI (Substack)
- Latent Space
- Last Week in AI
- SemiAnalysis
- Interconnects
- Understanding AI
- AI Tidbits
- One Useful Thing
