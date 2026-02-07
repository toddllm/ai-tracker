# AI Tracker (Terminal)

Terminal-first AI tracker that aggregates updates from:
- X (optional, requires `X_BEARER_TOKEN`)
- arXiv
- AI newsletters

It runs as a live terminal dashboard and:
- starts immediately with an active TUI, then refreshes in the background
- refreshes feed data hourly (default)
- keeps a real-time Ollama model list (size-sorted)
- analyzes top feed items using local Ollama model `gpt-oss:120b` by default
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

Default (hourly refresh, trending sort, Ollama analysis with `gpt-oss:120b`):

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
- `--ollama-model gpt-oss:120b` analysis model (default `gpt-oss:120b`)
- `--analysis-limit 15` number of items sent to Ollama each cycle
- `--model-poll-seconds 30` refresh interval for live Ollama model list
- `--sources newsletters,arxiv,x`
- `--sort newest|trending`
- `--lookback-days 15`
- `--max-items 25`
- `--newsletters all` or comma-separated names
- `--quick-filter "open-source"`

## Slash Commands (while running)

Hotkeys:

- `/` enter command mode
- `Enter` submit command (or open selected story in normal mode)
- `Tab` next story
- `Shift+Tab` previous story
- `j`/`k` or `Down`/`Up` next/previous story
- `o` open selected story link
- `b` toggle brief focus mode (makes brief much larger)
- `[` / `]` resize right-hand panel narrower/wider
- `Esc` cancel command mode
- `m` toggle menu
- `q` quit immediately

Commands:

- `/help` show command list
- `/config` show active runtime config
- `/set <key> <value>` update settings live
- `/sources <newsletters,arxiv,x>` change sources
- `/newsletters <all|csv names>` change newsletter set
- `/refresh` trigger immediate refresh now
- `/open [index]` open selected story, or a 1-based index
- `/brief [focus|normal|toggle]` change brief focus mode
- `/layout right <2..6>` resize right panel via command
- `/export [md|json|csv] [path]` export current feed snapshot
- `/menu [on|off|toggle]` show/hide menu panel
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
ollama pull gpt-oss:120b
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
