# Prior-Art Patent Search Agent

An agentic, multi-turn, tool-using system for automated prior-art search over patent corpora.

## Recommended Training Order

1. **Tinker (recommended, current default)**
2. **ART / OpenPipe ART (legacy path kept in repo)**

## Tinker (Recommended)

The current training entrypoint is:

- [TINKER_grpo_train.py](/Users/mounselam/Developer/Patent-Search/prior_art_search/TINKER_grpo_train.py)

It uses:

- `tinker_cookbook.rl.train.main(...)` (official common RL loop)
- group rollouts + centered group advantages
- selectable loss: `ppo` (default) or `importance_sampling`
- local Chroma tools (`search_patents`, `lookup_patent`) through `PatentTools`
- persistent run artifacts under `training_logs/`

### One-time setup

Build the local retrieval dataset from Hugging Face HUPD:

```bash
uv run python prepare.py --limit 500
```

This downloads `HUPD/hupd`, writes `Evals/patent_search_queries.csv`, and builds the
local Chroma collection in `.chroma_db/`. Retrieval indexes title, abstract, claims,
classifications, status, and key dates.

Then ensure `.env` contains:

```env
TINKER_API_KEY=...
WANDB_API_KEY=...
```

### Default run (config-driven)

Default config file:

- [TINKER_grpo_train.config.json](/Users/mounselam/Developer/Patent-Search/prior_art_search/TINKER_grpo_train.config.json)

Run with defaults:

```bash
uv run prior_art_search/TINKER_grpo_train.py
```

CLI flags override config values:

```bash
uv run prior_art_search/TINKER_grpo_train.py --steps 2 --group-size 3
```

### Logging layout

Each run persists to:

- `training_logs/runs/<run_id>/` (raw cookbook run files such as `logs.log`, `config.json`, `code.diff`)
- `training_logs/metrics/<run_id>/` (`metrics.jsonl`, `checkpoints.jsonl`)
- `training_logs/traces/<run_id>/` (`tool_trace.jsonl`; optional `trace_events*.jsonl`; HTML reports only if `--num-groups-to-log > 0`)
- `training_logs/latest_run.json` (pointer to latest run + final checkpoint)

## ART / OpenPipe ART (Legacy)

ART path is still present and unchanged:

- [training_loop_qwen.py](/Users/mounselam/Developer/Patent-Search/prior_art_search/training_loop_qwen.py)
- [rollout.py](/Users/mounselam/Developer/Patent-Search/prior_art_search/rollout.py)
- [prior_art_tools.py](/Users/mounselam/Developer/Patent-Search/prior_art_search/prior_art_tools.py)

Use this path if you explicitly want the ART workflow. The default and maintained path is now Tinker.
