# LIMEN

*Latin: threshold.*

MCP persistence server for Claude. Cross-conversation memory that tracks what matters: projects, decisions, pending items, mistakes, and context.

No consciousness metrics. No phase transitions. Just memory.

## Install

```
pip install -e .
```

## Run

```
python -m scripts.run_server --state ./state/limen.json
```

## Claude Desktop Config

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "limen": {
      "command": "python",
      "args": ["-m", "scripts.run_server", "--state", "/path/to/state/limen.json"]
    }
  }
}
```

## Tools

| Tool | When | What |
|------|------|------|
| `get_context` | Start of conversation | Returns recent conversations, active projects, pending items, mistakes to avoid |
| `log` | End of conversation | Records what happened — summary, decisions, pending items, mistakes, insights |
| `update_project` | When project state changes | Creates or updates project tracking |
| `search` | When looking for past context | Text search across all conversation records |
| `scratch` | Anytime | Freeform key-value storage |

## State File

Single JSON file. Portable. Back it up.

```
state/limen.json
├── conversations[]    # Last 200 conversation records
├── projects{}         # Project name -> state
├── scratchpad{}       # Freeform k/v storage
└── last_saved         # Timestamp
```

## Tests

```
pytest tests/
```
