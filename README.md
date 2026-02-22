# LIMEN

*Latin: threshold.*

Persistence layer for Claude. Cross-conversation memory that tracks what matters: projects, decisions, pending items, mistakes, and context.

Every Claude instance — Desktop, Code, claude.ai — gets the same memory.

## How It Works

LIMEN runs as a server with two interfaces to the same state:

- **MCP (stdio)** — Claude Desktop and Claude Code connect natively
- **REST (HTTP)** — Claude.ai uses `web_fetch` to read/write state

One state file. Multiple Claude instances. Shared memory.

```
┌──────────────┐     stdio      ┌─────────┐
│Claude Desktop├───────────────►│         │
└──────────────┘                │         │
┌──────────────┐     stdio      │  LIMEN  ├──► state.json
│ Claude Code  ├───────────────►│         │
└──────────────┘                │         │
┌──────────────┐   HTTP :8452   │         │
│  claude.ai   ├───────────────►│         │
└──────────────┘  (web_fetch)   └─────────┘
```

## Install

```
git clone https://github.com/denster32/LIMEN.git
cd LIMEN
pip install mcp starlette uvicorn
```

## Run

```bash
# MCP only (Claude Desktop / Code)
python -m scripts.run_server --mode mcp --state ./state/limen.json

# REST only (claude.ai via web_fetch)
python -m scripts.run_server --mode rest --state ./state/limen.json --port 8452

# Both simultaneously (recommended)
python -m scripts.run_server --mode both --state ./state/limen.json --port 8452
```

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "limen": {
      "command": "python3",
      "args": ["-m", "scripts.run_server", "--mode", "mcp", "--state", "/path/to/state/limen.json"],
      "cwd": "/path/to/LIMEN"
    }
  }
}
```

## claude.ai Usage

With the REST server running, Claude can use `web_fetch` to read/write state:

```
GET  http://localhost:8452/context        — Get conversation context
POST http://localhost:8452/log            — Log a conversation
POST http://localhost:8452/project        — Update a project
GET  http://localhost:8452/search?q=term  — Search past conversations
GET  http://localhost:8452/scratch/key    — Read scratchpad
POST http://localhost:8452/scratch/key    — Write scratchpad
GET  http://localhost:8452/health         — Health check
```

For remote access (claude.ai can't hit localhost), run on a server with a public IP or use a tunnel (ngrok, tailscale, etc).

## Tools (MCP)

| Tool | When | What |
|------|------|------|
| `get_context` | Start of conversation | Recent conversations, active projects, pending items, mistakes |
| `log` | End of conversation | Record what happened — summary, decisions, pending, mistakes, insights |
| `update_project` | Project state changes | Create or update project tracking |
| `search` | Looking for past context | Text search across conversation records |
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
pytest tests/ -v
```

## Roadmap

- [ ] Auth token for REST API (currently open)
- [ ] Multi-instance write locking
- [ ] Auto-log (infer records from conversation context)
- [ ] Hosted deployment guide (VPS, tailscale, etc)
- [ ] Context export (paste-ready block for clients without MCP or HTTP)

## License

LGPL-2.1
