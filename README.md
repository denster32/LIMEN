# LIMEN

*Latin: threshold.*

MCP persistence server for Claude. Cross-conversation memory that tracks what matters: projects, decisions, pending items, mistakes, and context.

No consciousness metrics. No phase transitions. Just memory.

## Vision

Every Claude instance — Desktop, Code, claude.ai, mobile — should have persistent local memory. LIMEN is the first step. Currently supports Claude Desktop and Claude Code via MCP stdio transport. Remote transport and broader client support are on the roadmap.

## Current Support

| Client | Status | Transport |
|--------|--------|-----------|
| Claude Desktop | ✅ Works | stdio (local MCP) |
| Claude Code / Cursor | ✅ Works | stdio (local MCP) |
| claude.ai | ❌ Not yet | No user MCP support — manual context paste as workaround |
| Claude mobile | ❌ Not yet | No user MCP support |

## Roadmap

- [ ] **HTTP/SSE transport** — Run LIMEN as a remote server on local network or VPS, enabling any MCP-capable client to connect
- [ ] **Context export** — Generate a paste-ready context block for claude.ai conversations
- [ ] **Multi-instance sync** — Multiple Claude clients writing to the same state file with conflict resolution
- [ ] **Auto-log** — Reduce friction by inferring conversation records from context rather than requiring explicit `log` calls

## Install

```
git clone https://github.com/denster32/LIMEN.git
cd LIMEN
pip install mcp
```

## Run

```
python -m scripts.run_server --state ./state/limen.json
```

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "limen": {
      "command": "python3",
      "args": ["-m", "scripts.run_server", "--state", "/path/to/state/limen.json"],
      "cwd": "/path/to/LIMEN"
    }
  }
}
```

Restart Claude Desktop after editing.

## Claude Code / Cursor Setup

Add to your MCP config or run the server and point your client at it.

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

## License

LGPL-2.1
