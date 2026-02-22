# LIMEN

*Latin: threshold.*

Persistence layer for AI. Cross-conversation memory for any AI assistant that can fetch a URL.

One server. One state file. Every AI you talk to shares the same memory.

## What It Does

AI assistants forget everything between conversations. LIMEN gives them memory. Your AI checks LIMEN at the start of each conversation to know what happened before, and logs what happened at the end so the next conversation picks up where you left off.

Works with any AI that can make HTTP requests — Claude, ChatGPT, Gemini, local models, anything.

## How It Works

```
┌──────────────┐
│   Claude.ai  ├──┐
├──────────────┤  │
│   ChatGPT    ├──┤  GET/POST
├──────────────┤  ├──────────►  LIMEN server  ──► state.json
│   Gemini     ├──┤             (REST API)
├──────────────┤  │
│  Local LLM   ├──┘
└──────────────┘

Also:
┌──────────────┐     stdio
│Claude Desktop├───────────►  LIMEN server
├──────────────┤              (MCP protocol)
│ Claude Code  ├───────────►
└──────────────┘
```

Every operation works via simple GET requests with query parameters. No special SDK. No client library. If your AI can fetch a URL, it can use LIMEN.

## Quick Start

### Option 1: Self-hosted (your machine)

```bash
git clone https://github.com/denster32/LIMEN.git
cd LIMEN
pip install mcp starlette uvicorn
python -m scripts.run_server --mode rest --port 8452
```

### Option 2: One-click cloud deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/LIMEN)

```bash
# Or manually on any VPS:
git clone https://github.com/denster32/LIMEN.git && cd LIMEN && bash scripts/deploy.sh
```

### Option 3: Local + tunnel (free public URL)

```bash
# Terminal 1: Run LIMEN
python -m scripts.run_server --mode rest --port 8452

# Terminal 2: Expose publicly (free)
# Pick one:
cloudflared tunnel --url http://localhost:8452     # Cloudflare (persistent)
ngrok http 8452                                     # ngrok
tailscale funnel 8452                               # Tailscale
```

## API

Every endpoint works via GET. All write operations accept data as query parameters.

Auth: `?token=YOUR_TOKEN` on every request (except /health).

| Endpoint | What | Example |
|----------|------|---------|
| `GET /health` | Server status | `/health` |
| `GET /context` | Get memory | `/context?token=T&n=5` |
| `GET /log` | Record conversation | `/log?token=T&summary=Built+LIMEN&projects=LIMEN&pending=deploy` |
| `GET /project` | Update project | `/project?token=T&name=LIMEN&status=active` |
| `GET /search` | Search history | `/search?token=T&q=TASNI` |
| `GET /scratch` | Read/write notes | `/scratch?token=T&key=ip&value=192.168.5.49` |

POST with JSON body also works on all endpoints for clients that support it.

### List parameters

For fields that accept lists (projects, decisions, pending, mistakes, insights, tags, notes, blockers), use comma-separated values:

```
/log?token=T&summary=Session+summary&projects=LIMEN,TASNI&pending=deploy,test&tags=mcp,rest
```

## Connecting Your AI

### Claude (claude.ai)

Tell Claude in your conversation or add to your custom instructions:

> At the start of every conversation, fetch https://YOUR_URL/context?token=YOUR_TOKEN to get memory of past conversations. At the end, fetch https://YOUR_URL/log?token=YOUR_TOKEN&summary=SUMMARY to record what happened.

### Claude Desktop / Claude Code (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "limen": {
      "command": "python3",
      "args": ["-m", "scripts.run_server", "--mode", "mcp", "--state", "/path/to/.limen/limen.json"],
      "cwd": "/path/to/LIMEN"
    }
  }
}
```

### ChatGPT (Custom GPT or browsing)

In your Custom GPT instructions or conversation:

> Before responding, browse to https://YOUR_URL/context?token=YOUR_TOKEN to check for context from previous conversations. At the end of our conversation, browse to https://YOUR_URL/log?token=YOUR_TOKEN&summary=SUMMARY to save what we discussed.

### Gemini / Other AI with web access

Same pattern — any AI that can browse URLs can read and write LIMEN. Just tell it the URLs.

### Local models with function calling

Add LIMEN endpoints as tools in your function calling schema:

```json
{
  "name": "limen_context",
  "description": "Get memory from previous conversations",
  "parameters": {},
  "endpoint": "GET https://YOUR_URL/context?token=YOUR_TOKEN"
}
```

## State File

Single JSON file. Portable. Back it up.

```
~/.limen/limen.json
├── conversations[]    # Last 200 conversation records
├── projects{}         # Project name -> state
├── scratchpad{}       # Freeform k/v storage
└── last_saved         # Timestamp
```

## Tests

```
pytest tests/ -v    # 30 tests
```

## Security

- Auth token required on all endpoints (except /health)
- Token passed via `?token=` query param or `Authorization: Bearer` header
- Auto-generated on first run, saved to `~/.limen/.token`
- Set `LIMEN_TOKEN` env var or `--token` flag for custom tokens
- `--no-auth` flag available for local-only use (not recommended for public)

## Roadmap

- [x] MCP server (Claude Desktop / Code)
- [x] REST API with GET-writable endpoints
- [x] Token auth
- [ ] Hosted version (no self-hosting needed)
- [ ] Multi-user support (separate state per API key)
- [ ] Auto-log (infer records from conversation without explicit logging)
- [ ] Webhook notifications (get alerted when AI logs something)
- [ ] Web dashboard for viewing/editing state

## License

LGPL-2.1
