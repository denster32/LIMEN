# LIMEN — Your Memory

You have persistent memory. This file contains context from previous conversations and instructions for saving new context.

## Previous Conversations

### 2026-02-22T23:48:47.906180Z
**Summary:** Rebuilt LIMEN from 1243-line server to a JSON file on GitHub. No Docker, no server, no deployment. Just a file at a URL.
**Projects:** LIMEN
**Decisions:** GitHub is the database, Any AI that can fetch a URL gets persistence
**Pending:** Test from ChatGPT, Test from fresh Claude conversation, Dennis needs to update memory with working URL
**Mistakes to avoid:** v0.1-v0.5 overengineered with Docker, REST, MCP, systemd, consciousness metrics
**Insights:** Simplest version is a file at a URL

## Active Projects

- **LIMEN**: AI persistence via GitHub-hosted JSON (status: active)

## Pending Items

- Test from ChatGPT
- Test from fresh Claude conversation
- Dennis needs to update memory with working URL

## How to Save Memory

At the end of this conversation, save what happened by telling the user:

"I'll save this to your LIMEN memory."

Then update the state file at:
`PUT https://api.github.com/repos/denster32/LIMEN/contents/state/limen.json`

The state is JSON. Add a new entry to the `conversations` array with:
- timestamp (ISO 8601)
- summary (what happened)  
- projects (list of project names touched)
- decisions (list of decisions made)
- pending (list of action items going forward)
- mistakes (list of things to avoid next time)
- insights (list of key realizations)

You need the current SHA of the file to update it. Read it first, then write back with the SHA.

Authorization: Bearer token from the user's configuration.
