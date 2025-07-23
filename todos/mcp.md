# MCP Server
An mcp server for turboprop that is designed to work exceptionally well with coding agents like:

- Claude Code
- GitHub Copilot
- Cursor
- Windsurf

The user needs to be able to install turboprop via the .mcp.json config (or equivalent). It should be very 
easy to install. It should look something like this:

```
tp mcp --repo . 
```

This will do the following:
- start an mcp server that exposes a `search` tool
- automatically indexes the --repo (and uses any flags the user might have specified)
- watches for changes in any of the files and updates the index accordingly
- uses the config file if present in the repo
- DO NOT add extra requirements. Keep this simple.
- DO NOT create more tools than is necessary.
