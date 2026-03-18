---
name: DatabaseManager import pattern in api_server.py
description: api_server.py uses inline local imports for DatabaseManager — never relies on a top-level import
type: feedback
---

All uses of `DatabaseManager` in `api_server.py` must include a local `from credentialing_agent import DatabaseManager` immediately before the first usage inside the function's `try` block. There is no top-level import of `DatabaseManager` in `api_server.py`.

**Why:** The class lives in `credentialing_agent.py` and is not re-exported from any shared module. The established pattern throughout the file is a per-function inline import. Any new endpoint that uses `DatabaseManager` must follow this pattern.

**How to apply:** Whenever adding or reviewing a function in `api_server.py` that calls `DatabaseManager()`, ensure the line `from credentialing_agent import DatabaseManager` appears inside the same `try` block, directly before the first `DatabaseManager()` call.
