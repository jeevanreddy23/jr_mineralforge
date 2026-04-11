---
name: Hardened Environment Loading
description: Fixes bugs where API keys or LLM provider connections drop because python environments don't auto-read .env files dynamically.
---

# Hardened Environment Loading

## Problem Context
When configuring a project through a `.env.example` file that users rename to `.env`, variables are not automatically exposed to the `os.environ` array without a loader. Scripts running outside of Docker (which has native env parsing capabilities) will fall back to hardcoded defaults or raise unhelpful timeout/authorization errors connecting to resources downstream.

## The Fix
Integrate `python-dotenv` explicitly at the head of application routing files (`app.py`, `config.py`, `settings.py`, `main.py`).

### Implementation Pattern

1. **Include within requirements**:
```bash
pip install python-dotenv
```

2. **Inject at exact runtime root**:
```python
import os
from dotenv import load_dotenv

# MUST explicitly load before any configuration constants are bound!
load_dotenv()
```

## Instructions
1. Check the project `requirements.txt` includes `python-dotenv`.
2. Inspect the topmost imports of the core scripts initializing configuration objects and inject the `load_dotenv` pattern unconditionally.
