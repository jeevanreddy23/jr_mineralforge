---
name: Fix Gradio Chat Format
description: Ensures Gradio chat returns use dictionary format instead of tuples to prevent Data incompatible with messages format errors.
---

# Fix Gradio Chat Format

## Problem Context
When upgrading to Gradio 6.x or strictly enforcing message types in newer Gradio instances, returning a list of tuples `[(user_message, bot_message)]` from a chat function often triggers:
`Data incompatible with messages format. Each message should be a dictionary with 'role' and 'content' keys`

Furthermore, if the Langchain orchestrator outputs a native `AIMessage` or `ToolMessage` object, Gradio cannot parse the properties and will crash.

## The Fix
Override the custom application format so history outputs a pure JSON-friendly format containing: `{"role": "user", "content": "..."}` and `{"role": "assistant", "content": "..."}` arrays.

### Implementation Pattern

```python
# Before (Legacy tuples):
history.append((message, response))

# After (Gradio 5/6 Compatible):
if hasattr(response, "content"):
    response = response.content

history.append({"role": "user", "content": str(message)})
history.append({"role": "assistant", "content": str(response)})
```

## Instructions
1. Navigate to the frontend/UI logic (typically `app.py`).
2. Identify the `gr.Chatbot` callback execution loop.
3. Ensure the return history forces dictionary appending formatted appropriately.
