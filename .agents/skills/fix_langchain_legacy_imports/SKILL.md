---
name: Resolve Langchain Legacy Chains Import Error
description: Fixes "No module named 'langchain.chains'" errors when using Python 3.14+ or Langchain > 0.3.x limits.
---

# Fix Langchain Legacy Chains Error

## Problem Context
The `langchain.chains` module was completely decoupled in larger Langchain releases. When importing `from langchain.chains import RetrievalQA` on an environment with `langchain>=1.0` or missing backward compatibility, the pipeline instantly crashes with a `ModuleNotFoundError`.

## The Fix
Use `langchain-classic` to restore missing functionality while maintaining cutting-edge agent pipelines in Langchain ecosystem.

### Implementation Pattern

1. **Install Classic Extension**:
```bash
pip install langchain-classic
```

2. **Update Codebase Imports**:
```diff
- from langchain.chains import RetrievalQA
+ from langchain_classic.chains import RetrievalQA
```

3. **Lock Requirements Profile**:
Ensure `requirements.txt` specifies `langchain-classic` and anchors down generic langchain bounds if relying on legacy loaders:
```txt
langchain>=0.3.0,<1.0
langchain-classic>=0.1.0
```
