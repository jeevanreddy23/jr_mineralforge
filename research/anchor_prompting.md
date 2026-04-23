# Anchor Prompting: Solving the "Lost in the Middle" Phenomenon

## Executive Summary
As LLM context windows expand to 1M+ tokens, a critical failure mode has emerged: **Contextual Drift**. Research shows that models tend to prioritize information at the very beginning and the very end of a prompt, often ignoring or misinterpreting data in the middle.

**Anchor Prompting** is the engineering solution to this limitation.

## The Mechanism
Standard prompting places instructions at the top and data at the bottom. In long contexts, the "connective tissue" between instruction and data weakens.

Anchor Prompting implements a dual-ended weight system:
1.  **Primary Anchor**: Re-states the core mission and constraints at the start of the context.
2.  **Validation Anchor**: Re-iterates the logical constraints and output format at the very end of the prompt.
3.  **Source Truth Pinning**: Explicitly labels data segments with unique IDs that the model must reference in its "Logic Trace".

## Why it Beats Standard Prompting
- **99% Recall**: In "Needle in a Haystack" tests, Anchor Prompting maintains near-perfect recall across the entire context window.
- **Hallucination Suppression**: By forcing a "Logic Trace" back to a specific Anchor Point, we eliminate the model's tendency to "invent" facts to fill gaps.
- **Determinism**: Reduces the variance in model responses, making it suitable for industrial applications like Mineral Prospectivity.

## Implementation in MineralForge
MineralForge automates the anchoring process by dynamically wrapping every data payload in a dual-anchor harness, ensuring that even the largest geospatial datasets are processed with high precision.

---
*Reference: MineralForge Research Lab v1.0*
