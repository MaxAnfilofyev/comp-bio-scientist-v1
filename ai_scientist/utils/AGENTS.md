# Utilities Library

This directory (`ai_scientist/utils`) contains shared helper modules used across agents and tools.

## Core Utilities

- **`manifest.py`**: The backbone of the artifact memory system. Handles reading/writing the global `manifest_index.json` and its shards.
- **`pathing.py`**: Provides `resolve_output_path` and `resolve_input_path` to ensure agents read/write only within allowed directories (preventing path traversal).
- **`base_tool.py`**: (Located in `tools/` but fundamental) Defines the `BaseTool` class.

## Monitoring & Health
- **`health.py`**: Functions to check the integrity of the run (e.g., `check_manifest_health`).
- **`token_tracker.py`**: Tracks token usage across LLM calls for cost accounting.
- **`notes.py`**: Utilities for agents to save scratchpad notes.

## Domain-Specific Helpers
- **`per_compartment.py`**: Logic for analyzing compartmental simulation outputs (topology metrics, binary state classification).
- **`transport_index.py`**: Helper calculations for transport phenomena.
- **`repair_helpers.py`**: dedicated helpers for the `repair_sim_outputs` tool (e.g., safe file moving, directory normalization).
- **`experiment_utils.py`**: Common utilities for loading idea text, experiment summaries, and filtering them for specific steps (writeup, review, etc.).
