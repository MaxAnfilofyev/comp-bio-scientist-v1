# Orchestration Layer

This directory (`ai_scientist/orchestrator`) contains the logic for defining agents, managing their tools, and handling the artifacts they produce. It acts as the "operating system" for the AI Scientist.

## Key Modules

### `agents.py`
Defines the **Agent Roster**. This is where specific roles (Archivist, Modeler, etc.) are constructed.
- **`build_team(model, ...)`**: The main factory function that returns a dictionary of instantiated agents.
- **Roles**:
  - `archivist`: Specialist in literature search and organization.
  - `modeler`: Runs simulations and biological models.
  - `analyst`: Performs statistical analysis and plotting.
  - `interpreter`: Synthesizes results into scientific narratives.
  - `reviewer`: Critiques drafts and plans.
  - `publisher`: Handles "release engineering" (artifacts, manifests).
  - `pi` (Principal Investigator): The meta-agent that plans the workflow and delegates to others.

### `tool_wrappers.py`
The **Interface Layer**. This file wraps raw tools (from `ai_scientist/tools`) into LLM-friendly function schemas.
- **Responsibility**: It adapts complex python objects/returns into simple types (str, int, list) that an LLM can understand and output.
- **Safety**: It limits file I/O to allowed scopes and sanitizes inputs.
- **Metadata**: Many wrappers inject metadata (e.g., `reserve_and_register_artifact`) to track provenance.

### `artifacts.py` & `manifest_service.py`
The **Memory System**.
- **`ARTIFACT_TYPE_REGISTRY`**: Defines all valid output types (plots, JSONs, code bundles), their file patterns, and descriptions.
- **`reserve_and_register_artifact`**: The standard API for agents to "save" work. It handles versioning, quoting, and updating the global manifest.
- **`ManifestService`**: Maintains the global list of all files generated during a run, ensuring a complete and navigable history.

### `context_specs.py`
The **Access Control List**.
- Defines what artifacts each agent can *read* and *write*.
- Prevents "splash damage" (e.g., the Modeler overwriting the Literature Review).
- Controls context window usage by summarizing or truncating older artifacts.

### `pi_orchestration.py` & `pi_planning_helpers.py`
The **Executive Logic**.
- **Cumulative Planning**: Logic for the PI to maintain a living "Implementation Plan" (merged JSON/Markdown).
- **Tool Forcing**: Ensures the PI makes persistent updates (writes to the plan or inbox) rather than just "thinking" in circles.
