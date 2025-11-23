Transport run layout and naming

- Root: experiment_results/simulations/transport_runs
- Baseline folders: transport_runs/<baseline>/
- Transport folders: transport_runs/<baseline>/transport_<transport>/
- Seed folders: transport_runs/<baseline>/transport_<transport>/seed_<seed>/
- Files in each seed folder:
  - <baseline>_sim_failure_matrix.npy
  - <baseline>_sim_time_vector.npy
  - nodes_order_<baseline>_sim.txt
  - <baseline>_sim.json
  - <baseline>_sim.status.json

Completion rule
- A run is considered complete only when the arrays (failure_matrix, time_vector, nodes_order) and sim.json + sim.status.json exist.
- Prefer exporting arrays during the sim run; otherwise run sim_postprocess immediately on the sim.json so arrays are present before marking complete.

Canonical manifest
- Manifest path: experiment_results/simulations/transport_runs/manifest.json
- Each entry keyed by (baseline, transport, seed) with fields: status (complete|partial|error), paths, updated_at, notes, actor.
- Use the manifest as the source of truth for skip/verify; if missing or stale, run scan_transport_manifest to rebuild from disk.
