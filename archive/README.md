# Archive

This folder contains historical development artifacts preserved for reference. All functionality has been consolidated into the current production systems.

## Contents

| Folder | Original Location | Description | Superseded By |
|--------|-------------------|-------------|---------------|
| `legacy_cost_analysis/` | `legacy/` | Early FLOPs/cost scripts | `training_planner/` |
| `development_phases/` | `system_implementation/` | nanoGPT → ZeRO-1 → Triton → FSDP | `enhanced_training_system/` |
| `intermediate_sharing/` | `system_branch/` | Cross-repo sharing artifacts | N/A |
| `scaling_law_standalone/` | `scaling_law/` | Standalone scaling law tool | `training_planner/` |
| `mfu_compute_standalone/` | `MFU_compute/` | Standalone MFU calculator | `training_planner/` |

## Details

### `legacy_cost_analysis/`
Early FLOPs and cost analysis scripts - the first iteration of compute planning tools.

### `development_phases/`
Incremental development from nanoGPT base:
- `nanoGPT/` - Original Karpathy fork
- `phase1_zero1/` - Added ZeRO-1 optimizer sharding
- `phase2_triton/` - Added Triton custom kernels  
- `phase3_fsdp/` - Added FSDP full sharding

### `intermediate_sharing/`
Intermediate work prepared for sharing to external repos.

### `scaling_law_standalone/`
Standalone scaling law analysis tool. Functionality now integrated into `training_planner/`.

### `mfu_compute_standalone/`
Standalone MFU (Model FLOPs Utilization) calculator. Functionality now integrated into `training_planner/`.

---

## Current Production Systems

| System | Purpose |
|--------|---------|
| `enhanced_training_system/` | Core LLM training framework |
| `training_planner/` | FLOPs, parameters, scaling laws, MFU |
| `post_training/` | SFT & DPO alignment |
| `evaluation_system/` | Benchmark evaluation |
| `serving_system/` | Production deployment |

**Note:** These archived folders are kept for historical reference only. Use the current production systems for active development.

