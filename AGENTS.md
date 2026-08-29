# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++ implementation (`main.cpp`, `DataLoader.cpp`, `Optimizer.cpp`, `GeometryTools.cpp`, `FileUtils.cpp`).
- `include/`: Public headers (`Optimizer.h`, `DataLoader.h`, `GeometryTools.h`, `FileUtils.h`).
- `data/`: Runtime I/O (`conf.yaml`, source CSVs, generated `optimized_quaternions.csv`).
- `utils/`: Helper scripts (`plot_quaternions.py`), output plots under `utils/plots/`.
- `build/`: Generated build artifacts.

## Algorithm Overview
**Sliding-window quaternion optimization using Gauss-Newton + Augmented Lagrangian Method (ALM).**

Per window:
1. **GN inner loop** (2-3 iterations): minimize scalar objective
   - Velocity: `||Log(q_{i-1}^{-1}·q_i)||²` with analytical `Jac⁻¹·adjoint` Jacobian
   - Acceleration: Lie-algebra subtraction `||r_i − r_{i-1}||²` (R³ approximation)
   - Jerk: `||r_i − 2r_{i-1} + r_{i-2}||²`
   - Fidelity: `||Log(q_i⁻¹·q_target)||²`
   - LDLT solver on 3N×3N banded Hessian
   - Tangent-space incremental update `q = exp(dx)·q`

2. **ALM outer loop**: SDF boundary constraint (inequality `g ≤ 0`)
   - Constraint: warp RCS boundary by homography `H = K·R·K⁻¹`, compute signed distance of 4 crop corners to warped polygon
   - Gradient: analytical chain rule `∂SDF/∂pixel → ∂pixel/∂H → ∂H/∂R → ∂R/∂δ`
   - Penalty: `b −= (λ+ρ·g)·Jᵀ`, `H += ρ·JᵀJ`
   - Multiplier update: `λ ← max(0, λ + ρ·g)`, `ρ ← min(ρ·γ, ρ_max)`

3. **Coordinate system**: center-origin pixel coordinates
   - RCS grid: `x = x_rsc × (W−1)/2`, `y = y_rsc × (H−1)/2`
   - Crop corners: `±(W−1)/2·(1−cropH)`, `±(H−1)/2·(1−cropV)`
   - K matrix: `diag(f, f, 1)` with principal point at (0,0)

## Dependencies
- Eigen3 (linear algebra)
- yaml-cpp (config parsing)
- NLopt **removed** — replaced by hand-written GN+ALM solver

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build --config Release`
- Run: `./build/quaternion_optimizer`
- Clean rebuild: `cmake --build build --target clean && cmake --build build`
- Plot: `python utils/plot_quaternions.py`

## Key Files & Functions
| File | Key contents |
|---|---|
| `Optimizer.cpp` | `solveWindowGN()` — main GN+ALM solver; `evaluateSDF()` — constraint eval + gradient; `rightJacobianInverseSO3`, `expMap`, `logMap` — SO(3) utilities |
| `DataLoader.cpp` | `loadRcsGridData()` — RCS grid scaled to center-origin pixel coordinates |
| `GeometryTools.cpp` | `signedDistanceFromPointToPolygon()` — SDF with optional closest-edge output |
| `main.cpp` | Sliding-window loop |

## Tuning Parameters

All solver knobs live in `SolverConfig` (include/Optimizer.h).

| Parameter | Default | Effect |
|---|---|---|
| `λ1` | 1.0 | Velocity smoothness |
| `λ2` | 2.0 | Acceleration smoothness |
| `λ3` | 8.0 | Jerk smoothness |
| `μ` | 0.02 | Fidelity (lower = more smoothing, closer to boundary) |
| `ρ_init` | 20.0 | ALM initial penalty |
| `ρ_max` | 500.0 | ALM max penalty |
| `γ` | 2.0 | ALM penalty growth rate |
| `max_outer` / `max_inner` | 5 / 8 | ALM / GN iteration limits |

## Coding Style & Naming Conventions
- Language baseline: C++17.
- Indentation: 4 spaces; braces on the same line for functions/conditionals.
- Types/classes: `PascalCase`. Methods/functions: `camelCase`.
- File names: `PascalCase.cpp/.h` matching class/module names.
- Header/source separation: declarations in `include/`, implementations in `src/`.

## Testing Guidelines
- No dedicated automated test suite.
- Validate by: `cmake --build build --config Release && ./build/quaternion_optimizer`.
- Verify output CSV and plot via `utils/plot_quaternions.py`.
- If adding tests, prefer CTest-compatible targets under `tests/`.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages (e.g. `fix: correct RCS grid coordinate scaling`).
- Keep commits focused by module.
- PRs should include: purpose, key changes, validation steps, sample output paths.
