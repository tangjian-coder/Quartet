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
   - K matrix: real K in top-left origin pixels (`calFocalX/Y`, `calPrincipalX/Y`);
     the loader must center it: `diag(fx, fy, 1)` with principal point
     `(calPrincipalX − (W−1)/2, calPrincipalY − (H−1)/2)`

## Attitude Convention (c2w / w2c) — verify before deriving pose/warp formulas
- Pose quaternions from IMU integration (`quat_at`, `adapters/gopro_imu/gopro_common.py`)
  are rotations whose matrix maps **world-frame coordinates into the current camera
  frame**: `p_current = R(q) · p_world`. Comments call this convention **"c2w"** — do NOT
  infer direction from textbook c2w/w2c naming; this behavioral definition is authoritative.
- Frame poses written to conf/rcs (`gopro_imu_to_input.py`) are `conj(q)` (dubbed "w2c"):
  they map current-frame coordinates back to the world frame.
- Same convention as product HIS_V2 (its `integrateGyro` is a first-order approximation
  of the same family). De-rolling-shutter (row frame -> mid frame) homography:
  `H = K · R(q_mid) · R(q_row)ᵀ · K⁻¹`.
- Inverse / render chain (`gopro_imu_validate.warp_virtual`, `gopro_common.warp_pixels`)
  applies the SAME rotation `R(q_to) · R(q_row)ᵀ`, but to the source-coordinate guess rather
  than to the output ray: it drives rows back to the view frame, so transposing it here is
  WRONG. Verified by lock-frame-0 far-field drift (reversed direction 15 px vs this one
  0.9 px), see `gopro_imu_align_check.py` and the row loop in `gopro_imu_validate.py`.
- `warp_grid` (forward, rcs.csv/constraint chain) and the render chain share ldc, row-time
  and pose interpolation, but the rotation plays a different role in each (forward maps
  row-frame points into the view frame; inverse feeds the source-coordinate guess).

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

Solver knobs are split by role:
- **Algorithm/policy knobs** (changing them alters the solution quality/trade-off): `SolverConfig` in `include/Optimizer.h`
- **Numerical/implementation constants** (solver-internal, affect only robustness/efficiency): `constexpr` in `src/Optimizer.cpp`

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

## Baselines
- Real-K stats are regenerated on demand (5 GoPro sequences, `--verbose`); the local
  snapshot stays outside the repo at `~/data/gopro_imu/output/baseline_realK.txt`.
- Legacy path (virtual-frame conf + `data/DbgFile_*`): byte-identical to the HEAD build
  (1095 frames, max abs diff 0.0), so the old case baselines remain valid.
- Crop-window check: `python3 tests/sdf_scale_check.py <case>`; constraint activity scan
  showed `cropFactor` must drop to ~0.05-0.10 before the SDF reaches zero on playground3.

## Commit & Pull Request Guidelines
- Use clear, imperative commit messages (e.g. `fix: correct RCS grid coordinate scaling`).
- Keep commits focused by module.
- PRs should include: purpose, key changes, validation steps, sample output paths.
