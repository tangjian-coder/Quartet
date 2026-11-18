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

## Attitude Convention — ONE convention end to end
- Every pose in the pipeline (`quat_at`, `quat.csv`, `opt.csv`, ABI to C++/legacy) satisfies
  **`p_current = R(q) · p_world`** (world -> camera). `integrate_gyro` integrates
  `dq/dt = -[0, w_body/2] x q`, i.e. `R = exp(-[w]·t)`, which is exactly this convention.
  Cross-checked against the dataset's own reference implementation (Kontiki
  `trajectory.h`: `x_world = orientation * x_body`, gyro `= orientation.conjugate()*omega_world`)
  and by pixel-level tests.
- Nothing conjugates anymore: `gopro_imu_to_input.py` writes `q_mid` as-is (no `conj`), and
  `gopro_imu_render.py` reads `opt.csv` as-is. `rcs.csv` and `quat.csv` hold the same w2c poses.
- De-rolling-shutter (row frame -> view frame) homography, the **forward** map used by
  `warp_grid` (rcs.csv) and the C++ boundary polygon:
  `H = K · R(q_view) · R(q_row)ᵀ · K⁻¹`.
- The renderer (`common.warp_pixels`) is the **inverse gather**: it rotates the output
  ray by `R(q_row) · R(q_view)ᵀ`. The two matrices are exact inverses (verified: warp_grid
  round-trip 0.00 px); using the forward form in the gather doubles the rolling-shutter shear
  (~38 px) and was the original "stab worse than raw" defect.
- Both renderers require an explicit mode: `--fix0` (lock the view to the first rendered frame,
  a verification that a correct chain holds still) or `--opt-quat` (final stabilized output).
  fix0 is never the default, and the single-pass `output_row` variant is gone: every mode
  iterates the source row so the de-RS solve is self-consistent.
- `utils/render_videos.py` (legacy/product renderer) is already written for this convention:
  `Ht = K·R(W_opt)·R(W_mid)ᵀ·K⁻¹` warps the RCS grid forward and `griddata` inverts it.
- ADVIO: ARKit and ground-truth files are camera->world, so `advio_common.make_quat_at`
  conjugates them on import; the gyro branch left-multiplies `R_cam_imu` (w2c composition).
  There is no switch for the other reading: treating those files as w2c measured 4-8x worse
  fix0 residual (arkit 0.42 -> 3.25 px, gt 0.88 -> 4.45 px).
- Historical note: the pipeline used to integrate the c2w attitude and conjugate only when
  writing `quat.csv`; `warp_pixels` compensated with the mirrored rotation, so the video looked
  right while `rcs.csv` (fed to the C++ SDF) was mirrored. Do not reintroduce a per-file
  convention switch.

## Dependencies
- Eigen3 (linear algebra)
- yaml-cpp (config parsing)
- NLopt **removed** — replaced by hand-written GN+ALM solver

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build --config Release`
- Run: `./build/quartet`
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
- Validate by: `cmake --build build --config Release && ./build/quartet`.
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
