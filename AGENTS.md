# Repository Guidelines

## Project Structure & Module Organization
- `src/`: C++ implementation files (`main.cpp`, `DataLoader.cpp`, `Optimizer.cpp`, `GeometryTools.cpp`, `FileUtils.cpp`).
- `include/`: Public headers matching source modules (for example, `Optimizer.h` for `Optimizer.cpp`).
- `data/`: Runtime input and output files (`conf.yaml`, source CSVs, generated `optimized_quaternions.csv`).
- `utils/`: Helper scripts, currently `plot_quaternions.py`, with generated images under `utils/plots/`.
- `build/`: Local build artifacts and executable output; treat as generated content.

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build --config Release`
- Run optimizer: `.\build\quaternion_optimizer.exe`
- Rebuild from clean state: `cmake --build build --target clean && cmake --build build`
- Plot results: `python .\utils\plot_quaternions.py`

Notes: this project expects Eigen3, NLopt, and yaml-cpp (see `CMakeLists.txt` paths/toolchain settings).

## Coding Style & Naming Conventions
- Language baseline: C++17.
- Indentation: 4 spaces; braces on the same line for functions/conditionals.
- Naming patterns:
  - Types/classes: `PascalCase` (e.g., `DataLoader`).
  - Methods/functions: `camelCase` (e.g., `loadRcsGridData`).
  - File names: `PascalCase.cpp/.h` matching class/module names.
  - Constants/local immutable values: `snake_case` with descriptive names.
- Keep header/source separation strict: declarations in `include/`, implementations in `src/`.

## Testing Guidelines
- No dedicated automated test suite is currently present.
- For changes, validate by:
  - Building successfully via CMake.
  - Running `.\build\quaternion_optimizer.exe` against files in `data/`.
  - Verifying output CSV shape/content and optional plot generation via `utils/plot_quaternions.py`.
- If adding tests, prefer CTest-compatible targets and place test sources under a new `tests/` directory.

## Commit & Pull Request Guidelines
- Git metadata is not included in this workspace snapshot, so historical commit conventions are not discoverable here.
- Use clear, imperative commit messages (for example: `fix: handle missing RCS rows`).
- Keep commits focused by module (`DataLoader`, `Optimizer`, etc.) and include data/config changes only when required.
- PRs should include: purpose, key changes, local validation steps, and sample output paths (for example, `data/optimized_quaternions.csv` or `utils/plots/quaternion_comparison.png`).
