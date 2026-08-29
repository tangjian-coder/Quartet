#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "DataLoader.h"
#include "FileUtils.h"
#include "ScopedTimer.h"
#include "Optimizer.h"

namespace {

struct SlidingWindowConfig {
    size_t anchor_frames = 3;    // frames shared with previous window (fixed prefix)
    size_t output_frames = 15;   // frames output per window
    size_t future_frames = 30;   // future frames for look-ahead smoothing
};

void printUsage(const char* exe_name) {
    std::cout << "Usage: " << exe_name
              << " [--anchor-frames N] [--output-frames N] [--future-frames N] [--verbose]"
              << std::endl;
    std::cout << "  --quat <path>    raw quaternion csv (default ../data/DbgFile_Frame_OriginalCamOrient.csv)" << std::endl;
    std::cout << "  --yaml <path>    config yaml      (default ../data/conf.yaml)" << std::endl;
    std::cout << "  --rcs <path>     RCS grid csv     (default ../data/DbgFile_WarpGrids_RSCFRMMid.csv)" << std::endl;
    std::cout << "  --output <path>  optimized quat csv (default ../data/optimized_quaternions.csv)" << std::endl;
}

bool parseSizeT(const std::string& value, size_t& out) {
    try {
        size_t parsed = static_cast<size_t>(std::stoull(value));
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool readOptionValue(int argc, char** argv, int& index, const std::string& option, size_t& target) {
    if (index + 1 >= argc) {
        std::cerr << "Missing value for " << option << std::endl;
        return false;
    }

    size_t parsed = 0;
    if (!parseSizeT(argv[index + 1], parsed)) {
        std::cerr << "Invalid numeric value for " << option << ": " << argv[index + 1] << std::endl;
        return false;
    }

    target = parsed;
    ++index;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    // Flush output immediately so progress logs appear live.
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    const std::string quat_path = "../data/DbgFile_Frame_OriginalCamOrient.csv";
    const std::string yaml_path = "../data/conf.yaml";
    const std::string rcs_path = "../data/DbgFile_WarpGrids_RSCFRMMid.csv";
    const std::string output_path = "../data/optimized_quaternions.csv";
    std::string quat_path_opt = quat_path, yaml_path_opt = yaml_path;
    std::string rcs_path_opt = rcs_path, output_path_opt = output_path;

    SlidingWindowConfig window_cfg;
    bool verbose = false;
    // Parse command-line options.
    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        std::string arg = argv[arg_idx];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        // Path overrides (defaults keep ../data/... for backward compat).
        if (arg == "--quat") {
            if (arg_idx + 1 >= argc) { std::cerr << "Missing value for --quat" << std::endl; return 1; }
            quat_path_opt = argv[++arg_idx];
            continue;
        }
        if (arg == "--yaml") {
            if (arg_idx + 1 >= argc) { std::cerr << "Missing value for --yaml" << std::endl; return 1; }
            yaml_path_opt = argv[++arg_idx];
            continue;
        }
        if (arg == "--rcs") {
            if (arg_idx + 1 >= argc) { std::cerr << "Missing value for --rcs" << std::endl; return 1; }
            rcs_path_opt = argv[++arg_idx];
            continue;
        }
        if (arg == "--output") {
            if (arg_idx + 1 >= argc) { std::cerr << "Missing value for --output" << std::endl; return 1; }
            output_path_opt = argv[++arg_idx];
            continue;
        }
        if (arg == "--future-frames") {
            if (!readOptionValue(argc, argv, arg_idx, arg, window_cfg.future_frames)) {
                return 1;
            }
            continue;
        }
        if (arg == "--anchor-frames") {
            if (!readOptionValue(argc, argv, arg_idx, arg, window_cfg.anchor_frames)) {
                return 1;
            }
            continue;
        }
        if (arg == "--output-frames") {
            if (!readOptionValue(argc, argv, arg_idx, arg, window_cfg.output_frames)) {
                return 1;
            }
            continue;
        }

        if (arg == "--verbose") {
            verbose = true;
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (window_cfg.output_frames == 0) {
        std::cerr << "output-frames must be > 0" << std::endl;
        return 1;
    }

    const size_t total_window_frames = window_cfg.anchor_frames + window_cfg.output_frames + window_cfg.future_frames;

    std::cout << "Sliding window config:" << std::endl;
    std::cout << "  anchor frames: " << window_cfg.anchor_frames << std::endl;
    std::cout << "  output frames: " << window_cfg.output_frames << std::endl;
    std::cout << "  future frames: " << window_cfg.future_frames << std::endl;
    std::cout << "  total input frames per window: " << total_window_frames << std::endl;

    // Check input files exist before loading.
    for (const std::string& path : {quat_path_opt, yaml_path_opt, rcs_path_opt}) {
        if (!std::ifstream(path).good()) {
            std::cerr << "File not found: " << path << std::endl;
            return 1;
        }
    }

    DataLoader loader(quat_path_opt, yaml_path_opt, rcs_path_opt);
    if (!loader.load()) {
        std::cerr << "Failed to load input data." << std::endl;
        return 1;
    }

    const auto& frames = loader.frame_data;
    if (frames.empty()) {
        std::cerr << "No frames loaded." << std::endl;
        return 1;
    }
    std::cout << "Loaded frame count: " << frames.size() << std::endl;
    std::cout << "Camera: " << loader.width << "x" << loader.height
              << " crop H=" << loader.crop_factor_h << " V=" << loader.crop_factor_v
              << " f=" << loader.K(0, 0) << std::endl;

    std::vector<Eigen::Quaterniond> optimized_quaternions;
    optimized_quaternions.reserve(frames.size());

    // Optimization parameters for the GN+ALM solver.
    SolverConfig solver_config;
    solver_config.verbose = verbose;

    // Sliding-window optimization: process `frames` in overlapping windows,
    // each window = anchor + output + future frames; results accumulate in `optimized_quaternions`.
    size_t next_output_start = 0;
    for (size_t window_index = 0; next_output_start < frames.size(); ++window_index) {
        const bool first_window = (window_index == 0);

        std::vector<size_t> window_indices;
        std::vector<Eigen::Quaterniond> fixed_prefix_quaternions;
        window_indices.reserve(total_window_frames);
        fixed_prefix_quaternions.reserve(window_cfg.anchor_frames);

        // Anchor region: the first `anchor_frames` frames are fixed (not optimized).
        for (size_t i = 0; i < window_cfg.anchor_frames; ++i) {
            if (first_window) {
                fixed_prefix_quaternions.push_back(frames.front().orientation);
                window_indices.push_back(0);              // clamp to frame 0
            } else {
                const size_t anchor_idx = next_output_start - window_cfg.anchor_frames + i;
                fixed_prefix_quaternions.push_back(optimized_quaternions[anchor_idx]);
                window_indices.push_back(anchor_idx);
            }
        }

        // Core region: the next `output + future` frames, clamped at the tail.
        for (size_t i = 0; i < window_cfg.output_frames + window_cfg.future_frames; ++i) {
            const size_t idx = next_output_start + i;
            window_indices.push_back(idx < frames.size() ? idx : frames.size() - 1);
        }

        PROFILE_SCOPE("Window #" + std::to_string(window_index + 1));

        Optimizer optimizer(frames, std::move(window_indices),
                            loader.K,
                            loader.width,
                            loader.height,
                            loader.crop_factor_h,
                            loader.crop_factor_v,
                            solver_config);

        std::cout << "Window #" << (window_index + 1) << std::endl;
        std::vector<Eigen::Quaterniond> optimized_window = optimizer.solveWindowGN(fixed_prefix_quaternions);
        if (optimized_window.size() != total_window_frames) {
            std::cerr << "Optimization output size mismatch in window #" << (window_index + 1) << std::endl;
            return 1;
        }

        // Output this window's frames (skipping the anchor region) and advance the window
        const size_t output_count = std::min(window_cfg.output_frames, frames.size() - next_output_start);
        for (size_t i = 0; i < output_count; ++i) {
            optimized_quaternions.push_back(optimized_window[window_cfg.anchor_frames + i]);
        }
        std::cout << "Output=" << output_count
                  << " Total_output=" << optimized_quaternions.size()
                  << "\n" << std::endl;

        next_output_start += output_count;
    }

    if (optimized_quaternions.size() != frames.size()) {
        std::cerr << "Final output frame count mismatch. expected=" << frames.size()
                  << " actual=" << optimized_quaternions.size() << std::endl;
        return 1;
    }

    if (!FileUtils::saveQuaternions(output_path_opt, optimized_quaternions)) {
        std::cerr << "Failed to save output: " << output_path_opt << std::endl;
        return 1;
    }

    std::cout << "Saved optimized quaternions: " << output_path_opt << std::endl;
    return 0;
}
