#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "DataLoader.h"
#include "FileUtils.h"
#include "Optimizer.h"

namespace {

struct SlidingWindowConfig {
    size_t core_frames = 100;   // emit + future
    size_t overlap_frames = 3;  // fixed prefix for alignment
    size_t emit_frames = 70;    // output count per window
};

void printUsage(const char* exe_name) {
    std::cout << "Usage: " << exe_name
              << " [--core-frames N] [--overlap-frames N] [--emit-frames N]" << std::endl;
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
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    const std::string quat_path = "../data/DbgFile_Frame_OriginalCamOrient.csv";
    const std::string yaml_path = "../data/conf.yaml";
    const std::string rcs_path = "../data/DbgFile_WarpGrids_RSCFRMMid.csv";
    const std::string output_path = "../data/optimized_quaternions.csv";

    // Optimization parameters
    const double lambda1 = 1.0;
    const double lambda2 = 2.0;
    const double lambda3 = 4.0;
    const double mu = 0.25;

    SlidingWindowConfig window_cfg;
    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        std::string arg = argv[arg_idx];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--core-frames") {
            if (!readOptionValue(argc, argv, arg_idx, arg, window_cfg.core_frames)) {
                return 1;
            }
            continue;
        }
        if (arg == "--overlap-frames") {
            if (!readOptionValue(argc, argv, arg_idx, arg, window_cfg.overlap_frames)) {
                return 1;
            }
            continue;
        }
        if (arg == "--emit-frames") {
            if (!readOptionValue(argc, argv, arg_idx, arg, window_cfg.emit_frames)) {
                return 1;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (window_cfg.core_frames == 0) {
        std::cerr << "core-frames must be > 0" << std::endl;
        return 1;
    }
    if (window_cfg.emit_frames == 0) {
        std::cerr << "emit-frames must be > 0" << std::endl;
        return 1;
    }
    if (window_cfg.emit_frames > window_cfg.core_frames) {
        std::cerr << "emit-frames must be <= core-frames" << std::endl;
        return 1;
    }

    const size_t future_frames = window_cfg.core_frames - window_cfg.emit_frames;
    const size_t total_window_frames = window_cfg.overlap_frames + window_cfg.core_frames;

    std::cout << "Sliding window config:" << std::endl;
    std::cout << "  overlap frames: " << window_cfg.overlap_frames << std::endl;
    std::cout << "  emit frames: " << window_cfg.emit_frames << std::endl;
    std::cout << "  future frames: " << future_frames << std::endl;
    std::cout << "  total input frames per window: " << total_window_frames << std::endl;

    auto fileExists = [](const std::string& path) -> bool {
        std::ifstream file(path);
        return file.good();
    };

    if (!fileExists(quat_path)) {
        std::cerr << "Quaternion file not found: " << quat_path << std::endl;
        return 1;
    }
    if (!fileExists(yaml_path)) {
        std::cerr << "YAML file not found: " << yaml_path << std::endl;
        return 1;
    }
    if (!fileExists(rcs_path)) {
        std::cerr << "RCS grid file not found: " << rcs_path << std::endl;
        return 1;
    }

    DataLoader loader(quat_path, yaml_path, rcs_path);
    if (!loader.load()) {
        std::cerr << "Failed to load input data." << std::endl;
        return 1;
    }

    const auto& frames = loader.getFrameData();
    if (frames.empty()) {
        std::cerr << "No frames loaded." << std::endl;
        return 1;
    }
    std::cout << "Loaded frame count: " << frames.size() << std::endl;

    std::vector<Eigen::Quaterniond> optimized_quaternions;
    optimized_quaternions.reserve(frames.size());

    size_t next_emit_start = 0;
    size_t window_index = 0;
    while (next_emit_start < frames.size()) {
        const long long window_start = static_cast<long long>(next_emit_start) -
                                       static_cast<long long>(window_cfg.overlap_frames);

        std::vector<FrameData> window_frames;
        window_frames.reserve(total_window_frames);
        for (size_t local_idx = 0; local_idx < total_window_frames; ++local_idx) {
            long long source_idx = window_start + static_cast<long long>(local_idx);
            if (source_idx < 0) {
                source_idx = 0;
            }
            if (source_idx >= static_cast<long long>(frames.size())) {
                source_idx = static_cast<long long>(frames.size()) - 1;
            }
            window_frames.push_back(frames[static_cast<size_t>(source_idx)]);
        }

        std::vector<Eigen::Quaterniond> fixed_prefix;
        fixed_prefix.reserve(window_cfg.overlap_frames);
        if (window_cfg.overlap_frames > 0) {
            if (window_index == 0) {
                for (size_t i = 0; i < window_cfg.overlap_frames; ++i) {
                    fixed_prefix.push_back(frames.front().orientation);
                }
            } else {
                const size_t fixed_global_start = next_emit_start - window_cfg.overlap_frames;
                for (size_t i = 0; i < window_cfg.overlap_frames; ++i) {
                    fixed_prefix.push_back(optimized_quaternions[fixed_global_start + i]);
                }
            }
        }

        std::cout << "Window #" << (window_index + 1)
                  << " start=" << window_start
                  << " fixed_prefix=" << fixed_prefix.size()
                  << " total_input=" << window_frames.size() << std::endl;

        Optimizer optimizer(window_frames,
                            loader.getCameraMatrix(),
                            loader.getWidth(),
                            loader.getHeight(),
                            loader.getCropFactorH(),
                            loader.getCropFactorV(),
                            lambda1, lambda2, lambda3, mu);

        std::vector<Eigen::Quaterniond> optimized_window = optimizer.optimize(fixed_prefix);
        if (optimized_window.size() != total_window_frames) {
            std::cerr << "Optimization output size mismatch in window #" << (window_index + 1) << std::endl;
            return 1;
        }

        const size_t emit_count = std::min(window_cfg.emit_frames, frames.size() - next_emit_start);
        for (size_t i = 0; i < emit_count; ++i) {
            optimized_quaternions.push_back(optimized_window[window_cfg.overlap_frames + i]);
        }

        std::cout << "Window #" << (window_index + 1)
                  << " emitted=" << emit_count
                  << " total_emitted=" << optimized_quaternions.size()
                  << std::endl;

        next_emit_start += emit_count;
        ++window_index;
    }

    if (optimized_quaternions.size() != frames.size()) {
        std::cerr << "Final output frame count mismatch. expected=" << frames.size()
                  << " actual=" << optimized_quaternions.size() << std::endl;
        return 1;
    }

    if (!FileUtils::saveQuaternions(output_path, optimized_quaternions)) {
        std::cerr << "Failed to save output: " << output_path << std::endl;
        return 1;
    }

    std::cout << "Saved optimized quaternions: " << output_path << std::endl;
    return 0;
}
