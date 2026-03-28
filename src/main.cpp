
#include <iostream>
#include <fstream>
#include "DataLoader.h"
#include "Optimizer.h"
#include "FileUtils.h"

int main(int argc, char** argv) {
    // Force unbuffered output
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    
    // Test output immediately
    std::cout << "TEST: Program starting... " << std::endl;
    std::cout << "TEST: argc = " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "TEST: argv[" << i << "] = " << argv[i] << std::endl;
    }
    
    // --- Configuration ---
    std::cout << "Starting quaternion optimization program..." << std::endl;
    const std::string quat_path = "../data/DbgFile_Frame_OriginalCamOrient.csv";
    const std::string yaml_path = "../data/conf.yaml";
    const std::string rcs_path = "../data/DbgFile_WarpGrids_RSCFRMMid.csv";
    const std::string output_path = "../data/optimized_quaternions.csv";

    // Optimization parameters (these should be tuned)
    const double lambda1 = 1;  // Velocity
    const double lambda2 = 2;  // Acceleration
    const double lambda3 = 4;  // Jerk
    const double mu = 1;       // Fidelity

    // --- 1. Load Data ---
    std::cout << "MAIN: Loading data..." << std::endl;
    std::cout << "MAIN: Quaternion file: " << quat_path << std::endl;
    std::cout << "MAIN: YAML config file: " << yaml_path << std::endl;
    std::cout << "MAIN: RCS grid file: " << rcs_path << std::endl;
    
    // Check if files exist
    auto fileExists = [](const std::string& path) -> bool {
        std::ifstream f(path);
        bool exists = f.good();
        std::cout << "MAIN: File " << path << (exists ? " EXISTS" : " NOT FOUND") << std::endl;
        return exists;
    };
    
    if (!fileExists(quat_path)) {
        std::cerr << "MAIN ERROR: Quaternion file not found: " << quat_path << std::endl;
        return 1;
    }
    if (!fileExists(yaml_path)) {
        std::cerr << "MAIN ERROR: YAML file not found: " << yaml_path << std::endl;
        return 1;
    }
    if (!fileExists(rcs_path)) {
        std::cerr << "MAIN ERROR: RCS grid file not found: " << rcs_path << std::endl;
        return 1;
    }
    
    std::cout << "MAIN: All files exist, creating DataLoader..." << std::endl;
    DataLoader loader(quat_path, yaml_path, rcs_path);
    std::cout << "MAIN: Calling loader.load()..." << std::endl;
    if (!loader.load()) {
        std::cerr << "MAIN ERROR: Failed to load data. Exiting." << std::endl;
        return 1;
    }
    std::cout << "MAIN: Data loaded successfully." << std::endl;
    const auto& frames = loader.getFrameData();
    std::cout << "MAIN: Loaded " << frames.size() << " frames." << std::endl;
    if (!frames.empty()) {
        std::cout << "MAIN: First frame orientation: w=" << frames[0].orientation.w() 
                  << " x=" << frames[0].orientation.x() 
                  << " y=" << frames[0].orientation.y() 
                  << " z=" << frames[0].orientation.z() << std::endl;
        if (!frames[0].rcs_grid.empty()) {
            std::cout << "MAIN: First frame has " << frames[0].rcs_grid.size() << " grid points (expected 121 for 11x11 grid)" << std::endl;
            // Print first few points for verification
            std::cout << "MAIN: First 3 grid points: ";
            for (int i = 0; i < 3 && i < frames[0].rcs_grid.size(); ++i) {
                std::cout << "(" << frames[0].rcs_grid[i].x() << ", " << frames[0].rcs_grid[i].y() << ") ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "MAIN: First frame has NO grid points!" << std::endl;
        }
    }
    std::cout << "MAIN: Camera matrix K:\n" << loader.getCameraMatrix() << std::endl;
    std::cout << "MAIN: Image size: " << loader.getWidth() << " x " << loader.getHeight() << std::endl;
    std::cout << "MAIN: Crop factors: H=" << loader.getCropFactorH() << ", V=" << loader.getCropFactorV() << std::endl;
    std::cout << "MAIN: Moving to optimization phase..." << std::endl;

    // --- 2. Run Optimization ---
    std::cout << "Starting optimization..." << std::endl;
    Optimizer optimizer(loader.getFrameData(), 
                        loader.getCameraMatrix(),
                        loader.getWidth(),
                        loader.getHeight(),
                        loader.getCropFactorH(),
                        loader.getCropFactorV(),
                        lambda1, lambda2, lambda3, mu);
    
    std::vector<Eigen::Quaterniond> optimized_quaternions = optimizer.optimize();

    if (optimized_quaternions.empty()) {
        std::cerr << "Optimization failed or returned no result. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Optimization finished." << std::endl;

    // --- 3. Save Results ---
    std::cout << "Saving results to " << output_path << "..." << std::endl;
    if (FileUtils::saveQuaternions(output_path, optimized_quaternions)) {
        std::cout << "Results saved successfully." << std::endl;
    } else {
        std::cerr << "Failed to save results." << std::endl;
        return 1;
    }

    return 0;
}
