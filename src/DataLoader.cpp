
#include "DataLoader.h"
#include <fstream>
#include <iostream>
#include <sstream>

DataLoader::DataLoader(const std::string& quat_path, const std::string& yaml_path, const std::string& rcs_path)
    : quat_file_path(quat_path), yaml_file_path(yaml_path), rcs_file_path(rcs_path),
      K(Eigen::Matrix3d::Zero()), width(0), height(0), crop_factor_h(0.0), crop_factor_v(0.0) {
    std::cout << "DataLoader constructor called with paths:" << std::endl;
    std::cout << "  Quaternion: " << quat_file_path << std::endl;
    std::cout << "  YAML: " << yaml_file_path << std::endl;
    std::cout << "  RCS: " << rcs_file_path << std::endl;
}

bool DataLoader::load() {
    std::cout << "DataLoader::load() - Starting to load quaternion data..." << std::endl;
    if (!loadQuaternionData()) {
        std::cerr << "Error loading quaternion data." << std::endl;
        return false;
    }
    std::cout << "DataLoader::load() - Quaternion data loaded, now loading YAML..." << std::endl;
    if (!loadYamlData()) {
        std::cerr << "Error loading YAML data." << std::endl;
        return false;
    }
    std::cout << "DataLoader::load() - YAML data loaded, now loading RCS grid..." << std::endl;
    if (!loadRcsGridData()) {
        std::cerr << "Error loading RCS grid data." << std::endl;
        return false;
    }
    std::cout << "DataLoader::load() - All data loaded successfully!" << std::endl;
    return true;
}

const std::vector<FrameData>& DataLoader::getFrameData() const {
    return frame_data;
}

const Eigen::Matrix3d& DataLoader::getCameraMatrix() const {
    return K;
}

int DataLoader::getWidth() const {
    return width;
}

int DataLoader::getHeight() const {
    return height;
}

double DataLoader::getCropFactorH() const {
    return crop_factor_h;
}

double DataLoader::getCropFactorV() const {
    return crop_factor_v;
}

bool DataLoader::loadQuaternionData() {
    std::cout << "DEBUG: loadQuaternionData() - Opening file: " << quat_file_path << std::endl;
    std::ifstream file(quat_file_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open quaternion file: " << quat_file_path << std::endl;
        return false;
    }
    std::cout << "DEBUG: File opened successfully" << std::endl;

    std::string line;
    // Skip header
    if (!std::getline(file, line)) {
        std::cerr << "ERROR: Failed to read header from quaternion file." << std::endl;
        return false;
    }
    std::cout << "DEBUG: Header skipped: " << line.substr(0, 50) << "..." << std::endl;

    size_t line_num = 1;
    size_t frames_loaded = 0;
    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss(line);
        std::string token;
        double w = 0, x = 0, y = 0, z = 0;
        int count = 0;

        // Skip time (first column)
        if (!std::getline(ss, token, ',')) continue;

        // Parse quaternion components
        while (std::getline(ss, token, ',') && count < 4) {
            try {
                double val = std::stod(token);
                if (count == 0) w = val;
                else if (count == 1) x = val;
                else if (count == 2) y = val;
                else if (count == 3) z = val;
                count++;
            } catch (...) {
                std::cerr << "Line " << line_num << ": invalid number format" << std::endl;
                return false;
            }
        }

        if (count == 4) {
            frame_data.push_back({Eigen::Quaterniond(w, x, y, z)});
            frames_loaded++;
        } else {
            std::cerr << "Line " << line_num << ": expected 4 values but got " << count << std::endl;
            return false;
        }
        
        if (frames_loaded % 100 == 0) {
            std::cout << "DEBUG: Loaded " << frames_loaded << " frames so far..." << std::endl;
        }
    }
    
    if (frame_data.empty()) {
        std::cerr << "ERROR: No quaternion data loaded from file." << std::endl;
        return false;
    }
    
    std::cout << "DEBUG: Successfully loaded " << frame_data.size() << " quaternion frames." << std::endl;
    return true;
}

bool DataLoader::loadYamlData() {
    try {
        YAML::Node config = YAML::LoadFile(yaml_file_path);

        // Read focal length
        double focal_length = config["calFocalLength"].as<double>();
        
        // Set up camera intrinsic matrix K
        // K = [f 0 0; 0 f 0; 0 0 1]
        K.setZero();
        K(0, 0) = focal_length;
        K(1, 1) = focal_length;
        K(2, 2) = 1.0;

        // Read image width and height
        width = config["frmWidth"].as<int>();
        height = config["frmHeight"].as<int>();

        // Read crop factors
        crop_factor_h = config["gridFrmCropFctrH"].as<double>();
        crop_factor_v = config["gridFrmCropFctrV"].as<double>();

        // Log loaded parameters for verification
        std::cout << "Loaded YAML parameters:" << std::endl;
        std::cout << "  Focal length: " << focal_length << std::endl;
        std::cout << "  Image size: " << width << " x " << height << std::endl;
        std::cout << "  Crop factors: H=" << crop_factor_h << ", V=" << crop_factor_v << std::endl;

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool DataLoader::loadRcsGridData() {
    std::ifstream file(rcs_file_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open RCS grid file: " << rcs_file_path << std::endl;
        return false;
    }

    std::string line;
    size_t frame_idx = 0;
    while (std::getline(file, line) && frame_idx < frame_data.size()) {
        std::stringstream ss(line);
        std::string token;
        size_t point_count = 0;
        
        while (std::getline(ss, token, ',')) {
            try {
                double x = std::stod(token) * width;
                if (!std::getline(ss, token, ',')) {
                    std::cerr << "ERROR: Frame " << frame_idx << ": missing y coordinate for point " << point_count << std::endl;
                    return false;
                }
                double y = std::stod(token) * height;
                frame_data[frame_idx].rcs_grid.push_back(Eigen::Vector2d(x, y));
                point_count++;
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Frame " << frame_idx << ": invalid number format at point " << point_count << ": " << e.what() << std::endl;
                return false;
            } catch (...) {
                std::cerr << "ERROR: Frame " << frame_idx << ": unknown error parsing point " << point_count << std::endl;
                return false;
            }
        }
        frame_idx++;
    }
    
    if (frame_idx != frame_data.size()) {
        std::cerr << "ERROR: Loaded " << frame_idx << " RCS frames but expected " << frame_data.size() << std::endl;
        return false;
    }
    
    // 11x11网格应该有121个点
    if (!frame_data.empty()) {
        size_t points = frame_data[0].rcs_grid.size();
        std::cout << "RCS grid loaded: " << points << " points per frame (expected 121 for 11x11 grid)" << std::endl;
        if (points != 121) {
            std::cout << "  Note: Config says gridCntH:11, gridCntV:11 (11x11=121)" << std::endl;
        }
    }
    
    return true;
}
