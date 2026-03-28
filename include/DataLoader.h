
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

struct FrameData {
    Eigen::Quaterniond orientation;
    std::vector<Eigen::Vector2d> rcs_grid;
};

class DataLoader {
public:
    DataLoader(const std::string& quat_path, const std::string& yaml_path, const std::string& rcs_path);

    bool load();

    const std::vector<FrameData>& getFrameData() const;
    const Eigen::Matrix3d& getCameraMatrix() const;
    int getWidth() const;
    int getHeight() const;
    double getCropFactorH() const;
    double getCropFactorV() const;

private:
    bool loadQuaternionData();
    bool loadYamlData();
    bool loadRcsGridData();

    std::string quat_file_path;
    std::string yaml_file_path;
    std::string rcs_file_path;

    std::vector<FrameData> frame_data;
    Eigen::Matrix3d K;                // Camera intrinsic matrix
    int width;                        // Image width
    int height;                       // Image height
    double crop_factor_h;             // Horizontal crop factor
    double crop_factor_v;             // Vertical crop factor
};

#endif // DATA_LOADER_H
