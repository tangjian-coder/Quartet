
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include "FrameData.h"
#include <yaml-cpp/yaml.h>

class DataLoader {
public:
    DataLoader(const std::string& quat_path, const std::string& yaml_path, const std::string& rcs_path);

    bool load();

    // Data members: read-only after load()
    std::vector<FrameData> frame_data;
    Eigen::Matrix3d K;                // Camera intrinsic matrix
    int width;                        // Image width
    int height;                       // Image height
    double crop_factor_h;             // Horizontal crop factor
    double crop_factor_v;             // Vertical crop factor

private:
    bool loadQuaternionData();
    bool loadYamlData();
    bool loadRcsGridData();

    std::string quat_file_path;
    std::string yaml_file_path;
    std::string rcs_file_path;
};

#endif // DATA_LOADER_H
