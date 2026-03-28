
#include "FileUtils.h"
#include <fstream>
#include <iomanip>

namespace FileUtils {

bool saveQuaternions(const std::string& file_path, const std::vector<Eigen::Quaterniond>& quaternions) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }

    file << std::fixed << std::setprecision(8);

    for (const auto& q : quaternions) {
        file << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "\n";
    }

    return true;
}

}
