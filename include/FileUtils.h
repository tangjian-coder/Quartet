
#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace FileUtils {

/**
 * @brief Saves a vector of quaternions to a text file.
 *
 * @param file_path The path to the output file.
 * @param quaternions The vector of quaternions to save.
 * @return True if successful, false otherwise.
 */
bool saveQuaternions(const std::string& file_path, const std::vector<Eigen::Quaterniond>& quaternions);

}

#endif // FILE_UTILS_H
