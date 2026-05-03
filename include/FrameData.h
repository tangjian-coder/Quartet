#ifndef FRAME_DATA_H
#define FRAME_DATA_H

#include <Eigen/Dense>
#include <vector>

struct FrameData {
    Eigen::Quaterniond orientation;
    std::vector<Eigen::Vector2d> rcs_grid;
};

#endif // FRAME_DATA_H
