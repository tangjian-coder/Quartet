
#ifndef GEOMETRY_TOOLS_H
#define GEOMETRY_TOOLS_H

#include <Eigen/Dense>
#include <vector>
#include "DataLoader.h"

namespace GeometryTools {

/**
 * @brief Calculates the homography matrix H = K * (R2 * R1.transpose()) * K.inverse().
 *
 * @param q1 Original orientation.
 * @param q2 Smoothed orientation.
 * @param K Camera intrinsics matrix.
 * @return The 3x3 homography matrix.
 */
Eigen::Matrix3d calculateHomography(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, const Eigen::Matrix3d& K);

/**
 * @brief Transforms a 2D point using a homography matrix.
 *
 * @param point The point in 2D image coordinates.
 * @param H The 3x3 homography matrix.
 * @return The transformed 2D point.
 */
Eigen::Vector2d transformPoint(const Eigen::Vector2d& point, const Eigen::Matrix3d& H);

/**
 * @brief Calculates the signed distance from a point to a polygon.
 * The distance is negative if the point is inside the polygon, positive if outside,
 * and zero if on the boundary.
 *
 * @param point The point to check.
 * @param polygon The vertices of the polygon in order.
 * @return The signed distance.
 */
double signedDistanceFromPointToPolygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon);

}

#endif // GEOMETRY_TOOLS_H
