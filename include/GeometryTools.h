
#ifndef GEOMETRY_TOOLS_H
#define GEOMETRY_TOOLS_H

#include <Eigen/Dense>
#include <vector>

namespace GeometryTools {

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
double signedDistanceFromPointToPolygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon,
                                           int* closest_edge_idx = nullptr, double* t_param = nullptr);

}

#endif // GEOMETRY_TOOLS_H
