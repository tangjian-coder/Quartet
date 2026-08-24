
#include "GeometryTools.h"
#include <iostream>
#include <limits>
#include <algorithm>

namespace GeometryTools {

Eigen::Vector2d transformPoint(const Eigen::Vector2d& point, const Eigen::Matrix3d& H) {
    Eigen::Vector3d p_homogeneous(point.x(), point.y(), 1.0);
    Eigen::Vector3d p_transformed = H * p_homogeneous;
    return Eigen::Vector2d(p_transformed.x() / p_transformed.z(), p_transformed.y() / p_transformed.z());
}

// Helper function to check if a point is inside a polygon using the winding number algorithm.
bool isPointInsidePolygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon) {
    int winding_number = 0;
    for (size_t i = 0; i < polygon.size(); ++i) {
        Eigen::Vector2d p1 = polygon[i];
        Eigen::Vector2d p2 = polygon[(i + 1) % polygon.size()];

        if (p1.y() <= point.y()) {
            if (p2.y() > point.y() && (p2.x() - p1.x()) * (point.y() - p1.y()) - (p2.y() - p1.y()) * (point.x() - p1.x()) > 0) {
                winding_number++;
            }
        } else {
            if (p2.y() <= point.y() && (p2.x() - p1.x()) * (point.y() - p1.y()) - (p2.y() - p1.y()) * (point.x() - p1.x()) < 0) {
                winding_number--;
            }
        }
    }
    return winding_number != 0;
}

double signedDistanceFromPointToPolygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon,
                                           int* closest_edge_idx, double* t_param) {
    if (polygon.empty()) {
        if (closest_edge_idx) *closest_edge_idx = -1;
        if (t_param) *t_param = 0.0;
        return std::numeric_limits<double>::max();
    }

    double min_dist = std::numeric_limits<double>::max();
    int best_idx = -1;
    double best_t_raw = 0.0;
    int winding_number = 0;   // inside test fused into the same edge pass
    for (size_t i = 0; i < polygon.size(); ++i) {
        Eigen::Vector2d p1 = polygon[i];
        Eigen::Vector2d p2 = polygon[(i + 1) % polygon.size()];
        Eigen::Vector2d ab = p2 - p1;
        Eigen::Vector2d ap = point - p1;
        double t_raw = ap.dot(ab) / ab.squaredNorm();
        double t = std::max(0.0, std::min(1.0, t_raw));
        Eigen::Vector2d closest = p1 + t * ab;
        double d = (point - closest).norm();
        if (d < min_dist) {
            min_dist = d;
            best_idx = static_cast<int>(i);
            best_t_raw = t_raw;
        }
        if (p1.y() <= point.y()) {
            if (p2.y() > point.y() && (p2.x() - p1.x()) * (point.y() - p1.y()) - (p2.y() - p1.y()) * (point.x() - p1.x()) > 0) {
                winding_number++;
            }
        } else {
            if (p2.y() <= point.y() && (p2.x() - p1.x()) * (point.y() - p1.y()) - (p2.y() - p1.y()) * (point.x() - p1.x()) < 0) {
                winding_number--;
            }
        }
    }

    if (closest_edge_idx) *closest_edge_idx = best_idx;
    if (t_param) *t_param = best_t_raw;

    return (winding_number != 0) ? -min_dist : min_dist;
}

}
