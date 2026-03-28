
#include "GeometryTools.h"
#include <iostream>
#include <limits>
#include <algorithm>

namespace GeometryTools {

Eigen::Matrix3d calculateHomography(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, const Eigen::Matrix3d& K) {
    Eigen::Matrix3d R1 = q1.toRotationMatrix();
    Eigen::Matrix3d R2 = q2.toRotationMatrix();
    return K * (R2 * R1.transpose()) * K.inverse();
}

Eigen::Vector2d transformPoint(const Eigen::Vector2d& point, const Eigen::Matrix3d& H) {
    Eigen::Vector3d p_homogeneous(point.x(), point.y(), 1.0);
    Eigen::Vector3d p_transformed = H * p_homogeneous;
    return Eigen::Vector2d(p_transformed.x() / p_transformed.z(), p_transformed.y() / p_transformed.z());
}

// Helper function to calculate the minimum distance from a point to a line segment.
double pointToLineSegmentDistance(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d ap = p - a;
    double t = ap.dot(ab) / ab.squaredNorm();
    t = std::max(0.0, std::min(1.0, t));
    Eigen::Vector2d closest_point = a + t * ab;
    return (p - closest_point).norm();
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

double signedDistanceFromPointToPolygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon) {
    if (polygon.empty()) {
        return std::numeric_limits<double>::max();
    }

    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < polygon.size(); ++i) {
        Eigen::Vector2d p1 = polygon[i];
        Eigen::Vector2d p2 = polygon[(i + 1) % polygon.size()];
        min_dist = std::min(min_dist, pointToLineSegmentDistance(point, p1, p2));
    }

    if (isPointInsidePolygon(point, polygon)) {
        return -min_dist;
    } else {
        return min_dist;
    }
}

}
