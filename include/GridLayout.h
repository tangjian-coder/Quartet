#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include <cstddef>
#include <vector>

/* Shared RCS grid layout: the optimizer derives its boundary polygon from the
 * constant below, so the loader must validate conf against the same value. */

constexpr int kGridSide = 11;        // RCS grid is kGridSide x kGridSide
constexpr int kEdgeSegs = kGridSide - 1;

// Boundary indices for a kGridSide x kGridSide row-major RCS grid (4*kEdgeSegs
// outer points, walked clockwise from the top-left corner).
inline const std::vector<size_t>& getBoundaryIndices() {
    static const std::vector<size_t> idx = []() {
        std::vector<size_t> v;
        for (int j = 0; j < kGridSide; ++j) v.push_back(j);
        for (int i = 1; i <= kEdgeSegs; ++i) v.push_back(i * kGridSide + kEdgeSegs);
        for (int j = kEdgeSegs - 1; j >= 0; --j) v.push_back(kEdgeSegs * kGridSide + j);
        for (int i = kEdgeSegs - 1; i > 0; --i) v.push_back(i * kGridSide);
        return v;
    }();
    return idx;
}

/* TODO: generalizing to arbitrary grid dims requires boundary indexing by
 * (cols, rows) plus an Optimizer that receives those dims explicitly. */

#endif // GRID_LAYOUT_H
