#ifndef GRID_LAYOUT_H
#define GRID_LAYOUT_H

#include <array>
#include <cstddef>
#include <vector>

/* RCS grid layout helpers.
 *
 * The grid size is per-dataset (GoPro keeps 55x55 because its fisheye boundary is
 * strongly curved, ADVIO/DVS use 11x11), so it arrives at runtime through conf.yaml
 * and the loader validates the point count against it. */

// Boundary indices for a cols x rows row-major RCS grid, walked clockwise from
// the top-left corner. 2*(cols+rows)-4 points for cols,rows >= 2.
inline std::vector<size_t> boundaryIndices(int cols, int rows) {
    std::vector<size_t> v;
    v.reserve(static_cast<size_t>(2 * (cols + rows)));
    for (int j = 0; j < cols; ++j) v.push_back(static_cast<size_t>(j));
    for (int i = 1; i < rows; ++i) v.push_back(static_cast<size_t>(i) * cols + (cols - 1));
    for (int j = cols - 2; j >= 0; --j) v.push_back(static_cast<size_t>(rows - 1) * cols + j);
    for (int i = rows - 2; i > 0; --i) v.push_back(static_cast<size_t>(i) * cols);
    return v;
}

// Number of segments along each boundary side, in the order the walk visits them:
// top (cols-1), right (rows-1), bottom (cols-1), left (rows-1).
inline std::array<int, 4> boundaryEdgeSegs(int cols, int rows) {
    return {cols - 1, rows - 1, cols - 1, rows - 1};
}

#endif // GRID_LAYOUT_H
