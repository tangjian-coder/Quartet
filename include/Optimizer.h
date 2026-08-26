
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <array>
#include <Eigen/Dense>
#include "FrameData.h"

struct SolverConfig {
    double lambda1 = 1.0;
    double lambda2 = 2.0;
    double lambda3 = 8.0;
    double mu = 0.02;
    double rho_init = 20.0;
    double rho_max = 500.0;
    double gamma = 2.0;
    int max_outer = 5;
    int max_inner = 8;
    double eps_feas = 4e-5;   // Outer: feasibility ||[g]+|| (rad) ≈ violation_tol_px / f
    double eps_stat = 1e-3;   // Outer KKT: stationarity (diagnostic only; not used for stopping)
    double eps_comp = 1e-5;   // Outer KKT: complementarity ||lambda (.) g||
    // Physical violation tolerance (px). The outer loop stops below this
    // (inverse-warp metric); verifyConstraints accepts up to 2x (forward-warp).
    double violation_tol_px = 0.05;
    // Inner loop: damped GN with Armijo acceptance — rejected steps raise mu
    // and retry without consuming an iteration. Plain full-step GN was observed
    // to enter limit cycles (dPhi<0 steps) on this nonconvex objective.
    bool verbose = false;   // Print per-window solver diagnostics.
};

class Optimizer {
public:
    Optimizer(const std::vector<FrameData>& all_frames,
              std::vector<size_t> window_indices,
              const Eigen::Matrix3d& K,
              int width, int height,
              double crop_factor_h, double crop_factor_v,
              const SolverConfig& config = SolverConfig());

    std::vector<Eigen::Quaterniond> solveWindowGN(const std::vector<Eigen::Quaterniond>& fixed_prefix_filtered = {});

private:
    const std::vector<FrameData>* all_frames_;
    std::vector<size_t> window_indices_;
    Eigen::Matrix3d K;
    Eigen::Matrix3d Kinv;
    int width;
    int height;
    double crop_factor_h;
    double crop_factor_v;
    SolverConfig config_;

    size_t prefix_ = 0;
    size_t N_ = 0;
    std::vector<std::vector<Eigen::Vector2d>> fixed_boundary_;
    Eigen::Vector3d corner_rays_[4];
    // Per-frame coarse boundary: 4 coarse edges with inward safe offset
    // (bulge + margin). A corner inside all of them is constraint-inactive.
    struct CoarseEdge { Eigen::Vector2d p0, n_in; double offset; };
    std::vector<std::array<CoarseEdge, 4>> coarse_edges_;
    double focal_length_ = 0.0;
    std::vector<std::vector<double>> g_cached_;
    std::vector<std::vector<Eigen::RowVector3d>> J_cached_;

    void precomputeBoundary(size_t prefix);
    double evaluateSDF(const std::vector<Eigen::Quaterniond>& q_state);
    // Augmented objective Phi = 0.5||R||^2 (smoothness + fidelity + constraint
    // penalty); used by the damped-GN acceptance test. Requires g_cached_ fresh.
    double evaluateObjective(const std::vector<Eigen::Quaterniond>& q_state,
                             double rho, double inv_f,
                             const std::vector<std::vector<double>>& lambda);
    void buildSmoothnessHessian(const std::vector<Eigen::Quaterniond>& q_state,
                                Eigen::Ref<Eigen::MatrixXd> H,
                                Eigen::Ref<Eigen::VectorXd> b);
    void buildConstraintPenalty(double rho, double inv_f,
                                const std::vector<std::vector<double>>& lambda,
                                Eigen::Ref<Eigen::MatrixXd> H,
                                Eigen::Ref<Eigen::VectorXd> b);
    void verifyConstraints(const std::vector<Eigen::Quaterniond>& q_state);
    // Three KKT errors at (x, lambda^{k+1}) used as the outer stop criteria.
    void computeKKTErrors(const std::vector<Eigen::Quaterniond>& q_state,
                          double rho, double inv_f,
                          const std::vector<std::vector<double>>& lambda,
                          double& e_feas, double& e_stat, double& e_comp);
};

#endif // OPTIMIZER_H
