
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
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
    double eps_feas = 1e-5;   // Outer KKT: feasibility  ||[g]+||  (0.01 px @ f=1000)
    double eps_stat = 1e-3;   // Outer KKT: stationarity ||J_r^T r + J_g^T lambda||
    double eps_comp = 1e-5;   // Outer KKT: complementarity ||lambda (.) g||
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
    double focal_length_ = 0.0;
    std::vector<std::vector<double>> g_cached_;
    std::vector<std::vector<Eigen::RowVector3d>> J_cached_;

    void precomputeBoundary(size_t prefix);
    double evaluateSDF(const std::vector<Eigen::Quaterniond>& q_state);
    // Augmented objective Phi = 0.5||R||^2 (smoothness + fidelity + constraint penalty), for diagnostics.
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
