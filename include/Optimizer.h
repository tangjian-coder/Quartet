
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <Eigen/Dense>
#include <nlopt.hpp>

// Forward declaration
struct FrameData;

class Optimizer {
public:
    Optimizer(const std::vector<FrameData>& initial_data,
              const Eigen::Matrix3d& K,
              int width, int height,
              double crop_factor_h, double crop_factor_v,
              double lambda1, double lambda2, double lambda3, double mu);

    std::vector<Eigen::Quaterniond> optimize(const std::vector<Eigen::Quaterniond>& fixed_prefix_filtered = {});

    /// Set whether to use analytic gradient (if implemented) or numerical gradient.
    /// Default is true (use analytic gradient).
    void setUseAnalyticGradient(bool use) { use_analytic_gradient = use; }

    /// Enable/disable analytic gradient (same as setUseAnalyticGradient).
    void enableAnalyticGradient(bool enable = true) { use_analytic_gradient = enable; }

    /// Enable/disable analytic-vs-numerical gradient check logs.
    void enableGradientCheck(bool enable = true) { gradient_check_enabled = enable; }

private:
    static double objective(const std::vector<double>& x, std::vector<double>& grad, void* data);
    static void constraint(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data);

    // Helper to convert from axis-angle (3 doubles) to quaternion
    static Eigen::Quaterniond axisAngleToQuaternion(const Eigen::Vector3d& axis_angle);

    // Helper to convert from quaternion to axis-angle
    static Eigen::Vector3d quaternionToAxisAngle(const Eigen::Quaterniond& q);

    // Analytic gradient computation (to be implemented by user)
    void computeAnalyticGradient(const std::vector<double>& x, std::vector<double>& grad);

    // Numerical gradient computation using finite differences
    void computeNumericalGradient(const std::vector<double>& x, std::vector<double>& grad);

    // Compute objective value for given optimization variables x
    double computeObjectiveValue(const std::vector<double>& x);

    std::vector<FrameData> initial_frame_data;
    Eigen::Matrix3d K;                // Camera intrinsic matrix
    int width;                        // Image width
    int height;                       // Image height
    double crop_factor_h;             // Horizontal crop factor
    double crop_factor_v;             // Vertical crop factor

    double lambda1, lambda2, lambda3, mu;
    int iteration_count;
    bool use_analytic_gradient;       // Flag to choose gradient method
    bool gradient_check_enabled;      // Flag to compare analytic and numerical gradients
};

#endif // OPTIMIZER_H
