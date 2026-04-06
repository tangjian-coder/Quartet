#include "Optimizer.h"
#include "DataLoader.h"
#include "GeometryTools.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace {

Eigen::Matrix3d hat(const Eigen::Vector3d& vector) {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -vector.z();
    skew(0, 2) = vector.y();
    skew(1, 0) = vector.z();
    skew(1, 2) = -vector.x();
    skew(2, 0) = -vector.y();
    skew(2, 1) = vector.x();
    return skew;
}

Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d A = hat(phi);

    if (theta < 1e-8) {
        const Eigen::Matrix3d A2 = A * A;
        return I + 0.5 * A + (1.0 / 6.0) * A2;
    }

    const double theta2 = theta * theta;
    const double a = (1.0 - std::cos(theta)) / theta2;
    const double b = (theta - std::sin(theta)) / (theta2 * theta);
    return I + a * A + b * (A * A);
}

Eigen::Matrix3d leftJacobianInverseSO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d A = hat(phi);

    if (theta < 1e-8) {
        const Eigen::Matrix3d A2 = A * A;
        return I - 0.5 * A + (1.0 / 12.0) * A2;
    }

    const double theta2 = theta * theta;
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    const double c = (1.0 / theta2) - ((1.0 + cos_theta) / (2.0 * theta * sin_theta));
    return I - 0.5 * A + c * (A * A);
}

Eigen::Vector3d readFrameAxisAngle(const std::vector<double>& x, size_t frame_index) {
    const size_t base = frame_index * 3;
    return Eigen::Vector3d(x[base], x[base + 1], x[base + 2]);
}

void addFrameGradient(std::vector<double>& grad, size_t frame_index, const Eigen::Vector3d& delta) {
    const size_t base = frame_index * 3;
    grad[base] += delta.x();
    grad[base + 1] += delta.y();
    grad[base + 2] += delta.z();
}

} // namespace

Optimizer::Optimizer(const std::vector<FrameData>& initial_data,
                     const Eigen::Matrix3d& K,
                     int width, int height,
                     double crop_factor_h, double crop_factor_v,
                     double l1, double l2, double l3, double m)
    : initial_frame_data(initial_data),
      K(K),
      width(width),
      height(height),
      crop_factor_h(crop_factor_h),
      crop_factor_v(crop_factor_v),
      lambda1(l1), lambda2(l2), lambda3(l3), mu(m),
      iteration_count(0),
      use_analytic_gradient(true),
      gradient_check_enabled(false) {
    std::cout << "Optimizer initialized with:" << std::endl;
    std::cout << "  Image size: " << width << " x " << height << std::endl;
    std::cout << "  Crop factors: H=" << crop_factor_h << ", V=" << crop_factor_v << std::endl;
    std::cout << "  Camera matrix K:\n" << K << std::endl;
}

std::vector<Eigen::Quaterniond> Optimizer::optimize(const std::vector<Eigen::Quaterniond>& fixed_prefix_filtered) {
    size_t num_frames = initial_frame_data.size();
    if (num_frames == 0) {
        return {};
    }

    size_t fixed_prefix_count = std::min(fixed_prefix_filtered.size(), num_frames);
    std::vector<double> x(num_frames * 3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.05, 0.05); // +/-0.05 rad
    for (size_t i = 0; i < num_frames * 3; ++i) {
        x[i] = dist(gen);
    }

    nlopt::opt opt(nlopt::LD_MMA, num_frames * 3);
    opt.set_min_objective(Optimizer::objective, this);

    // TODO: Add constraints after gradient implementation is complete
    // std::vector<double> tol(num_frames * 4, 1e-8);
    // opt.add_inequality_mconstraint(Optimizer::constraint, this, tol);

    const double rotation_bound = std::acos(-1.0) / 6.0;
    std::vector<double> lb(num_frames * 3, -rotation_bound);
    std::vector<double> ub(num_frames * 3, rotation_bound);

    // Prefix hard-fix: lb == ub for each axis-angle variable.
    for (size_t i = 0; i < fixed_prefix_count; ++i) {
        Eigen::Quaterniond q_orig = initial_frame_data[i].orientation;
        Eigen::Quaterniond q_target = fixed_prefix_filtered[i];
        q_orig.normalize();
        q_target.normalize();

        Eigen::Quaterniond dq_fixed = q_target * q_orig.inverse();
        dq_fixed.normalize();
        if (dq_fixed.w() < 0.0) {
            dq_fixed.coeffs() *= -1.0;
        }

        Eigen::Vector3d fixed_axis_angle = quaternionToAxisAngle(dq_fixed);
        for (size_t axis = 0; axis < 3; ++axis) {
            size_t idx = i * 3 + axis;
            x[idx] = fixed_axis_angle[axis];
            lb[idx] = fixed_axis_angle[axis];
            ub[idx] = fixed_axis_angle[axis];
        }
    }

    if (fixed_prefix_count > 0) {
        std::cout << "Optimizer: fixed prefix frames = " << fixed_prefix_count << std::endl;
    }

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_xtol_rel(1e-6);
    opt.set_ftol_rel(1e-6);
    opt.set_maxeval(5000);
    opt.set_maxtime(300.0);

    try {
        double minf = 0.0;
        nlopt::result result = opt.optimize(x, minf);
        std::cout << "NLopt finished with result " << result << " and min value " << minf << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "NLopt failed: " << e.what() << std::endl;
    }

    std::vector<Eigen::Quaterniond> filtered_quaternions;
    filtered_quaternions.reserve(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
        Eigen::Quaterniond dq = axisAngleToQuaternion(axis_angle);
        Eigen::Quaterniond q_orig = initial_frame_data[i].orientation;
        Eigen::Quaterniond q_filter = dq * q_orig;
        q_filter.normalize();
        filtered_quaternions.push_back(q_filter);
    }

    return filtered_quaternions;
}

double Optimizer::objective(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    Optimizer* optimizer = static_cast<Optimizer*>(data);

    double f0 = optimizer->computeObjectiveValue(x);

    if (!grad.empty()) {
        if (optimizer->use_analytic_gradient) {
            optimizer->computeAnalyticGradient(x, grad);
        } else {
            optimizer->computeNumericalGradient(x, grad);
        }
    }

    optimizer->iteration_count++;

    if (optimizer->iteration_count % 50 == 0) {
        std::cout << "Iteration " << optimizer->iteration_count
                  << ": Objective = " << f0 << std::endl;
    }

    return f0;
}

void Optimizer::constraint(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data) {
    (void)m;
    Optimizer* optimizer = static_cast<Optimizer*>(data);
    size_t num_frames = n / 3;

    double crop_h = optimizer->crop_factor_h;
    double crop_v = optimizer->crop_factor_v;

    std::vector<Eigen::Vector2d> target_corners;
    double left = (1.0 - crop_h) / 2.0;
    double right = 1.0 - left;
    double top = (1.0 - crop_v) / 2.0;
    double bottom = 1.0 - top;

    target_corners.push_back(Eigen::Vector2d(left * optimizer->width, top * optimizer->height));
    target_corners.push_back(Eigen::Vector2d(right * optimizer->width, top * optimizer->height));
    target_corners.push_back(Eigen::Vector2d(right * optimizer->width, bottom * optimizer->height));
    target_corners.push_back(Eigen::Vector2d(left * optimizer->width, bottom * optimizer->height));

    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        Eigen::Quaterniond dq = axisAngleToQuaternion(axis_angle);
        Eigen::Quaterniond q_orig = optimizer->initial_frame_data[i].orientation;
        Eigen::Quaterniond q_filter = dq * q_orig;

        Eigen::Matrix3d H = GeometryTools::calculateHomography(q_orig, q_filter, optimizer->K);

        std::vector<Eigen::Vector2d> transformed_grid;
        transformed_grid.reserve(optimizer->initial_frame_data[i].rcs_grid.size());
        for (const auto& p : optimizer->initial_frame_data[i].rcs_grid) {
            transformed_grid.push_back(GeometryTools::transformPoint(p, H));
        }

        for (size_t j = 0; j < 4; ++j) {
            result[i * 4 + j] = GeometryTools::signedDistanceFromPointToPolygon(target_corners[j], transformed_grid);
        }
    }

    if (grad) {
        // Gradient for constraints is not implemented.
    }
}

Eigen::Quaterniond Optimizer::axisAngleToQuaternion(const Eigen::Vector3d& axis_angle) {
    double angle = axis_angle.norm();
    if (angle < 1e-9) {
        return Eigen::Quaterniond::Identity();
    }
    Eigen::Vector3d axis = axis_angle.normalized();
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

Eigen::Vector3d Optimizer::quaternionToAxisAngle(const Eigen::Quaterniond& q) {
    Eigen::AngleAxisd angle_axis(q);
    return angle_axis.angle() * angle_axis.axis();
}

double Optimizer::computeObjectiveValue(const std::vector<double>& x) {
    // Optimization variable layout:
    // - `x` is a flat axis-angle vector with 3 parameters per frame.
    // - For frame i: (x[3*i+0], x[3*i+1], x[3*i+2]) is the axis-angle delta rotation `dq_i`.
    // - This function assumes `x.size()` is divisible by 3.
    size_t num_frames = x.size() / 3;

    // Convert axis-angle deltas to quaternions dq_i (close to identity for small axis-angle).
    std::vector<Eigen::Quaterniond> dq_quaternions(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        dq_quaternions[i] = axisAngleToQuaternion(axis_angle);
    }

    // Apply deltas to the original orientations.
    // Convention (Eigen composition): q_filt = dq * q_orig means q_orig is applied first, then dq.
    std::vector<Eigen::Quaterniond> filtered_quaternions(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Quaterniond q_orig = initial_frame_data[i].orientation;
        q_orig.normalize();
        filtered_quaternions[i] = dq_quaternions[i] * q_orig;
        filtered_quaternions[i].normalize();
    }

    // Relative rotation increments between consecutive filtered frames (a "velocity" proxy in SO(3)).
    // Left-increment convention:
    // r_i = Log( q_i * q_{i-1}^{-1} ) expressed as axis-angle (stored as a 3D vector).
    std::vector<Eigen::Vector3d> relative_axis_angles(num_frames);
    relative_axis_angles[0] = Eigen::Vector3d::Zero();
    for (size_t i = 1; i < num_frames; ++i) {
        Eigen::Quaterniond relative_q = filtered_quaternions[i] * filtered_quaternions[i - 1].inverse();
        relative_axis_angles[i] = quaternionToAxisAngle(relative_q);
    }

    double smoothness_cost = 0.0;
    double fidelity_cost = 0.0;

    // Smoothness terms:
    // - velocity:     ||r_i||^2
    // - acceleration: ||r_i - r_{i-1}||^2
    // - jerk:         ||r_i - 2 r_{i-1} + r_{i-2}||^2

    // 1st-order term (velocity proxy): defined for i >= 1.
    for (size_t i = 1; i < num_frames; ++i) {
        smoothness_cost += lambda1 * relative_axis_angles[i].squaredNorm();
    }

    // 2nd-order term (acceleration proxy): backward difference, defined for i >= 2.
    for (size_t i = 2; i < num_frames; ++i) {
        Eigen::Vector3d acc = relative_axis_angles[i] - relative_axis_angles[i - 1];
        smoothness_cost += lambda2 * acc.squaredNorm();
    }

    // 3rd-order term (jerk proxy): backward stencil, defined for i >= 3.
    for (size_t i = 3; i < num_frames; ++i) {
        Eigen::Vector3d jerk = relative_axis_angles[i] - 2 * relative_axis_angles[i - 1] + relative_axis_angles[i - 2];
        smoothness_cost += lambda3 * jerk.squaredNorm();
    }

    // Fidelity/regularization term: penalize the magnitude of each delta axis-angle directly,
    // encouraging dq_i to stay close to identity (i.e., remain close to the original trajectory).
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d dq_axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        fidelity_cost += mu * dq_axis_angle.squaredNorm();
    }

    // Total objective = smoothness + fidelity.
    return smoothness_cost + fidelity_cost;
}

void Optimizer::computeNumericalGradient(const std::vector<double>& x, std::vector<double>& grad) {
    const double h = 1e-6;
    for (size_t j = 0; j < x.size(); ++j) {
        std::vector<double> x_plus = x;
        std::vector<double> x_minus = x;
        x_plus[j] += h;
        x_minus[j] -= h;
        double f_plus = computeObjectiveValue(x_plus);
        double f_minus = computeObjectiveValue(x_minus);
        grad[j] = (f_plus - f_minus) / (2.0 * h);
    }
}

void Optimizer::computeAnalyticGradient(const std::vector<double>& x, std::vector<double>& grad) {
    const size_t num_frames = x.size() / 3;
    grad.assign(x.size(), 0.0);

    if (num_frames == 0) {
        return;
    }

    // Forward cache: x_i -> dq_i -> q_i -> q_rel_i -> r_i, and Jacobians needed for backward pass.
    std::vector<Eigen::Vector3d> axis_angles(num_frames, Eigen::Vector3d::Zero());
    std::vector<Eigen::Quaterniond> filtered_quaternions(num_frames, Eigen::Quaterniond::Identity());
    std::vector<Eigen::Vector3d> relative_axis_angles(num_frames, Eigen::Vector3d::Zero());
    std::vector<Eigen::Matrix3d> Jl_x(num_frames, Eigen::Matrix3d::Identity());
    std::vector<Eigen::Matrix3d> Jl_inv_r(num_frames, Eigen::Matrix3d::Identity());
    std::vector<Eigen::Matrix3d> R_rel(num_frames, Eigen::Matrix3d::Identity());

    for (size_t i = 0; i < num_frames; ++i) {
        axis_angles[i] = readFrameAxisAngle(x, i);
        Jl_x[i] = leftJacobianSO3(axis_angles[i]);

        Eigen::Quaterniond dq = axisAngleToQuaternion(axis_angles[i]);
        Eigen::Quaterniond q_orig = initial_frame_data[i].orientation;
        q_orig.normalize();
        filtered_quaternions[i] = dq * q_orig;
        filtered_quaternions[i].normalize();
    }

    for (size_t i = 1; i < num_frames; ++i) {
        Eigen::Quaterniond relative_q = filtered_quaternions[i] * filtered_quaternions[i - 1].inverse();
        relative_axis_angles[i] = quaternionToAxisAngle(relative_q);
        Jl_inv_r[i] = leftJacobianInverseSO3(relative_axis_angles[i]);
        R_rel[i] = relative_q.toRotationMatrix();
    }

    // Accumulate dL/dr_i from all smoothness terms.
    std::vector<Eigen::Vector3d> dL_dr(num_frames, Eigen::Vector3d::Zero());

    // Velocity term: lambda1 * ||r_i||^2, i >= 1.
    for (size_t i = 1; i < num_frames; ++i) {
        dL_dr[i] += 2.0 * lambda1 * relative_axis_angles[i];
    }

    // Acceleration term: lambda2 * ||r_i - r_{i-1}||^2, i >= 2.
    for (size_t i = 2; i < num_frames; ++i) {
        Eigen::Vector3d acc = relative_axis_angles[i] - relative_axis_angles[i - 1];
        dL_dr[i] += 2.0 * lambda2 * acc;
        dL_dr[i - 1] -= 2.0 * lambda2 * acc;
    }

    // Jerk term: lambda3 * ||r_i - 2 r_{i-1} + r_{i-2}||^2, i >= 3.
    for (size_t i = 3; i < num_frames; ++i) {
        Eigen::Vector3d jerk = relative_axis_angles[i] - 2.0 * relative_axis_angles[i - 1] + relative_axis_angles[i - 2];
        dL_dr[i] += 2.0 * lambda3 * jerk;
        dL_dr[i - 1] -= 4.0 * lambda3 * jerk;
        dL_dr[i - 2] += 2.0 * lambda3 * jerk;
    }

    // Chain rule from r_i back to x_i and x_{i-1} under left-increment convention:
    // r_i = Log(q_i * q_{i-1}^{-1}).
    for (size_t i = 1; i < num_frames; ++i) {
        const Eigen::Vector3d& g_i = dL_dr[i];

        Eigen::Matrix3d dr_dxi = Jl_inv_r[i] * Jl_x[i];
        Eigen::Vector3d grad_xi = dr_dxi.transpose() * g_i;
        addFrameGradient(grad, i, grad_xi);

        Eigen::Matrix3d dr_dxim1 = -Jl_inv_r[i] * R_rel[i] * Jl_x[i - 1];
        Eigen::Vector3d grad_xim1 = dr_dxim1.transpose() * g_i;
        addFrameGradient(grad, i - 1, grad_xim1);
    }

    // Fidelity term: mu * ||x_i||^2.
    for (size_t i = 0; i < num_frames; ++i) {
        addFrameGradient(grad, i, 2.0 * mu * axis_angles[i]);
    }

    // Optional gradient check against finite differences.
    if (gradient_check_enabled) {
        std::vector<double> numerical_grad(grad.size(), 0.0);
        computeNumericalGradient(x, numerical_grad);

        double max_abs_error = 0.0;
        double max_rel_error = 0.0;
        size_t max_abs_index = 0;
        size_t max_rel_index = 0;
        double rel_error_sum = 0.0;

        for (size_t j = 0; j < grad.size(); ++j) {
            const double abs_error = std::abs(grad[j] - numerical_grad[j]);
            const double denom = std::max(1e-12, std::max(std::abs(grad[j]), std::abs(numerical_grad[j])));
            const double rel_error = abs_error / denom;

            rel_error_sum += rel_error;
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
                max_abs_index = j;
            }
            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
                max_rel_index = j;
            }
        }

        const double mean_rel_error = rel_error_sum / static_cast<double>(grad.size());
        std::cout << "Gradient check (analytic vs numerical):"
                  << " max_abs_err=" << max_abs_error << " @idx " << max_abs_index
                  << ", max_rel_err=" << max_rel_error << " @idx " << max_rel_index
                  << ", mean_rel_err=" << mean_rel_error << std::endl;
    }
}
