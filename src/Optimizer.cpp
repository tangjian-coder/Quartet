#include "Optimizer.h"
#include "DataLoader.h"
#include "GeometryTools.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

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
      use_analytic_gradient(false) {
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
    size_t num_frames = x.size() / 3;

    std::vector<Eigen::Quaterniond> dq_quaternions(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        dq_quaternions[i] = axisAngleToQuaternion(axis_angle);
    }

    std::vector<Eigen::Quaterniond> filtered_quaternions(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        filtered_quaternions[i] = dq_quaternions[i] * initial_frame_data[i].orientation;
    }

    std::vector<Eigen::Vector3d> relative_axis_angles(num_frames);
    relative_axis_angles[0] = Eigen::Vector3d::Zero();
    for (size_t i = 1; i < num_frames; ++i) {
        Eigen::Quaterniond relative_q = filtered_quaternions[i - 1].inverse() * filtered_quaternions[i];
        relative_axis_angles[i] = quaternionToAxisAngle(relative_q);
    }

    double smoothness_cost = 0.0;
    double fidelity_cost = 0.0;

    for (size_t i = 1; i < num_frames; ++i) {
        smoothness_cost += lambda1 * relative_axis_angles[i].squaredNorm();
        if (i > 1) {
            Eigen::Vector3d acc = relative_axis_angles[i] - relative_axis_angles[i - 1];
            smoothness_cost += lambda2 * acc.squaredNorm();
        }
        if (i > 1 && i < num_frames - 1) {
            Eigen::Vector3d jerk = relative_axis_angles[i + 1] - 2 * relative_axis_angles[i] + relative_axis_angles[i - 1];
            smoothness_cost += lambda3 * jerk.squaredNorm();
        }
    }

    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d dq_axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        fidelity_cost += mu * dq_axis_angle.squaredNorm();
    }

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
    std::cerr << "WARNING: Analytic gradient not yet implemented. Using numerical gradient instead." << std::endl;
    std::cerr << "         Please implement computeAnalyticGradient() for better performance." << std::endl;
    computeNumericalGradient(x, grad);
}
