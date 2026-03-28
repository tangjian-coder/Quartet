
#include "Optimizer.h"
#include "DataLoader.h"
#include "GeometryTools.h"
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
    
    // Log parameters for debugging
    std::cout << "Optimizer initialized with:" << std::endl;
    std::cout << "  Image size: " << width << " x " << height << std::endl;
    std::cout << "  Crop factors: H=" << crop_factor_h << ", V=" << crop_factor_v << std::endl;
    std::cout << "  Camera matrix K:\n" << K << std::endl;
}

std::vector<Eigen::Quaterniond> Optimizer::optimize() {
    // The optimization variable x is a flat vector of axis-angle representations
    // for each dq quaternion. 3 variables per dq.
    // We optimize dq such that q_filter = dq * q_original
    size_t num_frames = std::min((size_t)300, initial_frame_data.size());
    std::vector<double> x(num_frames * 3);

    // Initialize x with small random noise (small axis-angle perturbations)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.05, 0.05); // ±0.05 rad (~±2.86 degrees)
    
    for (size_t i = 0; i < num_frames * 3; ++i) {
        x[i] = dist(gen);
    }

    // NLopt setup - use LD_SLSQP (Sequential Least-Squares Quadratic Programming)
    nlopt::opt opt(nlopt::LD_MMA, num_frames * 3);
    opt.set_min_objective(Optimizer::objective, this);

    // TODO: Add constraints after gradient implementation is complete
    // One constraint for each corner of the target rectangle for each frame
    // std::vector<double> tol(num_frames * 4, 1e-8);
    // opt.add_inequality_mconstraint(Optimizer::constraint, this, tol);

    // Set bounds for axis-angle variables (keep dq small)
    // Axis-angle components should be relatively small to keep dq close to identity
    std::vector<double> lb(num_frames * 3, -M_PI/6);  // -0.1 rad (~5.7 degrees)
    std::vector<double> ub(num_frames * 3, M_PI/6);   // 0.1 rad (~5.7 degrees)
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    opt.set_xtol_rel(1e-6);
    opt.set_ftol_rel(1e-6);
    opt.set_maxeval(5000);  // Increase function evaluations
    opt.set_maxtime(300.0);  // 30 seconds timeout

    try {
        double minf;
        nlopt::result result = opt.optimize(x, minf);
        std::cout << "NLopt finished with result " << result << " and min value " << minf << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "NLopt failed: " << e.what() << std::endl;
    }

    // Convert axis-angle to dq quaternions, then compute filtered orientations
    std::vector<Eigen::Quaterniond> filtered_quaternions;
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3 + 0], x[i * 3 + 1], x[i * 3 + 2]);
        Eigen::Quaterniond dq = axisAngleToQuaternion(axis_angle);
        Eigen::Quaterniond q_orig = initial_frame_data[i].orientation;
        // q_filter = dq * q_original
        Eigen::Quaterniond q_filter = dq * q_orig;
        filtered_quaternions.push_back(q_filter);
    }

    return filtered_quaternions;
}

// --- Static callback functions ---

double Optimizer::objective(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    Optimizer* optimizer = static_cast<Optimizer*>(data);
    
    // Compute objective value using member function
    double f0 = optimizer->computeObjectiveValue(x);

    // If gradient is requested, compute gradient using selected method
    if (!grad.empty()) {
        if (optimizer->use_analytic_gradient) {
            optimizer->computeAnalyticGradient(x, grad);
        } else {
            optimizer->computeNumericalGradient(x, grad);
        }
    }

    optimizer->iteration_count++;

    if (optimizer->iteration_count % 50 == 0) {
        std::cout << "Iteration " << optimizer->iteration_count <<
                   ": Objective = " << f0 << std::endl;
    }

    return f0;
}

void Optimizer::constraint(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data) {
    Optimizer* optimizer = static_cast<Optimizer*>(data);
    size_t num_frames = n / 3;

    // Calculate target rectangle based on crop factors
    // The target rectangle is centered in the image and sized according to crop factors
    double crop_h = optimizer->crop_factor_h;
    double crop_v = optimizer->crop_factor_v;
    
    // Target rectangle corners in normalized image coordinates (centered at origin)
    // After cropping, the visible area is reduced by crop factors
    std::vector<Eigen::Vector2d> target_corners;
    
    // Calculate corners in normalized image coordinates (range [0,1])
    // The visible area after cropping is centered
    double left = (1.0 - crop_h) / 2.0;
    double right = 1.0 - left;
    double top = (1.0 - crop_v) / 2.0;
    double bottom = 1.0 - top;
    
    // Convert to pixel coordinates (assuming cx=width/2, cy=height/2)
    double cx = optimizer->width / 2.0;
    double cy = optimizer->height / 2.0;
    
    target_corners.push_back(Eigen::Vector2d(left * optimizer->width, top * optimizer->height));
    target_corners.push_back(Eigen::Vector2d(right * optimizer->width, top * optimizer->height));
    target_corners.push_back(Eigen::Vector2d(right * optimizer->width, bottom * optimizer->height));
    target_corners.push_back(Eigen::Vector2d(left * optimizer->width, bottom * optimizer->height));

    // Loop through each frame
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        Eigen::Quaterniond dq = axisAngleToQuaternion(axis_angle);
        Eigen::Quaterniond q_orig = optimizer->initial_frame_data[i].orientation;
        // Compute smoothed orientation: q_filter = dq * q_orig
        Eigen::Quaterniond q_filter = dq * q_orig;

        // Calculate homography from original to smoothed orientation
        Eigen::Matrix3d H = GeometryTools::calculateHomography(q_orig, q_filter, optimizer->K);

        // Transform the RCS grid
        std::vector<Eigen::Vector2d> transformed_grid;
        for (const auto& p : optimizer->initial_frame_data[i].rcs_grid) {
            transformed_grid.push_back(GeometryTools::transformPoint(p, H));
        }

        // Calculate constraints for this frame
        for (size_t j = 0; j < 4; ++j) {
            // The constraint is that the signed distance from the target corner to the
            // transformed polygon must be <= 0.
            result[i * 4 + j] = GeometryTools::signedDistanceFromPointToPolygon(target_corners[j], transformed_grid);
        }
    }

    // Gradients are not implemented for this example
    if (grad) {
        // You would calculate gradients here if using a gradient-based optimizer
    }
}

// --- Helper functions ---

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

// ============================================================================
// Objective value computation
// ============================================================================

double Optimizer::computeObjectiveValue(const std::vector<double>& x) {
    size_t num_frames = x.size() / 3;
    
    // Convert axis-angle to dq quaternions
    std::vector<Eigen::Quaterniond> dq_quaternions(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        dq_quaternions[i] = axisAngleToQuaternion(axis_angle);
    }

    // Compute filtered orientations: q_filter = dq * q_original
    std::vector<Eigen::Quaterniond> filtered_quaternions(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        filtered_quaternions[i] = dq_quaternions[i] * initial_frame_data[i].orientation;
    }

    // Compute relative rotations between consecutive filtered quaternions
    std::vector<Eigen::Vector3d> relative_axis_angles(num_frames);
    relative_axis_angles[0] = Eigen::Vector3d::Zero();
    for (size_t i = 1; i < num_frames; ++i) {
        Eigen::Quaterniond relative_q = filtered_quaternions[i-1].inverse() * filtered_quaternions[i];
        relative_axis_angles[i] = quaternionToAxisAngle(relative_q);
    }

    double smoothness_cost = 0.0;
    double fidelity_cost = 0.0;

    // Smoothness cost
    for (size_t i = 1; i < num_frames; ++i) {
        smoothness_cost += lambda1 * relative_axis_angles[i].squaredNorm();
        if (i > 1) {
            Eigen::Vector3d acc = relative_axis_angles[i] - relative_axis_angles[i-1];
            smoothness_cost += lambda2 * acc.squaredNorm();
        }
        if (i > 1 && i < num_frames - 1) {
            Eigen::Vector3d jerk = relative_axis_angles[i+1] - 2 * relative_axis_angles[i] + relative_axis_angles[i-1];
            smoothness_cost += lambda3 * jerk.squaredNorm();
        }
    }

    // Fidelity cost
    for (size_t i = 0; i < num_frames; ++i) {
        Eigen::Vector3d dq_axis_angle(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
        fidelity_cost += mu * dq_axis_angle.squaredNorm();
    }

    return smoothness_cost + fidelity_cost;
}

// ============================================================================
// Gradient computation methods
// ============================================================================

void Optimizer::computeNumericalGradient(const std::vector<double>& x, std::vector<double>& grad) {
    const double h = 1e-6;  // finite difference step
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
    // Analytic gradient placeholder - to be implemented by user
    // For now, fall back to numerical gradient and print a warning
    std::cerr << "WARNING: Analytic gradient not yet implemented. Using numerical gradient instead." << std::endl;
    std::cerr << "         Please implement computeAnalyticGradient() for better performance." << std::endl;
    computeNumericalGradient(x, grad);
}
