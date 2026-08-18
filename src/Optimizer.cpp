#include "Optimizer.h"
#include "DataLoader.h"
#include "GeometryTools.h"
#include "ScopedTimer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// Boundary indices for 11x11 RCS grid (40 outer points vs 121 total)
static const std::vector<size_t>& getBoundaryIndices() {
    static const std::vector<size_t> idx = []() {
        std::vector<size_t> v;
        for (int j = 0; j < 11; ++j) v.push_back(j);
        for (int i = 1; i <= 10; ++i) v.push_back(i * 11 + 10);
        for (int j = 9; j >= 0; --j) v.push_back(10 * 11 + j);
        for (int i = 9; i > 0; --i) v.push_back(i * 11);
        return v;
    }();
    return idx;
}

namespace {

Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d s = Eigen::Matrix3d::Zero();
    s(0, 1) = -v.z();
    s(0, 2) = v.y();
    s(1, 0) = v.z();
    s(1, 2) = -v.x();
    s(2, 0) = -v.y();
    s(2, 1) = v.x();
    return s;
}

static Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& phi) {
    const double theta2 = phi.squaredNorm();
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d P = hat(phi);
    if (theta2 < 1e-12) {
        return I + 0.5 * P + (1.0 / 12.0) * (P * P);
    }
    const double theta = std::sqrt(theta2);
    const double coeff = 1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    return I + 0.5 * P + coeff * (P * P);
}

static Eigen::Quaterniond expMap(const Eigen::Vector3d& w) {
    double theta2 = w.squaredNorm();
    double theta = std::sqrt(theta2);
    if (theta < 1e-10) {
        return Eigen::Quaterniond(1.0, 0.5 * w.x(), 0.5 * w.y(), 0.5 * w.z()).normalized();
    }
    double ha = 0.5 * theta;
    double sha = std::sin(ha), cha = std::cos(ha);
    return Eigen::Quaterniond(cha, sha / theta * w.x(), sha / theta * w.y(), sha / theta * w.z()).normalized();
}

static Eigen::Vector3d logMap(const Eigen::Quaterniond& q) {
    Eigen::Quaterniond qn = q.normalized();
    if (qn.w() < 0.0) qn.coeffs() *= -1.0;
    double wc = std::clamp(qn.w(), -1.0, 1.0);
    double ha = std::acos(wc);
    double sha = std::sin(ha);
    if (sha < 1e-10) return Eigen::Vector3d(2.0 * qn.x(), 2.0 * qn.y(), 2.0 * qn.z());
    return 2.0 * ha / sha * Eigen::Vector3d(qn.x(), qn.y(), qn.z());
}

static void accumulateGNSingle(double weight,
                               const Eigen::Vector3d& residual,
                               const Eigen::Matrix3d& J,
                               int frame_idx, int prefix_count,
                               Eigen::Ref<Eigen::MatrixXd> H,
                               Eigen::Ref<Eigen::VectorXd> b) {
    if (weight <= 0.0 || frame_idx < prefix_count) return;
    int idx = (frame_idx - prefix_count) * 3;
    b.segment<3>(idx) += weight * J.transpose() * residual;
    H.block<3, 3>(idx, idx) += weight * J.transpose() * J;
}

} // namespace

// -----------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------

Optimizer::Optimizer(const std::vector<FrameData>& all_frames,
                     std::vector<size_t> window_indices,
                     const Eigen::Matrix3d& K,
                     int width, int height,
                     double crop_factor_h, double crop_factor_v,
                     const SolverConfig& config)
    : all_frames_(&all_frames),
      window_indices_(std::move(window_indices)),
      K(K),
      Kinv(K.inverse()),
      width(width),
      height(height),
      crop_factor_h(crop_factor_h),
      crop_factor_v(crop_factor_v),
      config_(config) {
}

// -----------------------------------------------------------------------
// Precompute boundary polygons and corner rays
// -----------------------------------------------------------------------

void Optimizer::precomputeBoundary(size_t prefix) {
    prefix_ = prefix;
    N_ = window_indices_.size();

    const auto& bidx = getBoundaryIndices();
    fixed_boundary_.resize(N_);
    for (size_t i = 0; i < N_; ++i) {
        fixed_boundary_[i].clear();
        fixed_boundary_[i].reserve(bidx.size());
        for (size_t b : bidx) {
            const Eigen::Vector2d& grid_point = (*all_frames_)[window_indices_[i]].rcs_grid[b];
            fixed_boundary_[i].push_back(grid_point);
        }
    }

    double half_w = (width - 1) / 2.0;
    double half_h = (height - 1) / 2.0;
    double lx = -half_w * (1.0 - crop_factor_h);
    double rx =  half_w * (1.0 - crop_factor_h);
    double ty = -half_h * (1.0 - crop_factor_v);
    double by =  half_h * (1.0 - crop_factor_v);
    Eigen::Vector2d corners[4] = {{lx,ty},{rx,ty},{rx,by},{lx,by}};
    for (int j = 0; j < 4; ++j)
        corner_rays_[j] = Kinv * Eigen::Vector3d(corners[j].x(), corners[j].y(), 1.0);
    focal_length_ = K(0, 0);
}

// -----------------------------------------------------------------------
// Evaluate SDF: reverse-warp corners -> fixed RCS boundary
// -----------------------------------------------------------------------

// Evaluate SDF constraints and cache values/Jacobians.
double Optimizer::evaluateSDF(const std::vector<Eigen::Quaterniond>& q_state) {
    double max_viol = -1e10;
    // Skip fixed anchor frames.
    for (size_t i = prefix_; i < N_; ++i) {
        // Relative raw/state rotation.
        const Eigen::Quaterniond q_raw = (*all_frames_)[window_indices_[i]].orientation.normalized();
        const Eigen::Quaterniond state_inv = q_state[i].inverse().normalized();
        Eigen::Matrix3d R_fs = (q_raw * state_inv).toRotationMatrix();

        for (int j = 0; j < 4; ++j) {
            // Warp one crop corner.
            Eigen::Vector3d y = R_fs * corner_rays_[j];
            double zz = y.z();
            // Avoid division by zero.
            if (std::abs(zz) < 1e-6) zz = (zz >= 0.0 ? 1e-6 : -1e-6);

            // Project to image coordinates.
            Eigen::Vector2d qp(focal_length_ * y.x() / zz, focal_length_ * y.y() / zz);

            int ci = -1;       // Closest edge.
            double tr = 0.0;   // Unclamped edge parameter.
            double s = GeometryTools::signedDistanceFromPointToPolygon(qp, fixed_boundary_[i], &ci, &tr);
            g_cached_[i][j] = s;
            max_viol = std::max(max_viol, s);

            J_cached_[i][j].setZero();
            if (ci >= 0 && std::abs(s) > 1e-8) {
                // Rebuild the closest boundary point.
                size_t a = static_cast<size_t>(ci);
                size_t b_next = (a + 1) % fixed_boundary_[i].size();
                double tc = std::max(0.0, std::min(1.0, tr));
                Eigen::Vector2d C = fixed_boundary_[i][a] + tc * (fixed_boundary_[i][b_next] - fixed_boundary_[i][a]);
                double d = (qp - C).norm();
                if (d > 1e-8) {
                    // g = s is negative inside the polygon, so the gradient of g
                    // w.r.t. the image point flips sign there.
                    double sign = (s < 0.0) ? -1.0 : 1.0;
                    Eigen::Vector2d dg_dp = sign * (qp - C) / d;

                    // Perspective projection Jacobian.
                    Eigen::Matrix<double, 2, 3> Jproj;
                    Jproj << focal_length_ / zz, 0, -focal_length_ * y.x() / (zz * zz),
                             0, focal_length_ / zz, -focal_length_ * y.y() / (zz * zz);

                    // Chain rule to the rotation increment.
                    J_cached_[i][j] = dg_dp.transpose() * Jproj * hat(y) * R_fs;
                }
            }
        }
    }
    return max_viol;
}

// -----------------------------------------------------------------------
// Build smoothness + fidelity Hessian and gradient
// -----------------------------------------------------------------------

// Accumulate Gauss-Newton normal equations (H += J^T J, b += J^T r) for the
// smoothness terms (velocity/accel/jerk) and the fidelity term.
// Note: accel/jerk use finite differences of the first-order Lie-algebra
// residual r_i to approximate 2nd/3rd derivatives; this drops O(||r||^2)
// terms (BCH, adjoint transport, curvature) and is fine for small rotations.
// Anchor frames (index < prefix) are fixed and skipped.
void Optimizer::buildSmoothnessHessian(const std::vector<Eigen::Quaterniond>& q_state,
                                        Eigen::Ref<Eigen::MatrixXd> H,
                                        Eigen::Ref<Eigen::VectorXd> b) {
    size_t N = N_;
    size_t prefix = prefix_;

    // Per-frame caches: velocity residual, right-Jacobian inverse, R_k^T.
    std::vector<Eigen::Vector3d> vel_r(N, Eigen::Vector3d::Zero());
    std::vector<Eigen::Matrix3d> JrInv(N, Eigen::Matrix3d::Identity());
    std::vector<Eigen::Matrix3d> Radj(N, Eigen::Matrix3d::Identity());

    for (size_t i = 1; i < N; ++i) {
        Eigen::Quaterniond V = (q_state[i - 1].inverse() * q_state[i]).normalized();
        vel_r[i] = logMap(V);
        JrInv[i] = rightJacobianInverseSO3(vel_r[i]);
        Radj[i] = q_state[i].inverse().toRotationMatrix();   // R_i^T
    }

    // d(vel_r[k]) / d(delta-theta_k) under a left perturbation of q_state.
    // Under q_k <- Exp(delta) q_k this equals Jr^-1(r_k) * R_k^T (adjoint of the
    // perturbed frame, not the previous one).
    auto Jvel = [&](size_t k) -> Eigen::Matrix3d {
        return JrInv[k] * Radj[k];
    };

    // Velocity: lambda1 * ||r_i||^2. r_i depends on q_i and q_{i-1} -> 2x2 = 4 blocks.
    for (size_t i = 1; i < N; ++i) {
        Eigen::Matrix3d J = Jvel(i);
        if (i >= prefix) {
            int ii = static_cast<int>(i - prefix) * 3;
            b.segment<3>(ii) += config_.lambda1 * J.transpose() * vel_r[i];
            H.block<3, 3>(ii, ii) += config_.lambda1 * J.transpose() * J;
        }
        if (i - 1 >= prefix) {
            int im = static_cast<int>(i - 1 - prefix) * 3;
            b.segment<3>(im) -= config_.lambda1 * J.transpose() * vel_r[i];
            H.block<3, 3>(im, im) += config_.lambda1 * J.transpose() * J;
        }
        if (i >= prefix && i - 1 >= prefix) {
            int ii = static_cast<int>(i - prefix) * 3;
            int im1 = static_cast<int>(i - 1 - prefix) * 3;
            H.block<3, 3>(ii, im1) -= config_.lambda1 * J.transpose() * J;
            H.block<3, 3>(im1, ii) -= config_.lambda1 * J.transpose() * J;
        }
    }

    // Acceleration: lambda2 * ||r_i - r_{i-1}||^2. Depends on 3 frames -> 3x3 = 9 blocks.
    for (size_t i = 2; i < N; ++i) {
        Eigen::Vector3d acc = vel_r[i] - vel_r[i - 1];
        Eigen::Matrix3d Ji = Jvel(i), Ji1 = Jvel(i - 1);
        // (frame index, d(acc)/d(that frame)); q_{i-1} appears in both r_i and r_{i-1}.
        struct FJ { int fi; Eigen::Matrix3d J; };
        FJ blocks[3] = {
            {static_cast<int>(i), Ji},
            {static_cast<int>(i - 1), -(Ji + Ji1)},
            {static_cast<int>(i - 2), Ji1}
        };
        for (int a = 0; a < 3; ++a) {
            if (blocks[a].fi < static_cast<int>(prefix)) continue;
            int ia = (blocks[a].fi - static_cast<int>(prefix)) * 3;
            b.segment<3>(ia) += config_.lambda2 * blocks[a].J.transpose() * acc;
            for (int bb = 0; bb < 3; ++bb) {
                if (blocks[bb].fi < static_cast<int>(prefix)) continue;
                int ib = (blocks[bb].fi - static_cast<int>(prefix)) * 3;
                H.block<3, 3>(ia, ib) += config_.lambda2 * blocks[a].J.transpose() * blocks[bb].J;
            }
        }
    }

    // Jerk: lambda3 * ||r_i - 2r_{i-1} + r_{i-2}||^2. Depends on 4 frames -> 4x4 = 16 blocks.
    for (size_t i = 3; i < N; ++i) {
        Eigen::Vector3d jerk = vel_r[i] - 2.0 * vel_r[i - 1] + vel_r[i - 2];
        Eigen::Matrix3d Ji = Jvel(i), Ji1 = Jvel(i - 1), Ji2 = Jvel(i - 2);
        struct FJ { int fi; Eigen::Matrix3d J; };
        FJ blocks[4] = {
            {static_cast<int>(i), Ji},
            {static_cast<int>(i - 1), -Ji - 2.0 * Ji1},
            {static_cast<int>(i - 2), 2.0 * Ji1 + Ji2},
            {static_cast<int>(i - 3), -Ji2}
        };
        for (int a = 0; a < 4; ++a) {
            if (blocks[a].fi < static_cast<int>(prefix)) continue;
            int ia = (blocks[a].fi - static_cast<int>(prefix)) * 3;
            b.segment<3>(ia) += config_.lambda3 * blocks[a].J.transpose() * jerk;
            for (int bb = 0; bb < 4; ++bb) {
                if (blocks[bb].fi < static_cast<int>(prefix)) continue;
                int ib = (blocks[bb].fi - static_cast<int>(prefix)) * 3;
                H.block<3, 3>(ia, ib) += config_.lambda3 * blocks[a].J.transpose() * blocks[bb].J;
            }
        }
    }

    // Fidelity: mu * ||Log(q_i^{-1} * q_target)||^2. Depends on a single frame -> 1 block.
    for (size_t i = prefix; i < N; ++i) {
        Eigen::Quaterniond target = (*all_frames_)[window_indices_[i]].orientation.normalized();
        Eigen::Quaterniond dq = q_state[i].inverse() * target;
        dq.normalize();
        Eigen::Vector3d r_f = logMap(dq);
        // Left perturbation of q_i gives d(r_f)/d(delta) = -Jr^-1(r_f) * R_target^T.
        Eigen::Matrix3d J_f = -rightJacobianInverseSO3(r_f) * target.inverse().toRotationMatrix();
        accumulateGNSingle(config_.mu, r_f, J_f, static_cast<int>(i), static_cast<int>(prefix), H, b);
    }
}

// -----------------------------------------------------------------------
// Build ALM constraint penalty
// -----------------------------------------------------------------------

void Optimizer::buildConstraintPenalty(double rho, double inv_f,
                                       const std::vector<std::vector<double>>& lambda,
                                       Eigen::Ref<Eigen::MatrixXd> H,
                                       Eigen::Ref<Eigen::VectorXd> b) {
    for (size_t i = prefix_; i < N_; ++i) {
        int act_i = static_cast<int>(i - prefix_);
        int ii = act_i * 3;
        for (int j = 0; j < 4; ++j) {
            double g = g_cached_[i][j] * inv_f;
            double lam = lambda[act_i][j];
            double mu_eff = std::max(0.0, lam + rho * g);
            if (mu_eff <= 1e-12) continue;

            Eigen::RowVector3d Jc = J_cached_[i][j] * inv_f;
            b.segment<3>(ii) += mu_eff * Jc.transpose();
            H.block<3, 3>(ii, ii) += rho * (Jc.transpose() * Jc);
        }
    }
}

// -----------------------------------------------------------------------
// Evaluate augmented objective Phi = 0.5||R||^2 (diagnostics only)
// -----------------------------------------------------------------------

double Optimizer::evaluateObjective(const std::vector<Eigen::Quaterniond>& q_state,
                                    double rho, double inv_f,
                                    const std::vector<std::vector<double>>& lambda) {
    size_t N = N_;
    size_t prefix = prefix_;

    std::vector<Eigen::Vector3d> vel_r(N, Eigen::Vector3d::Zero());
    for (size_t i = 1; i < N; ++i)
        vel_r[i] = logMap((q_state[i - 1].inverse() * q_state[i]).normalized());

    double phi = 0.0;
    // Smoothness residuals (weights are the squared-norm coefficients).
    for (size_t i = 1; i < N; ++i)
        phi += config_.lambda1 * vel_r[i].squaredNorm();
    for (size_t i = 2; i < N; ++i)
        phi += config_.lambda2 * (vel_r[i] - vel_r[i - 1]).squaredNorm();
    for (size_t i = 3; i < N; ++i)
        phi += config_.lambda3 * (vel_r[i] - 2.0 * vel_r[i - 1] + vel_r[i - 2]).squaredNorm();
    // Fidelity residuals.
    for (size_t i = prefix; i < N; ++i) {
        Eigen::Quaterniond target = (*all_frames_)[window_indices_[i]].orientation.normalized();
        Eigen::Quaterniond dq = q_state[i].inverse() * target;
        dq.normalize();
        phi += config_.mu * logMap(dq).squaredNorm();
    }
    // Constraint penalty rho * [g + lambda/rho]_+^2 (uses cached g).
    for (size_t i = prefix; i < N; ++i) {
        for (int j = 0; j < 4; ++j) {
            double a = std::max(0.0, g_cached_[i][j] * inv_f + lambda[i - prefix][j] / rho);
            phi += rho * a * a;
        }
    }
    return 0.5 * phi;
}

// -----------------------------------------------------------------------
// Verify constraints using forward-warp cross-check
// -----------------------------------------------------------------------

void Optimizer::verifyConstraints(const std::vector<Eigen::Quaterniond>& q_state) {
    int viol_count = 0;
    double max_sdf_all = -1e10, min_sdf_all = 1e10;
    const auto& boundary = getBoundaryIndices();

    double max_delta_deg = 0.0;
    const double pi = std::acos(-1.0);

    double hw2 = (width - 1) / 2.0;
    double hh2 = (height - 1) / 2.0;
    double lpx = -hw2 * (1.0 - crop_factor_h);
    double rpx =  hw2 * (1.0 - crop_factor_h);
    double tpy = -hh2 * (1.0 - crop_factor_v);
    double bpy =  hh2 * (1.0 - crop_factor_v);
    Eigen::Vector2d corners_img[4] = {{lpx,tpy},{rpx,tpy},{rpx,bpy},{lpx,bpy}};

    for (size_t i = 0; i < N_; ++i) {
        const Eigen::Quaterniond q_raw = (*all_frames_)[window_indices_[i]].orientation.normalized();
        // Forward-warp must use the inverse of evaluateSDF's R_fs = q_raw * q_state^-1,
        // i.e. q_state * q_raw^-1, so the boundary is warped the opposite way to the corners.
        Eigen::Quaterniond dq = q_state[i] * q_raw.inverse();
        dq.normalize();
        double da = 2.0 * std::acos(std::clamp(std::abs(dq.w()), -1.0, 1.0));
        if (da * 180.0 / pi > max_delta_deg) max_delta_deg = da * 180.0 / pi;
        Eigen::Matrix3d R_dq = dq.toRotationMatrix();
        Eigen::Matrix3d Ht = K * R_dq * Kinv;

        std::vector<Eigen::Vector2d> warped;
        warped.reserve(boundary.size());
        for (size_t b : boundary) {
            const Eigen::Vector2d& grid_point = (*all_frames_)[window_indices_[i]].rcs_grid[b];
            warped.push_back(GeometryTools::transformPoint(grid_point, Ht));
        }

        double max_sdf = -1e10;
        double frame_min_sdf = 1e10;
        for (int j = 0; j < 4; ++j) {
            double s = GeometryTools::signedDistanceFromPointToPolygon(corners_img[j], warped);
            if (s > max_sdf) max_sdf = s;
            if (s < frame_min_sdf) frame_min_sdf = s;
        }
        if (max_sdf > 0) viol_count++;
        if (max_sdf > max_sdf_all) max_sdf_all = max_sdf;
        if (frame_min_sdf < min_sdf_all) min_sdf_all = frame_min_sdf;
    }

    // Print full diagnostics for every frame when verbose is enabled.
    if (config_.verbose) {
        std::cout << "\t[Verify] " << N_ << " frames"
                  << " | max_sdf=" << max_sdf_all
                  << " | min_sdf=" << min_sdf_all
                  << " | violations=" << viol_count
                  << " | max_\u0394=" << max_delta_deg << "\u00b0"
                  << " | violated=" << (viol_count > 0 ? "YES" : "no") << std::endl;
    }
}

// -----------------------------------------------------------------------
// Main solver: GN + ALM
// -----------------------------------------------------------------------

std::vector<Eigen::Quaterniond> Optimizer::solveWindowGN(
    const std::vector<Eigen::Quaterniond>& fixed_prefix_filtered)
{
    PROFILE_SCOPE("solveWindowGN");

    size_t N = window_indices_.size();
    if (N == 0) return {};

    size_t prefix = std::min(fixed_prefix_filtered.size(), N);
    int dof = static_cast<int>(N - prefix) * 3;

    std::vector<Eigen::Quaterniond> q_state(N);
    for (size_t i = 0; i < N; ++i) {
        const Eigen::Quaterniond q_raw = (*all_frames_)[window_indices_[i]].orientation.normalized();
        q_state[i] = q_raw;
    }
    for (size_t i = 0; i < prefix; ++i)
        q_state[i] = fixed_prefix_filtered[i].normalized();

    precomputeBoundary(prefix);

    g_cached_.assign(N, std::vector<double>(4, 0.0));
    J_cached_.assign(N, std::vector<Eigen::RowVector3d>(4));

    std::vector<std::vector<double>> lambda(N - prefix, std::vector<double>(4, 0.0));

    double rho = config_.rho_init;
    double max_viol = evaluateSDF(q_state);

    for (int outer = 0; outer < config_.max_outer; ++outer) {
        bool active_constraints = (max_viol > 0);
        double inv_f = 1.0 / focal_length_;

        // Augmented objective at the point entering the inner loop; serves as the
        // baseline so the first inner step already has a valid dPhi.
        double phi_prev = 0.0;
        if (config_.verbose)
            phi_prev = evaluateObjective(q_state, rho, inv_f, lambda);

        bool inner_stopped = false;
        for (int inner = 0; inner < config_.max_inner; ++inner) {
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dof, dof);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(dof);

            buildSmoothnessHessian(q_state, H, b);
            buildConstraintPenalty(rho, inv_f, lambda, H, b);

            for (int k = 0; k < dof; ++k) H(k, k) += 1e-6;

            Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
            if (ldlt.info() != Eigen::Success) {
                if (config_.verbose)
                    std::cout << "\t[ALM=" << outer + 1 << " | GN=" << inner + 1 << "] stop: LDLT decomposition failed" << std::endl;
                inner_stopped = true;
                break;
            }

            Eigen::VectorXd dx = ldlt.solve(-b);
            if (!dx.allFinite()) {
                if (config_.verbose)
                    std::cout << "\t[ALM=" << outer + 1 << " | GN=" << inner + 1 << "] stop: non-finite dx (NaN/Inf)" << std::endl;
                inner_stopped = true;
                break;
            }

            // Manifold update q <- exp(d) * q for each free frame, with a per-step
            // trust region: clamp each rotation increment to 30 deg. This mutates
            // q_state from q_k to q_{k+1}; b above stays built from q_k.
            const double max_angle = std::acos(-1.0) / 6.0;
            for (size_t i = prefix; i < N; ++i) {
                int ii = static_cast<int>(i - prefix) * 3;
                Eigen::Vector3d d = dx.segment<3>(ii);
                double d_norm = d.norm();
                if (d_norm > max_angle) d *= max_angle / d_norm;
                q_state[i] = (expMap(d) * q_state[i]).normalized();
            }

            double dx_norm = dx.norm();

            // Convergence indicators (diagnostics only, never used as stop criteria).
            // Since b is not rebuilt after the update:
            //   |JtR| = ||b||                  gradient at this step's START q_k
            //   dPhi  = Phi(q_k) - Phi(q_{k+1})  this step's actual objective decrease
            //   |d|   = ||dx||                 this step's length
            // The end point q_{k+1}'s gradient appears as |JtR| on the next line;
            // only the very last step's end gradient is never printed.
            double grad_norm = b.norm();
            double phi_cur = 0.0, dphi = 0.0;
            if (config_.verbose) {
                phi_cur = evaluateObjective(q_state, rho, inv_f, lambda);
                dphi = phi_prev - phi_cur;
            }

            // Print the three convergence indicators every inner iteration (|d|, dPhi, |JtR|).
            if (config_.verbose)
                std::cout << "\t[ALM=" << outer + 1 << " | GN=" << inner + 1 << "]"
                          << " |d|=" << dx_norm << " | dPhi=" << dphi << " | |JtR|=" << grad_norm;

            if (dx_norm < 1e-6) {
                if (config_.verbose)
                    std::cout << " -> converged" << std::endl;
                inner_stopped = true;
                break;
            }
            if (config_.verbose)
                std::cout << std::endl;

            phi_prev = phi_cur;   // carry Phi(q_{k+1}) as the next step's baseline

            if (active_constraints && inner > 0)
                evaluateSDF(q_state);
        }
        if (!inner_stopped && config_.verbose)
            std::cout << "\t[ALM=" << outer + 1 << " | GN=" << config_.max_inner << "] -> reached max_inner=" << config_.max_inner << std::endl;

        max_viol = evaluateSDF(q_state);

        // KKT errors on (x^{k+1}, lambda^{k+1}), used as the outer stop criteria.
        // e_feas: feasibility ||[g]+||_inf; e_stat: ||J_r^T r + J_g^T lambda||; e_comp: ||lambda (.) g||.
        Eigen::MatrixXd Hs = Eigen::MatrixXd::Zero(dof, dof);
        Eigen::VectorXd stat = Eigen::VectorXd::Zero(dof);
        buildSmoothnessHessian(q_state, Hs, stat);   // stat starts as J_r^T r (smoothness + fidelity)
        double e_feas = 0.0, e_comp2 = 0.0;
        for (size_t i = prefix; i < N; ++i) {
            int ii = static_cast<int>(i - prefix) * 3;
            for (int j = 0; j < 4; ++j) {
                double g = g_cached_[i][j] * inv_f;
                double lam_new = std::max(0.0, lambda[i - prefix][j] + rho * g);   // lambda^{k+1}
                Eigen::RowVector3d Jc = J_cached_[i][j] * inv_f;
                stat.segment<3>(ii) += lam_new * Jc.transpose();                   // + J_g^T lambda
                e_feas = std::max(e_feas, std::max(0.0, g));
                double c = lam_new * g;
                e_comp2 += c * c;
            }
        }
        double e_stat = stat.norm();
        double e_comp = std::sqrt(e_comp2);

        if (config_.verbose) {
            std::cout << "\t[ALM=" << outer + 1 << " | max_viol=" << max_viol << " | rho=" << rho << "]" << std::endl;
            std::cout << "\t[ALM=" << outer + 1 << " KKT] e_feas=" << e_feas
                      << " | e_stat=" << e_stat << " | e_comp=" << e_comp << std::endl;
        }

        // Stop when all three KKT errors are within tolerance.
        if (e_feas < config_.eps_feas && e_stat < config_.eps_stat && e_comp < config_.eps_comp)
            break;

        for (size_t i = prefix; i < N; ++i) {
            int act_i = static_cast<int>(i - prefix);
            for (int j = 0; j < 4; ++j) {
                lambda[act_i][j] = std::max(0.0, lambda[act_i][j] + rho * g_cached_[i][j] * inv_f);
            }
        }

        rho = std::min(rho * config_.gamma, config_.rho_max);
    }

    verifyConstraints(q_state);
    return q_state;
}
