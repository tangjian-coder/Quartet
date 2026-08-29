#include "Optimizer.h"
#include "GeometryTools.h"
#include "ScopedTimer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

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

    // Coarse boundary per frame: 4 coarse edges (fine-edge endpoint lines).
    // Safe depth of corner x = min_e n_e·(x - p0_e) - bulge_e - margin.
    static constexpr double kSafeMargin = 50.0;   // px
    coarse_edges_.resize(N_);
    for (size_t i = 0; i < N_; ++i) {
        const auto& bnd = fixed_boundary_[i];   // 40 pts, 4 fine edges of 10 segs
        for (int e = 0; e < 4; ++e) {
            const Eigen::Vector2d& p0 = bnd[e * 10];
            const Eigen::Vector2d& p1 = bnd[(e + 1) * 10 % 40];
            Eigen::Vector2d dir = p1 - p0;
            // Pixel coords are y-down; for this winding the left normal points inward.
            Eigen::Vector2d n_in(-dir.y(), dir.x());
            n_in.normalize();
            // Only inward bulge needs covering (it shrinks the polygon).
            double bulge = 0.0;
            for (int k = e * 10 + 1; k < (e + 1) * 10; ++k)
                bulge = std::max(bulge, n_in.dot(bnd[k] - p0));   // +n_in = inward
            coarse_edges_[i][e] = {p0, n_in, bulge + kSafeMargin};
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
double Optimizer::evaluateSDF(const std::vector<Eigen::Quaterniond>& q_state,
                              const std::vector<std::vector<double>>& lambda, double rho) {
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

            // Coarse test against 4 coarse edges: a corner deeper than margin
            // inside all of them is constraint-inactive (penalty and gradient
            // both zero), so the fine 40-edge pass can be skipped.
            double d = 1e10;
            for (int e = 0; e < 4; ++e) {
                const auto& ce = coarse_edges_[i][e];
                d = std::min(d, ce.n_in.dot(qp - ce.p0) - ce.offset);
            }
            // Release depth must exceed lambda*f/rho (pixel pressure): the
            // penalty and lambda update both clamp to zero, so -d stays exact.
            if (d > lambda[i - prefix_][j] * focal_length_ / rho) {
                g_cached_[i][j] = -d;   // conservative bound (true viol <= -d)
                J_cached_[i][j].setZero();
                max_viol = std::max(max_viol, -d);
                continue;
            }

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
// Compute the three KKT errors used as the outer stop criteria
// -----------------------------------------------------------------------
// e_feas: primal feasibility  ||[g]+||_inf
// e_stat: stationarity        ||J_r^T r + J_g^T lambda^{k+1}||
// e_comp: complementarity     ||lambda^{k+1} (.) g||_2
// Requires g_cached_/J_cached_ to be refreshed for q_state (call evaluateSDF
// first). The smoothness/fidelity part mirrors the b-accumulation of
// buildSmoothnessHessian without building H; keep both in sync when the
// residual terms change.
void Optimizer::computeKKTErrors(const std::vector<Eigen::Quaterniond>& q_state,
                                 double rho, double inv_f,
                                 const std::vector<std::vector<double>>& lambda,
                                 double& e_feas, double& e_stat, double& e_comp) {
    size_t N = N_;
    size_t prefix = prefix_;

    // Per-frame caches (same as buildSmoothnessHessian).
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
    auto Jvel = [&](size_t k) -> Eigen::Matrix3d {
        return JrInv[k] * Radj[k];
    };

    // stat accumulates J_r^T r (smoothness + fidelity), then J_g^T lambda^{k+1}.
    Eigen::VectorXd stat = Eigen::VectorXd::Zero((N - prefix) * 3);

    // Velocity: lambda1 * ||r_i||^2.
    for (size_t i = 1; i < N; ++i) {
        Eigen::Matrix3d J = Jvel(i);
        if (i >= prefix) {
            int ii = static_cast<int>(i - prefix) * 3;
            stat.segment<3>(ii) += config_.lambda1 * J.transpose() * vel_r[i];
        }
        if (i - 1 >= prefix) {
            int im = static_cast<int>(i - 1 - prefix) * 3;
            stat.segment<3>(im) -= config_.lambda1 * J.transpose() * vel_r[i];
        }
    }

    // Acceleration: lambda2 * ||r_i - r_{i-1}||^2.
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
            stat.segment<3>(ia) += config_.lambda2 * blocks[a].J.transpose() * acc;
        }
    }

    // Jerk: lambda3 * ||r_i - 2r_{i-1} + r_{i-2}||^2.
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
            stat.segment<3>(ia) += config_.lambda3 * blocks[a].J.transpose() * jerk;
        }
    }

    // Fidelity: mu * ||Log(q_i^{-1} * q_target)||^2.
    if (config_.mu > 0.0) {
        for (size_t i = prefix; i < N; ++i) {
            Eigen::Quaterniond target = (*all_frames_)[window_indices_[i]].orientation.normalized();
            Eigen::Quaterniond dq = q_state[i].inverse() * target;
            dq.normalize();
            Eigen::Vector3d r_f = logMap(dq);
            // Left perturbation of q_i gives d(r_f)/d(delta) = -Jr^-1(r_f) * R_target^T.
            Eigen::Matrix3d J_f = -rightJacobianInverseSO3(r_f) * target.inverse().toRotationMatrix();
            int ii = static_cast<int>(i - prefix) * 3;
            stat.segment<3>(ii) += config_.mu * J_f.transpose() * r_f;
        }
    }

    // Constraint part: J_g^T lambda^{k+1} + feasibility + complementarity.
    e_feas = 0.0;
    double e_comp2 = 0.0;
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

    e_stat = stat.norm();
    e_comp = std::sqrt(e_comp2);
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

// Evaluate the augmented objective Phi = 0.5||R||^2 (smoothness + fidelity +
// constraint penalty). Used by the damped-GN acceptance test in the inner loop.
// NOTE: the constraint penalty reads g_cached_; call evaluateSDF first so the
// caches correspond to q_state.

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

    // Violations must exceed a physical tolerance: ALM converges to g ~ 0 from
    // the infeasible side, leaving sub-pixel noise between the two warp paths.
    const double violation_tol = 2.0 * config_.violation_tol_px;   // px; forward-warp acceptance

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
        if (max_sdf > violation_tol) viol_count++;
        if (max_sdf > max_sdf_all) max_sdf_all = max_sdf;
        if (frame_min_sdf < min_sdf_all) min_sdf_all = frame_min_sdf;
    }

    std::cout << "\t[Verify] " << N_ << " frames"
              << " | max_sdf=" << max_sdf_all
              << " | min_sdf=" << min_sdf_all
              << " | violations=" << viol_count
              << " | max_\u0394=" << max_delta_deg << "\u00b0"
              << " | violated=" << (viol_count > 0 ? "YES" : "no") << std::endl;
}

// -----------------------------------------------------------------------
// Main solver: GN + ALM
// -----------------------------------------------------------------------

std::vector<Eigen::Quaterniond> Optimizer::solveWindowGN(
    const std::vector<Eigen::Quaterniond>& fixed_prefix_filtered)
{
    PROFILE_SCOPE("solveWindowGN");

    // Window size, anchored prefix and free-frame DOF
    size_t N = window_indices_.size();
    if (N == 0) return {};

    size_t prefix = std::min(fixed_prefix_filtered.size(), N);
    int dof = static_cast<int>(N - prefix) * 3;

    // Initial state: raw orientations, then overwrite anchored frames
    std::vector<Eigen::Quaterniond> q_state(N);
    for (size_t i = 0; i < N; ++i) {
        const Eigen::Quaterniond q_raw = (*all_frames_)[window_indices_[i]].orientation.normalized();
        q_state[i] = q_raw;
    }
    for (size_t i = 0; i < prefix; ++i)
        q_state[i] = fixed_prefix_filtered[i].normalized();

    // Precompute the fixed crop boundary
    precomputeBoundary(prefix);

    // SDF caches, multipliers, penalty weight; max_viol = initial violation
    g_cached_.assign(N, std::vector<double>(4, 0.0));
    J_cached_.assign(N, std::vector<Eigen::RowVector3d>(4));

    std::vector<std::vector<double>> lambda(N - prefix, std::vector<double>(4, 0.0));

    double rho = config_.rho_init;
    double max_viol = evaluateSDF(q_state, lambda, rho);

    // Best-so-far snapshot: ALM is not monotone at high rho — late rounds can
    // push a feasible state back out. best_viol starts at +inf so round 1
    // always updates (the raw initial state is never a candidate).
    std::vector<Eigen::Quaterniond> q_best = q_state;
    double best_viol = 1e10;
    double best_stat = 1e10;

    for (int outer = 0; outer < config_.max_outer; ++outer) {
        double inv_f = 1.0 / focal_length_;

        bool inner_stopped = false;
        double mu_damp = 1e-6;          // GN damping, reset each outer round
        int consec_rejects = 0;       // consecutive rejected trials (bail-out guard)
        double viol_step_start = max_viol;   // violation at the start of each accepted step
        int inner = 0;                // accepted steps only
        while (inner < config_.max_inner && !inner_stopped) {
            // UNDAMPED Gauss-Newton normal equations at the current point
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dof, dof);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(dof);
            buildSmoothnessHessian(q_state, H, b);
            buildConstraintPenalty(rho, inv_f, lambda, H, b);

            const double phi_cur = evaluateObjective(q_state, rho, inv_f, lambda);
            const double grad_norm = b.norm();
            double step_norm = 0.0, dphi = 0.0;
            bool accepted = false;

            // Damped GN with Armijo acceptance on the TRUE objective; rejected
            // trials raise mu and retry (caches restored, iteration not consumed).
            while (true) {
                Eigen::MatrixXd Hd = H;
                for (int k = 0; k < dof; ++k) Hd(k, k) += mu_damp;
                Eigen::LDLT<Eigen::MatrixXd> ldlt(Hd);
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

                // Clamp per-frame increments to 30 deg
                const double max_angle = std::acos(-1.0) / 6.0;
                for (size_t i = prefix; i < N; ++i) {
                    int ii = static_cast<int>(i - prefix) * 3;
                    Eigen::Vector3d d = dx.segment<3>(ii);
                    double d_norm = d.norm();
                    if (d_norm > max_angle) d *= max_angle / d_norm;
                    dx.segment<3>(ii) = d;
                }

                // Trial point on the manifold and TRUE objective there
                std::vector<Eigen::Quaterniond> q_trial = q_state;
                for (size_t i = prefix; i < N; ++i) {
                    int ii = static_cast<int>(i - prefix) * 3;
                    q_trial[i] = (expMap(dx.segment<3>(ii)) * q_trial[i]).normalized();
                }
                const double viol_trial = evaluateSDF(q_trial, lambda, rho);   // caches <- trial
                const double phi_trial = evaluateObjective(q_trial, rho, inv_f, lambda);

                // Armijo sufficient decrease: Phi_trial <= Phi_cur + c1*grad^T d
                if (phi_trial <= phi_cur + 1e-4 * b.dot(dx)) {
                    q_state = q_trial;
                    max_viol = viol_trial;
                    step_norm = dx.norm();
                    dphi = phi_cur - phi_trial;
                    mu_damp = std::max(mu_damp / 3.0, 1e-9);   // step accepted: more aggressive
                    accepted = true;
                    break;
                }
                evaluateSDF(q_state, lambda, rho);   // reject: restore caches to the current state
                mu_damp *= 3.0;                               // shrink the step, retry
                if (++consec_rejects > 40 || mu_damp > 1e12) {
                    if (config_.verbose)
                        std::cout << "\t[ALM=" << outer + 1 << " | GN=" << inner + 1
                                  << "] stop: damped GN stuck (mu=" << mu_damp << ", |JtR|=" << grad_norm << ")" << std::endl;
                    inner_stopped = true;
                    break;
                }
            }
            if (!accepted) break;   // inner_stopped already true

            ++inner;
            consec_rejects = 0;

            // Convergence diagnostics: |JtR| = gradient at the step-start point;
            // |d| and dPhi describe the accepted step; mu = LM damping used.
            if (config_.verbose) {
                std::cout << "\t[ALM=" << outer + 1 << " | GN=" << inner << "]"
                          << " |d|=" << step_norm << " | dPhi=" << dphi << " | |JtR|=" << grad_norm;
                if (mu_damp > 1e-6)
                    std::cout << " | mu=" << mu_damp;
            }

            // Creeping-step stop: at high rho the step norm never reaches 1e-6,
            // so stop when NEITHER the objective NOR the violation still moves
            // (a tiny dPhi with viol progress must not be cut off).
            if (dphi < 1e-6 && std::max(0.0, viol_step_start) - std::max(0.0, max_viol) < 1e-3) {
                if (config_.verbose)
                    std::cout << " -> stop: dPhi" << std::endl;
                inner_stopped = true;
                break;
            }
            viol_step_start = max_viol;
            if (config_.verbose)
                std::cout << std::endl;
        }
        // Inner loop hit max_inner without converging (verbose only)
        if (!inner_stopped && inner >= config_.max_inner && config_.verbose)
            std::cout << "\t[ALM=" << outer + 1 << " | GN=" << config_.max_inner << "] -> reached max_inner=" << config_.max_inner << std::endl;

        // KKT errors on (x^{k+1}, lambda^{k+1}), used as the outer stop criteria.
        double e_feas = 0.0, e_stat = 0.0, e_comp = 0.0;
        computeKKTErrors(q_state, rho, inv_f, lambda, e_feas, e_stat, e_comp);

        // Verbose: constraint status and KKT errors
        if (config_.verbose) {
            std::cout << "\t[ALM=" << outer + 1 << " | max_viol=" << max_viol << " | rho=" << rho << "]" << std::endl;
            std::cout << "\t[ALM=" << outer + 1 << " KKT] e_feas=" << e_feas
                      << " | e_stat=" << e_stat << " | e_comp=" << e_comp << std::endl;
        }

        // Track the best iterate: smallest positive violation, then stationarity.
        {
            double viol_pos = std::max(0.0, max_viol);
            if (viol_pos < best_viol - 1e-12 ||
                (viol_pos == best_viol && e_stat < best_stat)) {
                best_viol = viol_pos;
                best_stat = e_stat;
                q_best = q_state;
            }
        }

        // Stop when feasibility + complementarity are met and viol is within
        // tolerance. e_stat is unattainable at large rho, so it is diagnostic only.
        if (e_feas < config_.eps_feas && e_comp < config_.eps_comp && max_viol <= config_.violation_tol_px) {
            if (config_.verbose)
                std::cout << "\t[ALM=" << outer + 1 << "] early stop: feasible & complementary (e_stat=" << e_stat << ")" << std::endl;
            break;
        }

        // Multiplier update: lambda <- max(0, lambda + rho g)
        for (size_t i = prefix; i < N; ++i) {
            int act_i = static_cast<int>(i - prefix);
            for (int j = 0; j < 4; ++j) {
                lambda[act_i][j] = std::max(0.0, lambda[act_i][j] + rho * g_cached_[i][j] * inv_f);
            }
        }

        // Penalty weight update (tutorial step 4).
        rho = std::min(rho * config_.gamma, config_.rho_max);
    }


    // Return the best iterate found, not necessarily the last one.
    if (best_viol < std::max(0.0, max_viol)) {
        if (config_.verbose)
            std::cout << "\t[ALM] returning best-so-far: max_viol " << max_viol << " -> " << best_viol << std::endl;
        q_state = q_best;
        max_viol = evaluateSDF(q_state, lambda, rho);   // refresh caches for the returned state
    }

    // Independent constraint verification before returning (verbose only)
    if (config_.verbose)
        verifyConstraints(q_state);
    return q_state;
}
