// Smoothness residual (acc/jerk) construction comparison + Jacobian FD check.
//
// Companion to jac_check.cpp (which covers the velocity/fidelity Jacobians).
// This file fixes the facts about four possible acc/jerk residual constructions
// under the LEFT perturbation convention q <- Exp(d) q:
//
//   Production (R^3 differencing of r_i = Log(q_{i-1}^{-1} q_i)):
//     acc  = r_i - r_{i-1}
//     jerk = r_i - 2 r_{i-1} + r_{i-2}
//
//   [T1] Transport identity: moving r_{i-1} into frame q_{i-1} is a no-op,
//        because Exp(-r_{i-1}) acts trivially on its own rotation axis.
//        => R^3 differencing of acc is already the transported form.
//   [T2] Counterexamples that MUST stay broken (guard rails, do not "fix"):
//        - jerk literal group form  Log(V_i V_{i-1}^{-2} V_{i-2})
//        - jerk naive transport fix (r_i - 2 r_{i-1} + Exp(-r_{i-1}) r_{i-2})
//        Both destroy the finite-difference cancellation structure (~141% err).
//   [T3] R^3 differencing error calibration on the analytic two-axis
//        precession trajectory: R^3 acc/jerk are CENTERED differences of the
//        chord velocities; measured orders O((w*dt)^4)/O((w*dt)^2) here.
//   [T4] Group-recursive construction (A_k = V_{k-1}^{-1} V_k):
//        acc = Log(A_i), jerk = Log(A_{i-1}^{-1} A_i).
//        REFERENCE / COUNTEREXAMPLE: the BCH term 1/2[r_i, r_{i-1}] between
//        neighboring chords (~|w|dt apart in direction) is O(|w|dt) and does
//        NOT cancel => the group forms are only FIRST order, i.e. WORSE than
//        the plain R^3 differences. Kept here to fix that fact; do not adopt.
//        (Also verified on real smoothed output: the two constructions differ
//        by ~1% (acc) / ~5% (jerk) RMS of the residual, <0.3% of the energy.)
//   [T5] jac_check.cpp-style Jacobian verification with ONE baseline: the
//        central finite difference of the GROUP-recursive residual
//        (q <- Exp(+-eps e_k) q, (rp-rm)/(2eps)). Three numbers per term:
//          (a) group-recursive analytic Jacobian vs group FD (derivation check)
//          (b) production R^3 Jacobian vs the SAME group FD (mismatch = d/dd of
//              the BCH term 1/2[r_i,r_{i-1}]; swept over small/large angles)
//          (c) production R^3 Jacobian vs FD of its OWN R^3 residual
//              (production self-consistency)
//        Group derivation (left perturbations):
//          V_k = q_{k-1}^{-1} q_k,  V_k <- Exp(R_{k-1}^T (d_k - d_{k-1})) V_k
//          A_i = V_{i-1}^{-1} V_i, A_i <- Exp(R(V_{i-1})^T (eps_i - eps_{i-1})) A_i
//          da = Jl^{-1}(a) * eta   for  A <- Exp(eta) A, a = Log(A)
//        where Jl^{-1}(a) = Jr^{-1}(-a) = rightJacobianInverseSO3(-a).
//
// Exit code 0 = all pass; 1 = at least one assertion failed.

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>

using namespace Eigen;

// ---- same SO(3) helpers as src/Optimizer.cpp ----
static Matrix3d hat(const Vector3d& v) {
    Matrix3d s = Matrix3d::Zero();
    s(0, 1) = -v.z(); s(0, 2) = v.y();
    s(1, 0) = v.z();  s(1, 2) = -v.x();
    s(2, 0) = -v.y(); s(2, 1) = v.x();
    return s;
}
// Jr^{-1}(phi) = I + 0.5[phi]x + c[phi]x^2  (right Jacobian INVERSE, +1/2 term).
static Matrix3d rightJacobianInverseSO3(const Vector3d& phi) {
    const double theta2 = phi.squaredNorm();
    const Matrix3d I = Matrix3d::Identity();
    const Matrix3d P = hat(phi);
    if (theta2 < 1e-12) return I + 0.5 * P + (1.0 / 12.0) * (P * P);
    const double theta = std::sqrt(theta2);
    const double coeff = 1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    return I + 0.5 * P + coeff * (P * P);
}
// Jl^{-1}(phi) = Jr^{-1}(-phi).
static Matrix3d leftJacobianInverseSO3(const Vector3d& phi) {
    return rightJacobianInverseSO3(-phi);
}
static Quaterniond expMap(const Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-10) return Quaterniond(1.0, 0.5 * w.x(), 0.5 * w.y(), 0.5 * w.z()).normalized();
    double ha = 0.5 * theta, sha = std::sin(ha), cha = std::cos(ha);
    return Quaterniond(cha, sha / theta * w.x(), sha / theta * w.y(), sha / theta * w.z()).normalized();
}
static Vector3d logMap(const Quaterniond& q) {
    Quaterniond qn = q.normalized();
    if (qn.w() < 0.0) qn.coeffs() *= -1.0;
    double wc = std::clamp(qn.w(), -1.0, 1.0);
    double ha = std::acos(wc), sha = std::sin(ha);
    if (sha < 1e-10) return Vector3d(2.0 * qn.x(), 2.0 * qn.y(), 2.0 * qn.z());
    return 2.0 * ha / sha * Vector3d(qn.x(), qn.y(), qn.z());
}

static int g_failures = 0;
static void check(bool ok, const char* what) {
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << std::endl;
    if (!ok) ++g_failures;
}

// ---- residuals under comparison ----
static Quaterniond Vof(const Quaterniond& qa, const Quaterniond& qb) {
    return (qa.inverse() * qb).normalized();
}
static Vector3d velRes(const Quaterniond& qa, const Quaterniond& qb) {
    return logMap(Vof(qa, qb));
}
// R^3 differencing (production)
static Vector3d accR3(const Quaterniond& q0, const Quaterniond& q1, const Quaterniond& q2) {
    return velRes(q1, q2) - velRes(q0, q1);
}
static Vector3d jerkR3(const Quaterniond& q0, const Quaterniond& q1,
                       const Quaterniond& q2, const Quaterniond& q3) {
    return velRes(q2, q3) - 2.0 * velRes(q1, q2) + velRes(q0, q1);
}
// Group-recursive: A = Vprev^{-1} Vcur; acc = Log(A); jerk = Log(Aprev^{-1} Acur)
static Vector3d accGroup(const Quaterniond& q0, const Quaterniond& q1, const Quaterniond& q2) {
    return logMap((Vof(q0, q1).inverse() * Vof(q1, q2)).normalized());
}
static Vector3d jerkGroup(const Quaterniond& q0, const Quaterniond& q1,
                          const Quaterniond& q2, const Quaterniond& q3) {
    Quaterniond A1 = (Vof(q0, q1).inverse() * Vof(q1, q2)).normalized();
    Quaterniond A2 = (Vof(q1, q2).inverse() * Vof(q2, q3)).normalized();
    return logMap((A1.inverse() * A2).normalized());
}
// Counterexample 1: jerk literal group form Log(V3 V2^{-2} V1)
static Vector3d jerkGroupLiteral(const Quaterniond& q0, const Quaterniond& q1,
                                 const Quaterniond& q2, const Quaterniond& q3) {
    Quaterniond V2 = Vof(q1, q2);
    Quaterniond J = (Vof(q2, q3) * V2.inverse() * V2.inverse() * Vof(q0, q1)).normalized();
    return logMap(J);
}
// Counterexample 2: jerk with naive transport fix
static Vector3d jerkTransport(const Quaterniond& q0, const Quaterniond& q1,
                              const Quaterniond& q2, const Quaterniond& q3) {
    Vector3d r3 = velRes(q2, q3), r2 = velRes(q1, q2), r1 = velRes(q0, q1);
    return r3 - 2.0 * r2 + expMap(-r2).toRotationMatrix() * r1;
}

// ---- analytic Jacobians (LEFT perturbation q <- Exp(d) q) ----
// Three objects compared in T5, jac_check.cpp style:
//   FD        : central difference of the residual (baseline, (rp-rm)/(2eps))
//   R^3 prod  : production formulas, mirrors src/Optimizer.cpp Jvel combos
//   group-rec : group-recursive formulas derived below (reference only)
//
// Production Jvel: dr_k/dd_k = Jr^{-1}(r_k) R_k^T, dr_k/dd_{k-1} = -Jvel(k).
// Since acc/jerk are LINEAR combinations of the r_k, their exact Jacobians
// are the same linear combos of the Jvel blocks:
//   acc  = r_i - r_{i-1}            -> {Ji, -(Ji+Ji1), Ji1}
//   jerk = r_i - 2r_{i-1} + r_{i-2} -> {Ji, -(Ji+2Ji1), 2Ji1+Ji2, -Ji2}
// (Exactness holds because the combination is linear; no BCH terms appear.)
static Matrix3d JvelProd(const Vector3d& r, const Quaterniond& qk) {
    return rightJacobianInverseSO3(r) * qk.inverse().toRotationMatrix();
}
static void accR3Jac(const Quaterniond& q0, const Quaterniond& q1, const Quaterniond& q2,
                     Matrix3d& J0, Matrix3d& J1, Matrix3d& J2) {
    Matrix3d Jv1 = JvelProd(velRes(q0, q1), q1);
    Matrix3d Jv2 = JvelProd(velRes(q1, q2), q2);
    J2 = Jv2;
    J1 = -(Jv2 + Jv1);
    J0 = Jv1;
}
static void jerkR3Jac(const Quaterniond& q0, const Quaterniond& q1,
                      const Quaterniond& q2, const Quaterniond& q3,
                      Matrix3d& J0, Matrix3d& J1, Matrix3d& J2, Matrix3d& J3) {
    Matrix3d Jv1 = JvelProd(velRes(q0, q1), q1);
    Matrix3d Jv2 = JvelProd(velRes(q1, q2), q2);
    Matrix3d Jv3 = JvelProd(velRes(q2, q3), q3);
    J3 = Jv3;
    J2 = -(Jv3 + 2.0 * Jv2);
    J1 = 2.0 * Jv2 + Jv1;
    J0 = -Jv1;
}

// ---- analytic Jacobians of the group-recursive residuals (LEFT perturbation) ----
// acc: a = Log(A), A = V0^{-1} V1, frames (q0,q1,q2) = (i-2, i-1, i):
//   da/dd_i   = Jl^{-1}(a) R1^T R0 R1^T
//   da/dd_1   = -Jl^{-1}(a) (R1^T R0 R1^T + R1^T)
//   da/dd_0   = Jl^{-1}(a) R1^T
// (Sanity: at identity R=I this degenerates to the R^3 blocks {I, -2I, I}.)
static void accGroupJac(const Quaterniond& q0, const Quaterniond& q1, const Quaterniond& q2,
                        Matrix3d& J0, Matrix3d& J1, Matrix3d& J2) {
    Vector3d a = accGroup(q0, q1, q2);
    Matrix3d Jl = leftJacobianInverseSO3(a);
    Matrix3d R0 = q0.toRotationMatrix(), R1 = q1.toRotationMatrix();
    J2 = Jl * (R1.transpose() * R0 * R1.transpose());
    J1 = -Jl * (R1.transpose() * R0 * R1.transpose() + R1.transpose());
    J0 = Jl * R1.transpose();
}
// jerk: j = Log(J), J = A_{i-1}^{-1} A_i, frames (q0..q3) = (i-3..i).
//   A_k left-perturbation blocks (A_k uses frames k-2,k-1,k):
//     M(k,k)   = R_{k-1}^T R_{k-2} R_{k-1}^T
//     M(k,k-1) = -(R_{k-1}^T R_{k-2} R_{k-1}^T + R_{k-1}^T)
//     M(k,k-2) = R_{k-1}^T
//   dj/dd_m = Jl^{-1}(j) R(A_{i-1})^T (M_i^m - M_{i-1}^m)
static void jerkGroupJac(const Quaterniond& q0, const Quaterniond& q1,
                         const Quaterniond& q2, const Quaterniond& q3,
                         Matrix3d& J0, Matrix3d& J1, Matrix3d& J2, Matrix3d& J3) {
    Quaterniond V0 = Vof(q0, q1), V1 = Vof(q1, q2), V2 = Vof(q2, q3);
    Quaterniond A1 = (V0.inverse() * V1).normalized();
    Quaterniond A2 = (V1.inverse() * V2).normalized();
    Vector3d j = logMap((A1.inverse() * A2).normalized());
    Matrix3d Jl = leftJacobianInverseSO3(j);
    Matrix3d RA1 = A1.toRotationMatrix();
    Matrix3d R0 = q0.toRotationMatrix(), R1 = q1.toRotationMatrix(), R2 = q2.toRotationMatrix();
    // M_i^m (A2, frames q1,q2,q3)
    Matrix3d M2_3 = R2.transpose() * R1 * R2.transpose();
    Matrix3d M2_2 = -(R2.transpose() * R1 * R2.transpose() + R2.transpose());
    Matrix3d M2_1 = R2.transpose();
    // M_{i-1}^m (A1, frames q0,q1,q2)
    Matrix3d M1_2 = R1.transpose() * R0 * R1.transpose();
    Matrix3d M1_1 = -(R1.transpose() * R0 * R1.transpose() + R1.transpose());
    Matrix3d M1_0 = R1.transpose();
    Matrix3d P = Jl * RA1.transpose();
    J3 = P * M2_3;
    J2 = P * (M2_2 - M1_2);
    J1 = P * (M2_1 - M1_1);
    J0 = P * (-M1_0);
}

// finite-difference Jacobian under left perturbation q <- Exp(d) q
using Res3Fn = std::function<Vector3d(const Quaterniond&, const Quaterniond&, const Quaterniond&)>;
static Matrix3d fdJac3(Res3Fn f, const Quaterniond& q0, const Quaterniond& q1,
                       const Quaterniond& q2, int slot, double eps = 1e-6) {
    Matrix3d J;
    for (int k = 0; k < 3; ++k) {
        Vector3d e = Vector3d::Zero(); e(k) = eps;
        Quaterniond qp[3] = {q0, q1, q2}, qm[3] = {q0, q1, q2};
        qp[slot] = (expMap(e) * qp[slot]).normalized();
        qm[slot] = (expMap(-e) * qm[slot]).normalized();
        J.col(k) = (f(qp[0], qp[1], qp[2]) - f(qm[0], qm[1], qm[2])) / (2 * eps);
    }
    return J;
}
using Res4Fn = std::function<Vector3d(const Quaterniond&, const Quaterniond&,
                                      const Quaterniond&, const Quaterniond&)>;
static Matrix3d fdJac4(Res4Fn f, const Quaterniond* q, int slot, double eps = 1e-6) {
    Matrix3d J;
    for (int k = 0; k < 3; ++k) {
        Vector3d e = Vector3d::Zero(); e(k) = eps;
        Quaterniond qp[4] = {q[0], q[1], q[2], q[3]}, qm[4] = {q[0], q[1], q[2], q[3]};
        qp[slot] = (expMap(e) * qp[slot]).normalized();
        qm[slot] = (expMap(-e) * qm[slot]).normalized();
        J.col(k) = (f(qp[0], qp[1], qp[2], qp[3]) - f(qm[0], qm[1], qm[2], qm[3])) / (2 * eps);
    }
    return J;
}

int main() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> U(-1, 1);
    auto randq = [&](double ang_scale) {
        Vector3d axis(U(rng), U(rng), U(rng)); axis.normalize();
        return Quaterniond(AngleAxisd(ang_scale * U(rng), axis)).normalized();
    };

    // ===== T1: transport identity Exp(-r) r == r =====
    {
        double err = 0;
        for (int t = 0; t < 2000; ++t) {
            Vector3d r1 = logMap(randq(1.0));
            err = std::max(err, (expMap(-r1).toRotationMatrix() * r1 - r1).norm());
        }
        std::cout << "=== T1: transport identity Exp(-r)*r == r ===" << std::endl;
        std::cout << "  max |Exp(-r) r - r| = " << err << std::endl;
        check(err < 1e-12, "T1: adjoint transport of r by Exp(-r) is identity");
    }

    // ===== analytic two-axis precession trajectory =====
    //   q(t) = Exp(Om1 t z) Exp(Om2 t y)
    //   body w(t) = Om2 y + Om1 Exp(-Om2 t y) z
    //        a(t) = -Om1 Om2 Exp(-Om2 t y) x
    //        j(t) = -Om1 Om2^2 Exp(-Om2 t y) z
    // NOTE: return type must be explicit Vector3d; a bare Eigen expression
    // returned from a lambda dangles (references dead temporaries).
    const double dt = 1.0 / 30.0;
    const Vector3d ex(1, 0, 0), ey(0, 1, 0), ez(0, 0, 1);
    auto precessQ = [&](double O1, double O2, double t) {
        return (expMap(O1 * t * ez) * expMap(O2 * t * ey)).normalized();
    };
    auto precessA = [&](double O1, double O2, double t) -> Vector3d {
        return -O1 * O2 * (expMap(-O2 * t * ey).toRotationMatrix() * ex);
    };
    auto precessJ = [&](double O1, double O2, double t) -> Vector3d {
        return -O1 * O2 * O2 * (expMap(-O2 * t * ey).toRotationMatrix() * ez);
    };

    // ===== T2: counterexamples must stay broken =====
    {
        std::cout << "=== T2: counterexamples (literal/transport jerk) ===" << std::endl;
        double Om1 = 4.0, Om2 = 4.0, t0 = 1.0;
        Quaterniond q[5];
        for (int k = 0; k < 5; ++k) q[k] = precessQ(Om1, Om2, t0 + k * dt);
        // residual over frames 0..3, centered at t0 + 1.5 dt
        Vector3d jt = precessJ(Om1, Om2, t0 + 1.5 * dt);
        Vector3d j_lit = jerkGroupLiteral(q[0], q[1], q[2], q[3]) / (dt * dt * dt);
        Vector3d j_tr  = jerkTransport(q[0], q[1], q[2], q[3]) / (dt * dt * dt);
        double e_lit = (j_lit - jt).norm() / jt.norm();
        double e_tr  = (j_tr - jt).norm() / jt.norm();
        std::cout << "  literal group jerk err = " << e_lit * 100 << "%" << std::endl;
        std::cout << "  transport-fixed jerk err = " << e_tr * 100 << "%" << std::endl;
        check(e_lit > 0.5, "T2a: literal group jerk is structurally broken (>50%)");
        check(e_tr > 0.5, "T2b: naive transport-fixed jerk is structurally broken (>50%)");
    }

    // ===== T3/T4: error calibration on the analytic trajectory =====
    {
        std::cout << "=== T3/T4: error calibration, two-axis precession ===" << std::endl;
        double worst_r3_a = 0, worst_r3_j = 0, worst_g_a = 0, worst_g_j = 0;
        for (double Om : {1.0, 2.0, 4.0, 8.0}) {
            double t0 = 1.0;
            Quaterniond q[7];
            for (int k = -2; k < 5; ++k) q[k + 2] = precessQ(Om, Om, t0 + k * dt);
            // acc(q2,q3,q4): velocity chords [t0,t0+dt],[t0+dt,t0+2dt] -> center t0+1.0dt
            // jerk(q2..q5): center t0+1.5dt
            Vector3d at = precessA(Om, Om, t0 + 1.0 * dt);
            Vector3d jt = precessJ(Om, Om, t0 + 1.5 * dt);
            double wdt = (Om * ey + Om * ez).norm() * dt;
            double e_r3_a = (accR3(q[2], q[3], q[4]) / (dt * dt) - at).norm() / at.norm();
            double e_g_a  = (accGroup(q[2], q[3], q[4]) / (dt * dt) - at).norm() / at.norm();
            double e_r3_j = (jerkR3(q[2], q[3], q[4], q[5]) / (dt * dt * dt) - jt).norm() / jt.norm();
            double e_g_j  = (jerkGroup(q[2], q[3], q[4], q[5]) / (dt * dt * dt) - jt).norm() / jt.norm();
            std::cout << "  Om=" << Om << " (" << wdt * 57.3 << " deg/frame):"
                      << "  acc R3=" << e_r3_a * 100 << "% grp=" << e_g_a * 100 << "%"
                      << "  jerk R3=" << e_r3_j * 100 << "% grp=" << e_g_j * 100 << "%" << std::endl;
            worst_r3_a = std::max(worst_r3_a, e_r3_a / wdt);
            worst_g_a  = std::max(worst_g_a,  e_g_a / wdt);
            worst_r3_j = std::max(worst_r3_j, e_r3_j / wdt);
            worst_g_j  = std::max(worst_g_j,  e_g_j / wdt);
        }
        std::cout << "  error coefficients (rel_err / |w|dt): acc R3=" << worst_r3_a
                  << " grp=" << worst_g_a << " | jerk R3=" << worst_r3_j
                  << " grp=" << worst_g_j << std::endl;
        check(worst_r3_a < 2.0 && worst_r3_j < 2.0, "T3: R^3 differencing error < 2*|w|*dt");
        check(worst_g_a < 2.0 && worst_g_j < 2.0, "T4: group-recursive error < 2*|w|*dt");
        // The group form carries a non-cancelling O(|w|dt) BCH term and must
        // stay strictly worse than the plain R^3 differences on this trajectory.
        check(worst_g_a > worst_r3_a && worst_g_j > worst_r3_j,
              "T4b: group-recursive form is worse than R^3 differencing (BCH term)");
    }

    // ===== T5: Jacobian check, ONE baseline: central FD of the GROUP residual =====
    // Perturbation: q <- Exp(+-eps*e_k) q (left), eps=1e-6, central (rp-rm)/(2eps).
    // Three numbers per term:
    //   (a) group-recursive analytic Jacobian vs group FD  -> validates the
    //       Lie-group derivation (must be ~1e-10).
    //   (b) production R^3 Jacobian vs the SAME group FD -> mismatch of the
    //       production formula measured on the group path (= d/dd of the BCH
    //       term 1/2[r_i,r_{i-1}], i.e. O(|r|)). No pass/fail: this compares
    //       the Jacobians of two DIFFERENT residual functions.
    //   (c) production R^3 Jacobian vs FD of its OWN R^3 residual -> production
    //       self-consistency (must be ~1e-10).
    {
        std::cout << "=== T5: Jacobian check, baseline = central FD of group residual ===" << std::endl;
        double a_grp = 0, a_prod_vs_grp = 0, a_prod_self = 0;
        double j_grp = 0, j_prod_vs_grp = 0, j_prod_self = 0;
        for (int t = 0; t < 300; ++t) {
            Quaterniond q[4] = {randq(0.5), randq(0.5), randq(0.5), randq(0.5)};
            Matrix3d A0, A1, A2, A3, G0, G1, G2, G3;
            accGroupJac(q[0], q[1], q[2], G0, G1, G2);
            accR3Jac(q[0], q[1], q[2], A0, A1, A2);
            Matrix3d AG[3] = {G0, G1, G2}, AA[3] = {A0, A1, A2};
            for (int s = 0; s < 3; ++s) {
                Matrix3d FDg = fdJac3(accGroup, q[0], q[1], q[2], s);  // group baseline
                Matrix3d FDr = fdJac3(accR3,    q[0], q[1], q[2], s);  // own-residual FD
                a_grp         = std::max(a_grp,         (AG[s] - FDg).norm());
                a_prod_vs_grp = std::max(a_prod_vs_grp, (AA[s] - FDg).norm());
                a_prod_self   = std::max(a_prod_self,   (AA[s] - FDr).norm());
            }
            jerkGroupJac(q[0], q[1], q[2], q[3], G0, G1, G2, G3);
            jerkR3Jac(q[0], q[1], q[2], q[3], A0, A1, A2, A3);
            Matrix3d JG[4] = {G0, G1, G2, G3}, JA[4] = {A0, A1, A2, A3};
            for (int s = 0; s < 4; ++s) {
                Matrix3d FDg = fdJac4(jerkGroup, q, s);
                Matrix3d FDr = fdJac4(jerkR3,    q, s);
                j_grp         = std::max(j_grp,         (JG[s] - FDg).norm());
                j_prod_vs_grp = std::max(j_prod_vs_grp, (JA[s] - FDg).norm());
                j_prod_self   = std::max(j_prod_self,   (JA[s] - FDr).norm());
            }
        }
        std::cout << "  acc : (a) group-analytic vs group-FD = " << a_grp
                  << "  (b) production vs SAME group-FD = " << a_prod_vs_grp
                  << "  (c) production vs own R3-FD = " << a_prod_self << std::endl;
        std::cout << "  jerk: (a) group-analytic vs group-FD = " << j_grp
                  << "  (b) production vs SAME group-FD = " << j_prod_vs_grp
                  << "  (c) production vs own R3-FD = " << j_prod_self << std::endl;
        check(a_grp < 1e-6, "T5a: group-recursive acc Jacobian matches group FD");
        check(j_grp < 1e-6, "T5b: group-recursive jerk Jacobian matches group FD");
        check(a_prod_self < 1e-6, "T5c: production R3 acc Jacobian matches its own R3 FD");
        check(j_prod_self < 1e-6, "T5d: production R3 jerk Jacobian matches its own R3 FD");

        // Mismatch (b) is the derivative of the BCH term 1/2[r_i,r_{i-1}]; it
        // must scale LINEARLY with the inter-frame angle (O(|r|)). Sweep small
        // and large angles: tiny angle -> production Jacobian ~ group Jacobian.
        std::cout << "  --- (b) production-vs-group mismatch vs inter-frame angle ---" << std::endl;
        for (double ang : {0.01, 0.1, 0.5, 1.0}) {
            double ma = 0, mj = 0;
            for (int t = 0; t < 100; ++t) {
                Quaterniond q[4] = {randq(ang), randq(ang), randq(ang), randq(ang)};
                Matrix3d A0, A1, A2, A3;
                accR3Jac(q[0], q[1], q[2], A0, A1, A2);
                Matrix3d AA[3] = {A0, A1, A2};
                for (int s = 0; s < 3; ++s)
                    ma = std::max(ma, (AA[s] - fdJac3(accGroup, q[0], q[1], q[2], s)).norm());
                jerkR3Jac(q[0], q[1], q[2], q[3], A0, A1, A2, A3);
                Matrix3d JA[4] = {A0, A1, A2, A3};
                for (int s = 0; s < 4; ++s)
                    mj = std::max(mj, (JA[s] - fdJac4(jerkGroup, q, s)).norm());
            }
            std::cout << "    angle~" << ang << " rad: acc mismatch = " << ma
                      << "  jerk mismatch = " << mj << std::endl;
        }
    }

    std::cout << (g_failures == 0 ? "\nALL PASS" : "\nFAILURES PRESENT") << std::endl;
    return g_failures == 0 ? 0 : 1;
}
