// Finite-difference check for the SO(3) Jacobians used by the optimizer.
// Verifies review items 1 (velocity), 2 (fidelity) and 3 (SDF interior sign)
// independently: analytic Jacobian (old formula vs fixed formula) against a
// central finite difference under the LEFT perturbation q <- Exp(delta) * q.
//
// Build & run:
//   g++ -O2 -I/usr/include/eigen3 tests/fd_check.cpp -o /tmp/fd_check && /tmp/fd_check
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

using namespace Eigen;

// ---- copied verbatim from src/Optimizer.cpp ----
Matrix3d hat(const Vector3d& v) {
    Matrix3d s = Matrix3d::Zero();
    s(0,1)=-v.z(); s(0,2)=v.y(); s(1,0)=v.z(); s(1,2)=-v.x(); s(2,0)=-v.y(); s(2,1)=v.x();
    return s;
}
Matrix3d rightJacobianInverseSO3(const Vector3d& phi) {
    const double theta2 = phi.squaredNorm();
    const Matrix3d I = Matrix3d::Identity();
    const Matrix3d P = hat(phi);
    if (theta2 < 1e-12) return I + 0.5*P + (1.0/12.0)*(P*P);
    const double theta = std::sqrt(theta2);
    const double coeff = 1.0/theta2 - (1.0+std::cos(theta))/(2.0*theta*std::sin(theta));
    return I + 0.5*P + coeff*(P*P);
}
Quaterniond expMap(const Vector3d& w) {
    double theta2 = w.squaredNorm();
    double theta = std::sqrt(theta2);
    if (theta < 1e-10) return Quaterniond(1.0, 0.5*w.x(), 0.5*w.y(), 0.5*w.z()).normalized();
    double ha = 0.5*theta, sha = std::sin(ha), cha = std::cos(ha);
    return Quaterniond(cha, sha/theta*w.x(), sha/theta*w.y(), sha/theta*w.z()).normalized();
}
Vector3d logMap(const Quaterniond& q) {
    Quaterniond qn = q.normalized();
    if (qn.w() < 0.0) qn.coeffs() *= -1.0;
    double wc = std::clamp(qn.w(), -1.0, 1.0);
    double ha = std::acos(wc), sha = std::sin(ha);
    if (sha < 1e-10) return Vector3d(2.0*qn.x(), 2.0*qn.y(), 2.0*qn.z());
    return 2.0*ha/sha*Vector3d(qn.x(), qn.y(), qn.z());
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> U(-1,1);
    auto randq = [&](double ang_scale){
        Vector3d axis(U(rng),U(rng),U(rng)); axis.normalize();
        double ang = ang_scale*U(rng);
        return Quaterniond(AngleAxisd(ang, axis)).normalized();
    };
    const double eps = 1e-6;

    // ===== item 1: velocity r = Log(q0^-1 q1), perturb q1 =====
    {
        double err_cur=0, err_fix=0;
        for (int t=0;t<500;t++){
            Quaterniond q0=randq(2.0), q1=randq(2.0);
            Vector3d r = logMap((q0.inverse()*q1).normalized());
            Matrix3d JrInv = rightJacobianInverseSO3(r);
            Matrix3d J_cur = JrInv * q0.inverse().toRotationMatrix();   // production: Jr^-1 . R0^T
            Matrix3d J_fix = JrInv * q1.inverse().toRotationMatrix();   // proposed:   Jr^-1 . R1^T
            Matrix3d J_num;
            for (int k=0;k<3;k++){
                Vector3d e=Vector3d::Zero(); e(k)=eps;
                Vector3d rp = logMap((q0.inverse()*(expMap(e)*q1)).normalized());
                Vector3d rm = logMap((q0.inverse()*(expMap(-e)*q1)).normalized());
                J_num.col(k) = (rp-rm)/(2*eps);
            }
            err_cur = std::max(err_cur, (J_cur-J_num).norm());
            err_fix = std::max(err_fix, (J_fix-J_num).norm());
        }
        std::cout << "=== item 1: velocity Jacobian (large rotation ~2rad) ===\n";
        std::cout << "  production  Jr^-1 . R0^T : max err = " << err_cur << "\n";
        std::cout << "  proposed    Jr^-1 . R1^T : max err = " << err_fix << "\n";
    }

    // ===== item 2: fidelity r_f = Log(q^-1 qt), perturb q =====
    {
        double err_cur=0, err_fix=0;
        for (int t=0;t<500;t++){
            Quaterniond q=randq(2.0), qt=randq(2.0);
            Vector3d rf = logMap((q.inverse()*qt).normalized());
            Matrix3d JrInv = rightJacobianInverseSO3(rf);
            Matrix3d J_cur = -JrInv * q.inverse().toRotationMatrix();    // production: -Jr^-1 . R^T
            Matrix3d J_fix = -JrInv * qt.inverse().toRotationMatrix();   // proposed:   -Jr^-1 . Rt^T
            Matrix3d J_num;
            for (int k=0;k<3;k++){
                Vector3d e=Vector3d::Zero(); e(k)=eps;
                Vector3d rp = logMap(((expMap(e)*q).inverse()*qt).normalized());
                Vector3d rm = logMap(((expMap(-e)*q).inverse()*qt).normalized());
                J_num.col(k) = (rp-rm)/(2*eps);
            }
            err_cur = std::max(err_cur, (J_cur-J_num).norm());
            err_fix = std::max(err_fix, (J_fix-J_num).norm());
        }
        std::cout << "=== item 2: fidelity Jacobian (large residual ~2rad) ===\n";
        std::cout << "  production  -Jr^-1 . R^T  : max err = " << err_cur << "\n";
        std::cout << "  proposed    -Jr^-1 . Rt^T : max err = " << err_fix << "\n";
    }

    // ===== control: small angle (~5deg) to show why real data hides it =====
    {
        double err_cur=0;
        for (int t=0;t<500;t++){
            Quaterniond q0=randq(0.087), q1=randq(0.087);
            Vector3d r = logMap((q0.inverse()*q1).normalized());
            Matrix3d J_cur = rightJacobianInverseSO3(r) * q0.inverse().toRotationMatrix();
            Matrix3d J_num;
            for (int k=0;k<3;k++){
                Vector3d e=Vector3d::Zero(); e(k)=eps;
                Vector3d rp = logMap((q0.inverse()*(expMap(e)*q1)).normalized());
                Vector3d rm = logMap((q0.inverse()*(expMap(-e)*q1)).normalized());
                J_num.col(k) = (rp-rm)/(2*eps);
            }
            err_cur = std::max(err_cur, (J_cur-J_num).norm());
        }
        std::cout << "=== control: velocity production formula at small angle (~5deg) ===\n";
        std::cout << "  production Jr^-1 . R0^T : max err = " << err_cur << "  (small => real data never exposed it)\n";
    }

    // ===== item 3: SDF interior sign, g = signedDistance(qp, boundary), perturb q_state =====
    {
        // Rectangle boundary; signed distance is negative inside, positive outside (matches production).
        std::vector<Vector2d> rect = {{-300,-200},{300,-200},{300,200},{-300,200}};
        auto sdfRect = [&](const Vector2d& p, Vector2d& C)->double{
            double best=1e18; Vector2d bestC=p;
            for (int i=0;i<4;i++){
                Vector2d a=rect[i], b=rect[(i+1)%4], ab=b-a, ap=p-a;
                double t=std::clamp(ap.dot(ab)/ab.squaredNorm(),0.0,1.0);
                Vector2d c=a+t*ab; double d=(p-c).norm();
                if (d<best){best=d;bestC=c;}
            }
            C=bestC;
            bool inside = (p.x()>-300&&p.x()<300&&p.y()>-200&&p.y()<200);
            return inside ? -best : best;
        };
        const double f=685.0;
        auto run = [&](const Vector3d& ray_in, const char* label){
            Quaterniond q_raw=randq(0.05), q_state=randq(0.05);
            Vector3d ray=ray_in.normalized();
            auto computeG=[&](const Quaterniond& qs)->double{
                Matrix3d R_fs=(q_raw*qs.inverse()).toRotationMatrix();
                Vector3d y=R_fs*ray;
                Vector2d qp(f*y.x()/y.z(), f*y.y()/y.z());
                Vector2d C; return sdfRect(qp,C);
            };
            Matrix3d R_fs=(q_raw*q_state.inverse()).toRotationMatrix();
            Vector3d y=R_fs*ray; double zz=y.z();
            Vector2d qp(f*y.x()/zz, f*y.y()/zz);
            Vector2d C; double s=sdfRect(qp,C); double d=(qp-C).norm();
            Matrix<double,2,3> Jproj;
            Jproj<<f/zz,0,-f*y.x()/(zz*zz), 0,f/zz,-f*y.y()/(zz*zz);
            RowVector3d J_prod=((qp-C)/d).transpose()*Jproj*hat(y)*R_fs;              // production
            RowVector3d J_fix =(((s<0?-1.0:1.0)*(qp-C)/d)).transpose()*Jproj*hat(y)*R_fs; // fix: negate inside
            RowVector3d J_num;
            for (int k=0;k<3;k++){
                Vector3d e=Vector3d::Zero(); e(k)=eps;
                double gp=computeG((expMap(e)*q_state).normalized());
                double gm=computeG((expMap(-e)*q_state).normalized());
                J_num(k)=(gp-gm)/(2*eps);
            }
            std::cout<<"  ["<<label<<"] s="<<s<<(s<0?" (INSIDE)":" (outside)")
                     <<" | production err="<<(J_prod-J_num).norm()
                     <<" | fix err="<<(J_fix-J_num).norm()<<"\n";
        };
        std::cout << "=== item 3: SDF Jacobian sign ===\n";
        run(Vector3d(0.1,0.05,1.0), "interior");   // qp ~ (68,34) inside rect
        run(Vector3d(0.6,0.4,1.0),  "exterior");   // qp ~ (411,274) outside rect
    }
    return 0;
}
