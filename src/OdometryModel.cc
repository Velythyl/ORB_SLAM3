#include "OdometryModel.h"

#include <Eigen/Dense>

#include "KeyFrame.h"

namespace ORB_SLAM3
{

namespace
{
struct OdometryState
{
    bool enabled = false;
    Sophus::SE3f Tbc;
    Sophus::SE3f Tcb;
    OdometryParams params;
};

OdometryState &State()
{
    static OdometryState state;
    return state;
}
} // namespace

void OdometryModel::Configure(const Sophus::SE3f &Tbc, const OdometryParams &params)
{
    OdometryState &state = State();
    state.Tbc = Tbc;
    state.Tcb = Tbc.inverse();
    state.params = params;
    state.enabled = true;
}

bool OdometryModel::Enabled()
{
    return State().enabled;
}

const Sophus::SE3f &OdometryModel::Tbc()
{
    return State().Tbc;
}

const Sophus::SE3f &OdometryModel::Tcb()
{
    return State().Tcb;
}

const OdometryParams &OdometryModel::Params()
{
    return State().params;
}

Eigen::Matrix<double,6,6> OdometryModel::Information(const Sophus::SE3f &Tcicj)
{
    const OdometryParams &p = Params();
    const Sophus::SE3d Tbibj = (Tbc() * Tcicj * Tcb()).cast<double>();
    const double d = Tbibj.translation().norm();
    const double theta = Tbibj.so3().log().norm();

    const double sRollPitch = p.rotFloor + p.planarSigma * d;
    const double sYaw = p.rotFloor + p.rotSigma * theta + p.yawDrift * d;
    const double sXY = p.transFloor + p.transSigma * d;
    const double sZ = p.transFloor + p.planarSigma * d;

    Eigen::Matrix<double,6,1> sigma;
    sigma << sRollPitch, sRollPitch, sYaw, sXY, sXY, sZ;
    const Eigen::Matrix<double,6,6> covBase = sigma.cwiseAbs2().asDiagonal();

    // Twists in [omega; upsilon] order move from the base to the camera frame
    // with the adjoint of T_cb.
    const Sophus::SE3d Tcb_ = Tcb().cast<double>();
    const Eigen::Matrix3d R = Tcb_.rotationMatrix();
    Eigen::Matrix<double,6,6> Ad = Eigen::Matrix<double,6,6>::Zero();
    Ad.block<3,3>(0,0) = R;
    Ad.block<3,3>(3,3) = R;
    Ad.block<3,3>(3,0) = Sophus::SO3d::hat(Tcb_.translation()) * R;

    const Eigen::Matrix<double,6,6> covCam = Ad * covBase * Ad.transpose();
    Eigen::Matrix<double,6,6> info = covCam.inverse();
    info = 0.5 * (info + info.transpose());
    return info;
}

bool OdometryModel::RelativeMeasurement(KeyFrame *pKFi, KeyFrame *pKFj, g2o::SE3Quat &Zij, Eigen::Matrix<double,6,6> &info)
{
    if(!pKFi || !pKFj || !pKFi->mbHasOdom || !pKFj->mbHasOdom)
        return false;
    const Sophus::SE3f Tcicj = pKFi->mToc.inverse() * pKFj->mToc;
    Zij = ToSE3Quat(Tcicj);
    info = Information(Tcicj);
    return true;
}

g2o::SE3Quat OdometryModel::ToSE3Quat(const Sophus::SE3f &T)
{
    return g2o::SE3Quat(T.unit_quaternion().cast<double>(), T.translation().cast<double>());
}

} // namespace ORB_SLAM3
