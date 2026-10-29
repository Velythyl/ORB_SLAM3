/**
 * Wheel-odometry support for the RGB-D pipeline.
 *
 * The odometry gives the pose of the robot base in an odometry frame, T_ob(t),
 * for (most) colour frames. With the fixed mount T_bc (camera optical frame ->
 * base frame) every frame carries the camera pose in the odometry frame,
 * T_oc(t) = T_ob(t) * T_bc. Only relative motions T_ci_cj = T_oc_i^-1 T_oc_j are
 * ever used, so the odometry frame never has to coincide with a map frame.
 *
 * How the odometry is used (all of it is off unless Configure() was called):
 *  - motion prior: the odometry increment replaces the constant-velocity model
 *    (Tracking::TrackWithMotionModel), and enters Optimizer::PoseOptimization
 *    as a unary prior (EdgeSE3OdomPrior);
 *  - tracking loss: the camera pose is dead-reckoned from the odometry,
 *    keyframes keep being inserted in the same map, and a map that does
 *    restart is anchored to the previous one;
 *  - optimisation: consecutive keyframes (an odometry chain kept through
 *    keyframe culling) are tied by EdgeSE3Odom in local BA and global BA, and
 *    by Sim3 edges in the loop-closing essential graph;
 *  - ego-static mask: features that stay at the same pixel and depth while the
 *    odometry says the camera moved are attached to the robot (gripper,
 *    cables) and are dropped before tracking.
 *
 * Prior art followed: ORB-SLAM3's own IMU path (prediction replacing the
 * velocity model, prediction while RECENTLY_LOST, keyframe-to-keyframe edges
 * through mPrevKF), and the wheel-odometry formulations of VINS-on-Wheels
 * (Wu et al., ICRA 2017: odometry as relative-pose constraints, soft planar
 * motion) and Zheng & Liu (ICRA 2019, SE(2)-XYZ constraints in an ORB-SLAM2
 * style system: odometry edges between keyframes in BA).
 *
 * Noise model (base frame, [omega; upsilon] as in g2o), for a base increment
 * travelling d metres and turning theta radians:
 *   sigma_x = sigma_y = transFloor + transSigma * d
 *   sigma_z           = transFloor + planarSigma * d        (soft planar)
 *   sigma_yaw         = rotFloor + rotSigma * theta + yawDrift * d
 *   sigma_roll/pitch  = rotFloor + planarSigma * d         (soft planar)
 * and is moved into the camera frame with the adjoint of T_cb.
 */

#ifndef ODOMETRYMODEL_H
#define ODOMETRYMODEL_H

#include <Eigen/Core>
#include "sophus/se3.hpp"

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace ORB_SLAM3
{

class KeyFrame;

struct OdometryParams
{
    float transSigma = 0.05f;   // translation sigma per metre travelled
    float rotSigma = 0.05f;     // yaw sigma per radian turned
    float transFloor = 0.005f;  // metres
    float rotFloor = 0.002f;    // radians
    float planarSigma = 0.01f;  // z (m/m) and roll/pitch (rad/m)
    float yawDrift = 0.005f;    // yaw sigma per metre travelled (rad/m)
    bool staticMask = true;     // learn and drop camera-fixed features
};

class OdometryModel
{
public:
    static void Configure(const Sophus::SE3f &Tbc, const OdometryParams &params);
    static bool Enabled();
    static const Sophus::SE3f &Tbc();
    static const Sophus::SE3f &Tcb();
    static const OdometryParams &Params();

    // Information of a camera-frame relative motion T_ci_cj measured by the odometry.
    static Eigen::Matrix<double,6,6> Information(const Sophus::SE3f &Tcicj);

    // Relative odometry measurement T_ci_cj between two keyframes that both carry odometry.
    static bool RelativeMeasurement(KeyFrame *pKFi, KeyFrame *pKFj, g2o::SE3Quat &Zij, Eigen::Matrix<double,6,6> &info);

    static g2o::SE3Quat ToSE3Quat(const Sophus::SE3f &T);
};

// Unary prior on a frame pose Tcw. Error is the camera-frame twist between the
// estimate and the prior.
class EdgeSE3OdomPrior : public g2o::BaseUnaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3OdomPrior() {}
    bool read(std::istream &) { return false; }
    bool write(std::ostream &) const { return false; }
    void computeError()
    {
        const g2o::VertexSE3Expmap *v = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        _error = (v->estimate() * _measurement.inverse()).log();
    }
};

// Relative odometry constraint between two keyframe poses Tcw_i (vertex 0) and
// Tcw_j (vertex 1); the measurement is T_ci_cj.
class EdgeSE3Odom : public g2o::BaseBinaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3Odom() {}
    bool read(std::istream &) { return false; }
    bool write(std::ostream &) const { return false; }
    void computeError()
    {
        const g2o::VertexSE3Expmap *vi = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const g2o::VertexSE3Expmap *vj = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        _error = (_measurement.inverse() * vi->estimate() * vj->estimate().inverse()).log();
    }
};

} // namespace ORB_SLAM3

#endif // ODOMETRYMODEL_H
