#include "multiview.h"

#include <assert.h>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

#include "geometry3D.h"


namespace cv {
namespace evg {

using namespace std;
using namespace boost;


inline Matx31f cross (const Matx31f& a, const Matx31f& b)
{
    return Matx31f (a(1)*b(2)-a(2)*b(1),
                    a(2)*b(0)-a(0)*b(2),
                    a(0)*b(1)-a(1)*b(0));
}

inline float dot (const Matx31f& a, const Matx31f& b)
{
    return a.dot(b);
}



// calculate essential matrix
optional<Matx33f> findEssentialMat(const Mat& p1, const Mat& p2, float FundEstErrCutoff, float SRatioCutoff)
{
    // using normalized points, hence E directly
    Matx33f E = Matx33f(findFundamentalMat(p1, p2, CV_FM_RANSAC, FundEstErrCutoff));
    Matx31f S;
    Matx33f U, V;
    SVD::compute(E, S, U, V);

    if (std::abs(S(3)) > 1e-4)
        cout << "S: \n" << S << endl;
    assert (std::abs(S(3)) < 1e-4);  // since rank(F)==2 from findFundamentalMat
    float sRatio = min(S(0), S(1)) / max(S(0), S(1));
    if (sRatio < SRatioCutoff)
        return none;

    // rectify E
    float meanS = (S(0) + S(1)) / 2;
    Matx33f diagS(meanS, 0, 0, 0, meanS, 0, 0, 0, 0);
    E = U * diagS * V;

    // norm scale
    E = E * (1 / norm(E));

    return E;
}


cv::Matx44f E2pose (const cv::Matx33f& E12)
{
    // get epipoles in the 2nd camera for the 1st camera
    Matx31f S;
    Matx33f U, Vt;
    SVD::compute(E12.t(), S, U, Vt);
    assert (std::abs(S(2)) < 0.01);
    Matx31f epipole1in2 = Vt.row(2).t();
    assert (std::abs(sum(E12.t() * epipole1in2)[0]) < 0.01);
    
    // get R and t from E and epipoles
    Matx33f R = skewsym(epipole1in2) * E12.t();
    Matx31f t = -R * epipole1in2;
    
    // pose of the 2nd camera in the frame of the 1st camera
    Matx44f pose2 = pose(R, t);
    return pose2;
}


cv::Matx33f pose2E (const cv::Matx44f& pose)
{
    Matx33f R = decomPose2R(pose);
    Matx31f t = decomPose2t(pose);
    return skewsym(t) * R;
}


float epipolarError (const Matx44f& pose2, const Matx31f& p1, const Matx31f& p2)
{
    cv::Matx<float, 1, 1> error = p2.t() * pose2E(pose2) * p1;
    return std::abs(error(0));
}


float twoViewError (const Matx44f& pose2, const Matx31f& p1, const Matx31f& p2)
{
    Matx33f R2 = decomPose2R(pose2);
    Matx31f t2 = decomPose2t(pose2);

    Matx31f q1 = p1;
    Matx31f q2 = R2.t() * p2;
    return std::abs(dot (q1, cross(t2, q2)));
}


vector<float> threeViewErrors (const Matx44f& pose2, const Matx44f& pose3,
                               const Matx31f& p1, const Matx31f& p2, const Matx31f& p3)
{
    Matx33f R2 = decomPose2R(pose2);
    Matx31f t2 = decomPose2t(pose2);
    Matx33f R3 = decomPose2R(pose3);
    Matx31f t3 = decomPose2t(pose3);

    Matx31f q1 = p1;
    Matx31f q2 = R2.t() * p2;
    Matx31f q3 = R3.t() * p3;
    
    Matx31f t1to2 = t2;
    Matx31f t2to3 = t3 - t2;
    
    //TODO: case of three camera centers on one line. Then the return vector will have 4 elements
    
    float g1 = dot (q1, cross(t1to2, q2));
    float g2 = dot (q2, cross(t2to3, q3));
    float g3 = dot (cross(q2, q1), cross(q3, t2to3))
             - dot (cross(q1, t1to2), cross(q3, q2));
    
    Mat result (Matx31f(std::abs(g1), std::abs(g2), std::abs(g3)));
    return vector<float> (result.ptr<float>(0), result.ptr<float>(3));
}








}
}