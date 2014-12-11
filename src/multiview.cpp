#include "multiview.h"

#include <assert.h>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

#include "geometry3D.h"


namespace cv {
namespace evg {

using namespace std;
using namespace boost;

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


}
}