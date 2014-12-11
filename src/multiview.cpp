#include "multiview.h"

#include <assert.h>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>


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


}
}