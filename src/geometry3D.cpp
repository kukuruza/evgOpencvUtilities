#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "geometry3D.h"

namespace cv {
namespace evg {

using namespace std;


cv::Matx44f pose (const cv::Matx33f& R, const cv::Matx31f& t)
{
    return Matx44f (R(0,0), R(0,1), R(0,2), t(0),
                    R(1,0), R(1,1), R(1,2), t(1),
                    R(2,0), R(2,1), R(2,2), t(2),
                    0,      0,      0,      1    );
}


void decomPose (const cv::Matx44f& pose, cv::Matx33f& R, cv::Matx31f& t)
{
    R = Matx33f (pose(0,0), pose(0,1), pose(0,2),
                 pose(1,0), pose(1,1), pose(1,2),
                 pose(2,0), pose(2,1), pose(2,2) );
    t = Matx31f (pose(0,3), pose(1,3), pose(2,3) );
}


cv::Matx31f decomPose2t (const cv::Matx44f& pose)
{
    return Matx31f (pose(0,3), pose(1,3), pose(2,3));
}


cv::Matx33f decomPose2R (const cv::Matx44f& pose)
{
    return Matx33f (pose(0,0), pose(0,1), pose(0,2),
                    pose(1,0), pose(1,1), pose(1,2),
                    pose(2,0), pose(2,1), pose(2,2));
}


cv::Matx44f deltaPose (const cv::Matx44f& pose1, const cv::Matx44f& pose2)
{
    Matx33f R1, R2;
    Matx31f t1, t2;
    decomPose (pose1, R1, t1);
    decomPose (pose2, R2, t2);
    return pose( R2 * R1.t(), t2 - t1 );
}


void normHomogeneous (cv::Matx31f& homogCoord)
{
   homogCoord = (homogCoord(2) == 0 ? homogCoord
                                    : homogCoord * (1.f / homogCoord(2)) );
}

// bottleneck: function needs to be relatively quick
void normHomogeneousRows (cv::Mat& homogCoords)
{
    if (!homogCoords.data) return;
    
    // float only
    assert (homogCoords.type() == CV_32F);
    // 2D homogeneous vectors (3 elements) only
    assert (homogCoords.cols == 3);
    
    float norm;
    float *ptr;
    for (int row = 0; row < homogCoords.rows; ++row)
    {
        ptr = homogCoords.ptr<float>(row);
        norm = *(ptr + 2);
        if (norm != 0)
        {
            *ptr = *ptr / norm;
            ++ptr;
            *ptr = *ptr / norm;
            ++ptr;
            *ptr   = 1.f;
        }
    }
    return;
}

void normHomography (cv::Matx33f& H)
{
    H = H * (1.f / H(2,2));
}


void warpPerspectivePoints (const Matx33f& H, const Mat& coords, Mat& warpedCoords)
{
    // if warpedCoords were not preallocated, allocate now
    if (warpedCoords.size() != coords.size())
        warpedCoords = Mat(coords.size(), CV_32F);
    
    // check for non-valid. Otherwise, can't access type()
    if (!coords.data) return;
    
    // float only, points are located as rows
    assert (coords.type() == CV_32F);
    assert (warpedCoords.type() == CV_32F);
    assert (coords.cols == 3);
    
    // normalizing (last) number of a homogenous coordinate
    float norm;
    // pointers to 1,2,3 cols of a same row of src mat
    const float *ptrSrc0, *ptrSrc1, *ptrSrc2;
    // pointer to 1st col of dst mat
    float *ptrDst;
    for (int row = 0; row < coords.rows; ++row)
    {
        // take values
        ptrSrc0 = coords.ptr<float>(row);
        ptrSrc1 = ptrSrc0 + 1;
        ptrSrc2 = ptrSrc0 + 2;
        //
        // multiply
        ptrDst = warpedCoords.ptr<float>(row);
        *ptrDst++ = *ptrSrc0 * H(0,0) + *ptrSrc1 * H(0,1) + *ptrSrc2 * H(0,2);
        *ptrDst++ = *ptrSrc0 * H(1,0) + *ptrSrc1 * H(1,1) + *ptrSrc2 * H(1,2);
        *ptrDst   = *ptrSrc0 * H(2,0) + *ptrSrc1 * H(2,1) + *ptrSrc2 * H(2,2);
        //
        // normalize on the last element
        norm = *ptrDst;
        if (norm != 0)
        {
            *ptrDst-- = 1.f;
            *ptrDst-- /= norm;
            *ptrDst   /= norm;
        }
    }
    return;
}


cv::Matx33f cameraK (const cv::Size& imageSize, const float f)
{
    return Matx33f(f, 0, imageSize.width / 2,
                   0, f, imageSize.height / 2,
                   0, 0, 1 );
}

cv::Matx33f cameraK (const cv::Size& imageSize)
{
    // f is an average between width/2 and height/2
    float f = (imageSize.width + imageSize.height) / 2 / 2;
    return evg::cameraK (imageSize, f);
}


Matx33f pose2H (const Matx33f& K0, const Matx44f& p, const Matx33f& K)
{
    // init R and t separately
    Matx33f absR (p(0,0),p(0,1),p(0,2), p(1,0),p(1,1),p(1,2), p(2,0),p(2,1),p(2,2));
    Matx31f absT (p(0,3),p(1,3),p(2,3));
    // camera cannot be on the ground (z!=0)
    assert (absT(2) != 0);
    // from absolute reference frame to relative reference frame
    Matx33f R = absR.t();
    Matx31f t = - absR.t() * absT;
    // build H
    Matx33f H = R;
    H(0,2) = t(0);
    H(1,2) = t(1);
    H(2,2) = t(2);
    H = K * H * K0.inv();
    return H;
}


// from "Multiple View Geometry in Computer Vision 2nd Edition" Hartley, Zisserman
//   (pp.326-327 in 2nd edition)
cv::Matx33f pose2H (const cv::Matx33f& _K0, const cv::Matx41f& _plane,
                    const cv::Matx44f& _pose, const cv::Matx33f& _K)
{
    // projection matrix M=[R t] from camera pose
    Matx33f R = evg::decomPose2R(_pose);
    Matx31f t = - R * evg::decomPose2t(_pose);
    
    //Matx44f poseWeird = evg::pose(R, t);
    Matx41f plane = _pose.t() * _plane;
    Matx31f n = Matx31f (plane(0), plane(1), plane(2));
    float   d = plane(3);
    // camera cannot be on the plane
    assert (d != 0.f);

    Matx33f H = R - t * n.t() * (1.f / d);
    H = _K * H * _K0.inv();
    return H;
}


// from "Parameterizing Homographies" CMU-RI-TR-06-11 S.Baker, A.Datta, T.Kanade
Matx33f deltaPose2H (const cv::Matx44f& _pose1, const cv::Matx33f& K1,
                     const cv::Matx44f& _pose2, const cv::Matx33f& K2,
                     const cv::Matx41f& n )
{
    // get rotation and translation from poses
    Matx33f _R1, _R2;
    Matx31f _t1, _t2;
    evg::decomPose (_pose1, _R1, _t1);
    evg::decomPose (_pose2, _R2, _t2);
    
    // _R & _t are from frame to map. R & t are from map to frame
    Matx33f R1 = - _R1.t();    // because R.inv() == - R.t()
    Matx31f t1 = - R1 * _t1;
    Matx33f R2 = - _R2.t();    // because R.inv() == - R.t()
    Matx31f t2 = - R2 * _t2;
    
    Matx34f P1 (R1(0,0), R1(0,1), R1(0,2), t1(0,0),
                R1(1,0), R1(1,1), R1(1,2), t1(1,0),
                R1(2,0), R1(2,1), R1(2,2), t1(2,0) );
    
    Matx34f P2 (R2(0,0), R2(0,1), R2(0,2), t2(0,0),
                R2(1,0), R2(1,1), R2(1,2), t2(1,0),
                R2(2,0), R2(2,1), R2(2,2), t2(2,0) );
    
    P1 = K1 * P1;
    P2 = K2 * P2;
    
    Matx43f pseudoInvP1 = P1.t() * (P1 * P1.t()).inv();
    Matx41f p1 (_t1(0,0), _t1(1,0), _t1(2,0), 1);
    Matx44f A = Matx<float,1,1>(n.t() * p1)(0,0) * Matx44f::eye() - p1 * n.t();
    Matx33f H = P2 * A * pseudoInvP1;
    
    return H * (1 / H(2,2));
}


// utility function for evg::H2pose()
void normalizeVectorL2 (cv::Matx31f& v)
{
    v = v * (1.f / sqrtf (v(0) * v(0) + v(1) * v(1) + v(2) * v(2)) );
}


cv::Matx44f H2pose (const cv::Matx33f& _K0, const cv::Matx33f& _H, const cv::Matx33f& _K)
{
    Matx33f R;
    Matx31f t;
    // adjust for intrinsic
    Matx33f H = _K.inv() * _H * _K0;
    // get t
    t(0) = H(0,2);
    t(1) = H(1,2);
    t(2) = H(2,2);
    // norm two rotation vectors
    Matx31f r0 (H(0,0), H(1,0), H(2,0));
    Matx31f r1 (H(0,1), H(1,1), H(2,1));
    // recover scale
    float scale = float(norm(r0) + norm(r1)) / 2.f;
    t = t * (1.f / scale);
    // orthogonalize two rotation vectors
    normalizeVectorL2(r0);
    normalizeVectorL2(r1);
    Matx31f sum = r0 + r1;
    Matx31f dif = r0 - r1;
    normalizeVectorL2(sum);
    normalizeVectorL2(dif);
    const float SqrtOf2 = sqrtf(2.f);
    r0 = (sum + dif) * (1.f / SqrtOf2);
    r1 = (sum - dif) * (1.f / SqrtOf2);
    // write r0 and r1
    R(0,0) = r0(0);
    R(1,0) = r0(1);
    R(2,0) = r0(2);
    R(0,1) = r1(0);
    R(1,1) = r1(1);
    R(2,1) = r1(2);
    // get the last one
    R(0,2) = r0(1) * r1(2) - r0(2) * r1(1);
    R(1,2) = r0(2) * r1(0) - r0(0) * r1(2);
    R(2,2) = r0(0) * r1(1) - r0(1) * r1(0);
    // get rotation of camera in world reference frame
    R = R.t();
    t = - R * t;
    return Matx44f (R(0,0), R(0,1), R(0,2), t(0),
                    R(1,0), R(1,1), R(1,2), t(1),
                    R(2,0), R(2,1), R(2,2), t(2),
                    0,      0,      0,      1    );
}



void displayHomogeneousPairs (const Mat& _coords1, const Mat& _coords2, Mat& background)
{
    // check for non-valid. Otherwise, can't access type()
    if (!_coords1.data || !_coords2.data) return;

    assert (_coords1.type() == CV_32F && _coords2.type() == CV_32F);
    assert (_coords1.cols == 3 && _coords2.cols == 3);
    assert (_coords1.rows == _coords2.rows);

    // normalize homogeneous points
    Mat coords1 = _coords1.clone();
    Mat coords2 = _coords2.clone();
    evg::normHomogeneousRows(coords1);
    evg::normHomogeneousRows(coords2);

    // display keypoints
    cvtColor(background, background, CV_RGB2GRAY);
    cvtColor(background, background, CV_GRAY2RGB);
    const Scalar color1(255,0,0), color2(0,0,255);
    const float radiusPerc = 0.006f;
    float radius = (background.rows + background.cols) / 2.f * radiusPerc;
    float lineWidth = radius / 4.f;
    for (int i = 0; i != coords1.rows; ++i)
    {
        Point2f pts1 = Point2f(coords1.at<float>(i,0), coords1.at<float>(i,1));
        Point2f pts2 = Point2f(coords2.at<float>(i,0), coords2.at<float>(i,1));
        circle (background, pts1, radius, color1, CV_FILLED);
        circle (background, pts2, radius, color2, CV_FILLED);
        line (background, pts1, pts2, Scalar(255,255,255), lineWidth);
    }
}


void drawCamera (const cv::Matx44f& pose, cv::Mat& background)
{
    // camera parameters - center at (0,0,1) looking at (0,0,0)
    Matx41f camCenter (0,0,1,1);
    Matx44f camVerteces ( 0.1f, -0.1f, -0.1f,  0.1f,
                         -0.1f, -0.1f,  0.1f,  0.1f,
                          0.8f,  0.8f,  0.8f,  0.8f,
                          1,     1,     1,     1    );
    
    // drawing parameters
    const Scalar colorCenter(255,128,128), colorVert(128,255,128);
    const float dotSizePerc = 0.01f;
    float dotSize;
    
    
    // actually pose
    Matx34f projectionOnZ = Matx34f(1,0,0,0, 0,1,0,0, 0,0,0,1) * pose;
    
    // image intrinsic
    int halfWidth  = background.size().width / 2;
    int halfHeight = background.size().height / 2;
    int halfSize = (halfWidth + halfHeight) / 2;
    Matx33f imageK (halfSize,0,halfWidth, 0,halfSize,halfHeight, 0,0,1);
    
    // projection (camera-specific)
    Matx31f projCenter = imageK * projectionOnZ * camCenter;
    projCenter = projCenter * (1 / projCenter(2));
    Matx34f projVert = imageK * projectionOnZ * camVerteces;
    for (int i = 0; i != 4; ++i)
    { projVert(0,i) /= projVert(2,i); projVert(1,i) /= projVert(2,i); }
    
    // draw
    dotSize = halfSize * dotSizePerc;
    Point2f pointCenter (projCenter(0), projCenter(1));
    circle (background, pointCenter, dotSize, colorCenter, CV_FILLED);
    for (int i = 0; i != 4; ++i)
    {
        Point2f pointVert1 (projVert(0,i), projVert(1,i));
        Point2f pointVert2 (projVert(0,(i+1)%4), projVert(1,(i+1)%4));
        circle (background, pointVert1, dotSize, colorVert, CV_FILLED);
        line (background, pointCenter, pointVert1, colorCenter);
        line (background, pointVert1, pointVert2, colorCenter);
    }
}


void projectPlanes (const std::vector<cv::Mat>& maps,
                    const std::vector<cv::Matx44f>& planes,
                    const cv::Matx44f& viewPoint,
                    cv::Mat& background)
{
    assert (maps.size() == planes.size());
    
    for (int i = 0; i != maps.size(); ++i)
    {
        Matx33f HtoFrame = evg::pose2H (evg::cameraK(maps[i].size()),
                                        planes[i].inv() * viewPoint,
                                        evg::cameraK(background.size()) );
        evg::normHomography(HtoFrame);
        Mat show = Mat::zeros(background.size(), background.type());
        assert (maps[i].type() == background.type());
        warpPerspective (maps[i], show, HtoFrame, show.size());
        background += show;
    }
}


} // namespace evg
} // namespace cv


