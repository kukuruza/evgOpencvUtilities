#ifndef EVG_GEOMETRY_3D
#define EVG_GEOMETRY_3D

#include <opencv2/core/core.hpp>

//
// Evgeny Toropov, 2012
// Some functions for simple 3D geometry using Opencv
//
// Topics:
//   1. 3D pose vs. rotation + translation
//   2. homogeneous coordinates normalization, projection
//   3. composing intrinsic matrix
//      All of those are used for the next sections
//   4. pose to homography and back
//   5. some displaying functions
//
// Dependences:
//   opencv_core
//
// Notes:
//   everything is wrapped in namespace 'cv::evg' for convenience 
//   everything is float
//   error policy: exceptions are promoted, asserts are used
//   links to algorithms are given in the .cpp file
//
// Common Notations:
//   t:     camera translation is [3 x 1] vector of type      cv::Matx31f
//   R:     camera rotation matrix is [3 x 3] matrix of type  cv::Matx33f
//   pose:  camera pose is [4 x 4] matrix of type             cv::Matx44f
//   K:     camera intrinsic is [3 x 3] matrix of type        cv::Matx33f
//   H:     planar homography H is [3 x 3] matrix of type     cv::Matx33f
//   plane: [n, t] is [4 x 1] vector of type                  cv::Matx41f
//

namespace cv {
namespace evg {




//
// pose vs. rotation and translation
//

// pose from R and t
cv::Matx44f     pose (const cv::Matx33f& R = cv::Matx33f::eye(),
                      const cv::Matx31f& t = cv::Matx31f(0,0,0) );

// pose to R and t, scale of 't' is left unknown
void            decomPose (const cv::Matx44f& pose, cv::Matx33f& R, cv::Matx31f& t);
cv::Matx31f     decomPose2t (const cv::Matx44f& pose);
cv::Matx33f     decomPose2R (const cv::Matx44f& pose);

// difference between poses
cv::Matx44f     deltaPose (const cv::Matx44f& pose1, const cv::Matx44f& pose2);




//
// normalize and project homogeneous stuff
//

// normalize 2D homogeneous vectors on the last element
void            normHomogeneous (cv::Matx31f& homogCoord);

// normalize a number of 2D homogeneous vectors, each row is a vector
void            normHomogeneousRows (cv::Mat& homogCoords);

// normalize a planar homography on the last (9th) element
void            normHomography (cv::Matx33f& Homography);

// applies a homography to a set of homogeneous coordinates Mat[n x 3]
//   you can preallocate warpedCoords for speed
void            warpPerspectivePoints (const cv::Matx33f& H,
                                       const cv::Mat& coords, cv::Mat& warpedCoords);
    
    

//
// compose camera intrinsic matrix
//

// for f=fx=fy and [cx,cy] is the image center
cv::Matx33f     cameraK (const cv::Size& imageSize, const float f);

// like above, plus f is computed as average between width/2 and height/2
cv::Matx33f     cameraK (const cv::Size& imageSize);




//
// camera pose to planar homography and back
//
// warning: Homography is not between cameras M0=[I,0] and M=[R,t] as in books
//          Instead it is between a known image and how this image will be seen
//          in middle of ground plane Z=0 by a camera with pose = evg::pose(R,t)
//
// note:    each camera has a 'pose' and intrinsic matrix 'K'
//

// planar homography: a 'plane' seen by a camera 'pose'. Not tested
cv::Matx33f     pose2H (const cv::Matx33f& K0, const cv::Matx41f& plane,
                        const cv::Matx44f& pose, const cv::Matx33f& K);
    
// planar homography: ground plane (0,0,1,0) seen from a camera 'pose'
cv::Matx33f     pose2H (const cv::Matx33f& K0,
                        const cv::Matx44f& pose, const cv::Matx33f& K);

// planar homography: 'plane' from 1st to 2nd camera
cv::Matx33f     deltaPose2H (const cv::Matx44f& pose1, const cv::Matx33f& K1,
                             const cv::Matx44f& pose2, const cv::Matx33f& K2,
                             const cv::Matx41f& plane = cv::Matx41f(0,0,1,0) );
    
// decompose planar homography from ground plane (0,0,1,0) to camera pose
cv::Matx44f     H2pose (const cv::Matx33f& K0, const cv::Matx33f& H,
                        const cv::Matx33f& K);




//
// some displaying functions
//
    
// display pairs of homogeneous points on an image
void            displayHomogeneousPairs (const cv::Mat& coords1,
                                         const cv::Mat& coords2,
                                         cv::Mat& background);

// draw an icon of camera as its projection on background == ground plane Z=0
void            drawCamera (const cv::Matx44f& pose, cv::Mat& background);

// draw 'maps' as viewed from 'viewPoint' on 'result' image
//   'maps' are seen as ground planes in reference frames given by 'planePoses'
void            projectPlanes (const std::vector<cv::Mat>& maps,
                               const std::vector<cv::Matx44f>& planePoses,
                               const cv::Matx44f& viewPoint,
                               cv::Mat& result);


} // namespace evg
} // namespace cv

#endif // EVG_GEOMETRY_3D
