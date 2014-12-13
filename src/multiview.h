#ifndef EVG_MULTIVIEW
#define EVG_MULTIVIEW

#include <boost/optional.hpp>

#include <opencv2/core/core.hpp>

namespace cv {
namespace evg {

// calculate essential matrix based on calibrated points
boost::optional<cv::Matx33f> findEssentialMat (const cv::Mat& p1, const cv::Mat& p2,
                                               float FundEstErrCutoff, float SRatioCutoff);


cv::Matx44f E2pose (const cv::Matx33f& E12);
cv::Matx33f pose2E (const cv::Matx44f& pose);



float epipolarError (const Matx44f& pose2, const Matx31f& p1, const Matx31f& p2);


// two-view and three-view constraints of V.Indelman
// pose2, pose3 - is the pose of the 2nd and 3rd cameras in the frame of the 1st camera
// p1, p2, p3 - projections of a 3D point into 1st, 2nd, and 3rd cameras

float twoViewError (const cv::Matx44f& pose2, const cv::Matx31f& p1, const cv::Matx31f& p2);

std::vector<float>
threeViewErrors (const cv::Matx44f& pose2, const cv::Matx44f& pose3,
                 const cv::Matx31f& p1, const cv::Matx31f& p2, const cv::Matx31f& p3);

}
}

#endif