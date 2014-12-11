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

}
}

#endif