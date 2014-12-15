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


// epipolar constraint [1] and two-view and three-view constraints [2]
// pose2, pose3 - is the pose of the 2nd and 3rd cameras in the frame of the 1st camera
// p1, p2, p3 - projections of a 3D point into 1st, 2nd, and 3rd cameras

float epipolarError (const Matx44f& pose2, const Matx31f& p1, const Matx31f& p2);

float twoViewError (const cv::Matx44f& pose2, const cv::Matx31f& p1, const cv::Matx31f& p2);

std::vector<float>
threeViewErrors (const cv::Matx44f& pose2, const cv::Matx44f& pose3,
                 const cv::Matx31f& p1, const cv::Matx31f& p2, const cv::Matx31f& p3);


// probabilities





}
}

#endif


/*

[1] @Book{Hartley2004,
    author = "Hartley, R.~I. and Zisserman, A.",
    title = "Multiple View Geometry in Computer Vision",
    edition = "Second",
    year = "2004",
    publisher = "Cambridge University Press, ISBN: 0521540518"
    }

[2] @phdthesis{Indelman11thesis,
    author = {V. Indelman},
    title  = {Navigation Performance Enhancement Using Online Mosaicking},
    School = {Technion - Israel Institute of Technology},
    year   = 2011,
    }

[3] @article{Hebert_2014_7574,
        author = "Jean Ponce and Martial Hebert",
        title = "Trinocular Geometry Revisited",
        journal = "Proc. Computer Vision and Pattern Recognition (CVPR), 2014",
        month = "March",
        year = "2014",
    }
*/
