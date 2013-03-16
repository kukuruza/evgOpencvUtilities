#ifndef EVG_MEDIA_LOAD_SAVE
#define EVG_MEDIA_LOAD_SAVE

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//
// Evgeny Toropov, 2012
// Some functions for loading and saving media in Opencv using boost::filesystem
//
// Rationale: the goal of the library is to check IO in Opencv.
//   I want to be able to load/save images and video and then forget about it.
//   If it goes wrong, an error message should tell me what is wrong.
//   I do not like getting Opencv assertion failure somewhere in the midway.
//   Nothing complicated is there in the library, just useful functionality
//   Later I added other functionality connected with IO or with media
//
// Topics:
//   load/save/start images/video
//   smth extra for images
//   read/write space-delimited files
//   read/write cv::Mat as bin files (to save space compared to cv::FileStorage)
//   class to switch between video input from camera and from file
//
// Dependences:
//   opencv_core, opencv_imgproc, opencv_highgui, boost::filesystem, boost::system
//
// Notes:
//   everything is wrapped in namespace 'evg' for convenience
//   error policy: most functions exist in two forms:
//     1) void: write to std::cerr and throw exception on failure;
//     2) bool: write to std::cerr and return 0 on failure
//

namespace evg {
    

//
// load/save/start images/video
//

cv::Mat          loadImage (const std::string& imagePath);
bool             loadImage (const std::string& imagePath, cv::Mat& image);

cv::VideoCapture openVideo (const std::string& videoPath);
bool             openVideo (const std::string& videoPath, cv::VideoCapture& video);

// open a video for writing based on input video parameters
cv::VideoWriter  newVideo (const std::string& videoOutPath, const cv::VideoCapture& videoIn);
bool             newVideo (const std::string& videoOutPath, const cv::VideoCapture& videoIn,
                           cv::VideoWriter& videoOut);


//
// extra for images
//

// undistort image using pre-set calibration from calibration file
cv::Mat          undistortImage (const std::string& calibrationPath, const cv::Mat& image);
bool             undistortImageBool (const std::string& calibrationPath, cv::Mat& image);

// display image in a new window and wait for Esc key from user before closing it
void             testImage     (const cv::Mat& image);
bool             testImageBool (const cv::Mat& image);



//
// read/write space-delimited files
//

void             dlmwrite (const std::string& dlmfilePath, const cv::Mat& matrix);
bool             dlmwriteBool (const std::string& dlmfilePath, const cv::Mat& matrix);

// matrix will be always CV_32F on output, number of columns - from the largest row
// function will complete shorter rows with zeros
cv::Mat          dlmread (const std::string& dlmfilePath);
bool             dlmread (const std::string& dlmfilePath, cv::Mat& matrix);



//
// read/write Mat files as .bin
//

// only for 1 channel, CV_8U, CV_32F, CV_64F matrices
void             saveMat (const std::string& binFilepath, const cv::Mat& matrix);
bool             saveMatBool (const std::string& binFilepath, const cv::Mat& matrix);
cv::Mat          readMat (const std::string& binFilepath);
bool             readMat (const std::string& binFilepath, cv::Mat& matrix);



//
// class to switch between video input from camera and from file
//   To be used when video data can be from either camera or file
//

class SrcVideo {
public:
    enum Type { NOT_SET = -1, CAMERA = 0, FILE = 1 };
private:
    Type type;
    std::string videoPath;
    // your favorite camera resoluton, it is now set in constructor to 640 x 480
    const unsigned int CameraWidth, CameraHeight;

    friend bool operator== (const SrcVideo&, const SrcVideo&);
    friend bool operator!= (const SrcVideo&, const SrcVideo&);
public:

    // constructor
    SrcVideo (const Type _type = NOT_SET, const std::string _videoPath = "");
    
    // start video from specified input
    bool                        openResource (cv::VideoCapture& video);
    
    // goal of closeResource() is to manage cv::VideoCapture within this class
    bool                        closeResource (cv::VideoCapture& video);
    
    inline const Type           getType() const { return type; }
    inline const std::string&   getPath() const { return videoPath; }
};

inline bool operator== (const SrcVideo& a, const SrcVideo& b)
{
    return (a.type == b.type && a.videoPath == b.videoPath &&
            a.CameraWidth == b.CameraWidth && a.CameraHeight == b.CameraHeight);
}

inline bool operator!= (const SrcVideo& a, const SrcVideo& b)
{
    return !(a == b);
}
    

    
} // namespace evg

#endif // EVG_MEDIA_LOAD_SAVE