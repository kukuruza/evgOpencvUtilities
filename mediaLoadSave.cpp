#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include "mediaLoadSave.h"

using namespace std;
using namespace cv;
using namespace boost::filesystem;


Mat evg::loadImage (const std::string& imagePath)
{
    path p(imagePath);
    if (! exists(p) )
    {
        std::cerr << "evg::loadImage(): Path " << absolute(imagePath)
                  << " does not exist." << std::endl;
        throw std::exception();
    }
    cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    if (! image.data )
    {
        std::cerr << "Image " << absolute(imagePath) << " failed to open." << std::endl;
        throw std::exception();
    }
    return image;
}


bool evg::loadImage (const std::string& imagePath, cv::Mat& image)
{
    try {
        image = loadImage(imagePath);
        return 1;
    } catch(...) { return 0; }
}


VideoCapture evg::openVideo(const string& videoPath)
{
    path p(videoPath);
    if (! exists(p) )
    {
        cerr << "evg::openVideo(): Video path " << absolute(videoPath)
             << " does not exist." << endl;
        throw exception();
    }
    
    VideoCapture video;
    if (! video.open(videoPath) )
    {
        cerr << "evg::openVideo(): Video " << absolute(videoPath)
             << " failed to open." << endl;
        throw exception();
    }
    
    return video;
}


bool evg::openVideo(const string& videoPath, VideoCapture& _video)
{
    try {
        _video = openVideo(videoPath);
        return 1;
    } catch (...) { return 0; }
}
    
    
cv::VideoWriter evg::newVideo (const std::string& _videoOutPath, const cv::VideoCapture& _videoIn)
{
    // get parameters from the input video
    cv::VideoCapture videoIn = _videoIn;
    int codec = (int)(videoIn.get(CV_CAP_PROP_FOURCC));
    double fps = videoIn.get(CV_CAP_PROP_FPS);
    const Size frameSize ( (int)(videoIn.get(CV_CAP_PROP_FRAME_WIDTH)),
                           (int)(videoIn.get(CV_CAP_PROP_FRAME_HEIGHT)) );
    
    // in case of camera input
    // set default frame rate
    const int DefaultFps = 30;
    if (fps == 0) fps = DefaultFps;
    // set default code (combine ascii code of each of four letter into an int)
    const char* DefaultCodec = "XVID";
    if (codec == 0) codec = (DefaultCodec[0] << 24) + (DefaultCodec[1] << 16)
                          + (DefaultCodec[2] << 8) + DefaultCodec[3];
    
    cout << "evg::newVideo(): codec code = " << codec << endl;
    cout << "                 frame rate = " << fps << endl;
    cout << "                 frame size = [" << frameSize.width << " x "
         << frameSize.height << "]" << endl;

    // check the parent path for output video
    path p(_videoOutPath);
    if (! exists(p.parent_path()) )
    {
        cerr << "evg::newVideo(): Video directory path " << p.parent_path()
             << " does not exist." << endl;
        throw exception();
    }

    // open video for output
    VideoWriter videoOut;
    if (! videoOut.open(_videoOutPath, codec, fps, frameSize) )
    {
        cerr << "evg::newVideo(): Video " << absolute(_videoOutPath)
             << " failed to open." << endl;
        throw exception();
    }
    
    return videoOut;
}


bool evg::newVideo (const string& _videoOutPath, const VideoCapture& _videoIn, VideoWriter& videoOut)
{
    try {
        videoOut = evg::newVideo(_videoOutPath, _videoIn);
        return 1;
    } catch (...) { return 0; }
}


Mat evg::undistortImage(const std::string& calibrationPath, const Mat& _image)
{
    // open calibration file
    path p(calibrationPath);
    if (! exists(p) )
    {
        cerr << "evg::calibImage(): Path " << absolute(calibrationPath)
        << " does not exist." << endl;
        throw exception();
    }
    FileStorage fs(calibrationPath.c_str(), FileStorage::READ);
    if (! fs.isOpened() )
    {
        cerr << "evg::calibImage(): Calibration file " << absolute(calibrationPath)
        << " failed to open." << endl;
        throw exception();
    }
    
    // read calibration file
    Mat cameraMatrix, distCoeffs;
    fs["Camera_Matrix"] >> cameraMatrix;
    fs["Distortion_Coefficients"] >> distCoeffs;
    fs.release();
    
    // undistort image
    Mat resultImage;
    undistort (_image, resultImage, cameraMatrix, distCoeffs);
    
    return resultImage;
}


bool evg::undistortImageBool (const string& calibrationPath, Mat& image)
{
    try {
        image = undistortImage(calibrationPath, image);
        return 1;
    } catch (...) { return 0; }
}


void evg::testImage (const cv::Mat& image)
{
    cout << "testing image..." << endl;
    namedWindow("evg_test", CV_WINDOW_AUTOSIZE);
    imshow("evg_test", image);
    while (1)
        if ( cv::waitKey(20) == 27 )
            throw std::exception();
}


bool evg::testImageBool (const cv::Mat& image)
{
    try {
        testImage(image);
        return 1;
    } catch(...) { return 0; }
}


// delimiter is always space now
// matrix will be always CV_32F on output, it will put zeros for missing values
Mat evg::dlmread (const std::string& dlmfilePath)
{
    // open file
    path p(dlmfilePath);
    if (! exists(p) )
    {
        cerr << "evg::dlmread(): Path " << absolute(dlmfilePath)
             << " does not exist." << endl;
        throw exception();
    }
    ifstream fileStream (dlmfilePath.c_str());
    if (!fileStream.good())
    {
        cerr << "evg::dlmread(): File " << absolute(dlmfilePath)
             << " failed to open." << endl;
        throw exception();
    }
    
    // read file to compute the matrix size [numRows, numCols]
    string line;
    istringstream iss;
    unsigned int row = 0, col = 0, numRows = 0, numCols = 0;
    for (row = 0; getline(fileStream, line); ++row)
    {
        // process the empty line in the end of file
        if (line == "") { --row; break; }
        iss.clear();
        iss.str(line);
        double dummy;
        for (col = 0; (iss >> dummy); ++col)
            ;
        if (col > numCols) numCols = col;
    }
    numRows = row;
    if(fileStream.bad() || iss.bad())
    {
        std::cerr << "evg::dlmread(): error reading the file." << std::endl;
        throw exception();
    }
    
    // rewind the file
    fileStream.clear();
    fileStream.seekg(ios_base::beg);
    
    // create the matrix of result size
    Mat matrix = Mat::zeros(numRows, numCols, CV_32F);
    
    // read file and put values into Mat
    float num;
    for (int row = 0; getline(fileStream, line) && row != matrix.rows; ++row)
    {
        iss.clear();
        iss.str(line);
        for (int col = 0; col != matrix.cols && (iss >> num); ++col)
            matrix.at<float>(row, col) = num;
    }
    if(fileStream.bad() || iss.bad())
    {
        std::cerr << "evg::dlmread(): error reading the file." << std::endl;
        throw exception();
    }
    
    return matrix;
}


bool evg::dlmread (const std::string& dlmfilePath, cv::Mat& _matrix)
{
    try {
        _matrix = dlmread (dlmfilePath);
        return 1;
    } catch(...) { return 0; }
}


// delimiter is currently space only
void evg::dlmwrite (const std::string& dlmfilePath, const cv::Mat& _matrix)
{
    // check the directory path for output video
    path p(dlmfilePath);
    if (! exists(p.parent_path()) )
    {
        cerr << "evg::dlmwrite(): Directory path " << p.parent_path()
             << " does not exist." << endl;
        throw exception();
    }

    // open file
    ofstream ofs (p.string().c_str());
    if (!ofs.good())
    {
        cerr << "evg::dlmwrite(): File " << absolute(dlmfilePath)
             << " failed to open." << endl;
        throw exception();
    }
    
    // write stuff
    Mat matrix = _matrix;
    if (matrix.type() != CV_32F)
        matrix.convertTo(matrix, CV_32F);
    for (int i = 0; i != matrix.rows; ++i)
        for (int j = 0; j != matrix.cols; ++j)
            ofs << matrix.at<float>(i, j) << (j == matrix.cols-1 ? '\n' : ' ');
    ofs << flush;
}


bool evg::dlmwriteBool (const std::string& dlmfilePath, const cv::Mat& matrix)
{
    try {
        dlmwrite(dlmfilePath, matrix);
        return 1;
    } catch(...) { return 0; }
}


// from http://stackoverflow.com/questions/3190378/opencv-store-to-database
void evg::saveMat ( const string& filename, const Mat& M)
{
    try {
        // only one channel
        assert (M.channels() == 1);
    
        // check the parent path for output video
        path p(filename);
        if (! exists(p.parent_path()) )
        {
            cerr << "evg::saveMat(): Directory path " << p.parent_path()
                 << " does not exist" << endl;
            throw exception();
        }
        if (M.empty())
        {
            cerr << "evg::saveMat(): matrix is empty" << endl;
            throw exception();
        }
        ofstream out(filename.c_str(), ios::out|ios::binary);
        if (!out)
        {
            cerr << "evg::saveMat(): cannot open file for writing" << endl;
            throw exception();
        }
        int cols = M.cols;
        int rows = M.rows;
        int chan = M.channels();
        int eSiz = int((M.dataend-M.datastart)/(cols*rows*chan));
        
        // Write header
        out.write((char*)&cols,sizeof(cols));
        out.write((char*)&rows,sizeof(rows));
        out.write((char*)&chan,sizeof(chan));
        out.write((char*)&eSiz,sizeof(eSiz));
        
        // Write data.
        if (M.isContinuous())
            out.write((char *)M.data,cols*rows*chan*eSiz);
        else
        {
            cerr << "evg::saveMat(): matrix must be continious." << endl;
            throw exception();
        }
        out.close();
    } catch(...) {
        cerr << "evg::saveMat(): exception caught." << endl;
        throw exception();
    }
}


bool evg::saveMatBool ( const string& filename, const Mat& M)
{
    try {
        saveMat (filename, M);
        return 1;
    } catch(...) { return 0; }
}


// from http://stackoverflow.com/questions/3190378/opencv-store-to-database
cv::Mat evg::readMat( const string& filename)
{
    try {
        Mat M;
    
        // open file
        path p(filename);
        if (! exists(p) )
        {
            cerr << "evg::dlmread(): Path " << absolute(filename)
                 << " does not exist." << endl;
            throw exception();
        }
        ifstream in(filename.c_str(), ios::in|ios::binary);
        if (!in)
        {
            cerr << "evg::readMat(): cannot open file for reading" << endl;
            throw exception();
        }
        int cols;
        int rows;
        int chan;
        int eSiz;
        
        // Read header
        in.read((char*)&cols,sizeof(cols));
        in.read((char*)&rows,sizeof(rows));
        in.read((char*)&chan,sizeof(chan));
        in.read((char*)&eSiz,sizeof(eSiz));
        
        // Determine type of the matrix
        int type = 0;
        switch (eSiz){
            case sizeof(char):
                type = CV_8UC(chan);
                break;
            case sizeof(float):
                type = CV_32FC(chan);
                break;
            case sizeof(double):
                type = CV_64FC(chan);
                break;
            default:
                cerr << "evg::readMat(): bad matrix header" << endl;
                throw exception();
        }
        
        // Alocate Matrix.
        M = Mat(rows,cols,type,Scalar(1));
        
        // Read data.
        assert(M.isContinuous());
        in.read((char *)M.data,cols*rows*chan*eSiz);

        in.close();
        return M;
    } catch(...) {
        cerr << "evg::readMat(): exception caught." << endl;
        throw exception();
    }
}


bool evg::readMat( const string& filename, cv::Mat& M )
{
    try {
        M = evg::readMat(filename);
        return 1;
    } catch(...) { return 0; }
}



evg::SrcVideo::SrcVideo (const Type _type, const std::string _videoPath)
  : type (_type),
    videoPath(_videoPath),
    CameraWidth(640),
    CameraHeight(480)
{
    if (type == FILE && videoPath == "")
        std::cerr << "warning: evg::SrcVideo::SrcVideo(): input is set as file, "
                  << "but file path is not specified." << std::endl;
}

bool evg::SrcVideo::openResource(VideoCapture& video)
{
    try {
        if (video.isOpened()) return 1;
        
        if (type == evg::SrcVideo::CAMERA)
        {
            if (! video.open(0))
            {
                cerr << "evg::SrcVideo::openResource(): Camera failed to open." << endl;
                return 0;
            }
            video.set(CV_CAP_PROP_FRAME_WIDTH, CameraWidth);
            video.set(CV_CAP_PROP_FRAME_HEIGHT, CameraHeight);
            cout << "note: evg::SrcVideo::openResource(): width x height = "
                 << video.get(CV_CAP_PROP_FRAME_WIDTH) << " x "
                 << video.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
            cout << "note: evg::SrcVideo::openResource(): Camera successfully opened."
                 << endl;
        }
        else if (type == evg::SrcVideo::FILE)
        {
            if (! evg::openVideo(videoPath, video)) return 0;
        }
        else
        {
            cout << "evg::SrcVideo::openResource(): bad input type." << endl;
            return 0;
        }
        return 1;
    } catch (...) {
        cerr << "evg::SrcVideo::openResource(): excepton caught." << endl;
        return 0;
    }
}

bool evg::SrcVideo::closeResource(VideoCapture& video)
{
    try {
        if (! video.isOpened()) return 1;
        video.release();
        return 1;
    } catch(...) {
        cerr << "evg::SrcVideo::closeResource(): exception caught." << endl;
        return 0;
    }
}
