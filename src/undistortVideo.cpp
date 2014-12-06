//
//  main.cpp
//  undistortVideo
//
//  Created by Evgeny Toropov on 5/11/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <iostream>

#include <opencv2/core/core.hpp>

#include "tclap/CmdLine.h"

#include "mediaIO.h"

using namespace std;
using namespace cv;       // OpenCV
using namespace TCLAP;    // tclap - library for managing program input

int main(int argc, const char * argv[])
{
    try {
        // define requested input
        ///TODO: add input from camera
        CmdLine cmd ("undistort video using opencv", ' ', "0", false);
        typedef ValueArg<string> Arg;
        typedef UnlabeledValueArg<string> UnlablArg;
        Arg cmdOutput ("o", "output", "output video file path",
                       true, "", "string", cmd);
        Arg cmdCalibr ("c", "calibr", "opencv calibration file",
                       true, "", "string", cmd);
        UnlablArg cmdInput ("input", "input video file path",
                            true, "", "string", cmd);
        
        // parse user input
        cmd.parse(argc, argv);
        string videoOutPath = cmdOutput.getValue();
        string calibrPath = cmdCalibr.getValue();
        string videoInPath = cmdInput.getValue();
        
        // open video
        VideoCapture videoIn = evg::openVideo (videoInPath);
        
        // undistort video
        if ( !evg::undistortVideoBool (calibrPath, videoIn, videoOutPath) )
            cout << "Failed." << endl;
        
        cout << "Complete." << endl;
        return 0;
    } catch(...) {
        cerr << "Exception in undistortVideo(). Abort." << endl;
        return -1;
    }
}

