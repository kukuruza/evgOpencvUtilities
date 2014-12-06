//
//  main.cpp
//  undistortImage
//
//  Undistort a set of images using opencv calibration file
//    A set of images is
//
//  Created by Evgeny Toropov on 5/24/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include "tclap/CmdLine.h"

#include "mediaIO.h"

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace TCLAP;    // library for managing program input

int main(int argc, const char * argv[])
{
    try {
        
        // define requested input
        CmdLine cmd ("undistort a set of images using opencv", ' ', "0", false);
        typedef ValueArg<string> ArgStr;
        typedef UnlabeledMultiArg<string> MultiargStr;
        ArgStr cmdOutput ("o", "output", "directory for output images",
                          true, "", "string", cmd);
        ArgStr cmdCalibr ("c", "calibr", "opencv calibration file",
                          true, "", "string", cmd);
        MultiargStr cmdInput ("input", "input images path pattern",
                              true, "string", cmd);
        
        // parse user input
        cmd.parse(argc, argv);
        path outDirPath = path(cmdOutput.getValue());
        path calibrPath = path(cmdCalibr.getValue());
        vector<string> inPathList = cmdInput.getValue();
        
        // check existance of calibration file
        if (! exists(calibrPath) )
        {
            cerr << "Calibration file " << absolute(calibrPath) << " does not exist" << endl;
            return -1;
        }
        
        // create a new directory if necessary
        if (! exists(outDirPath) )
            if (! create_directory(outDirPath) )
            {
                cerr << "Could not create directory " << absolute(outDirPath)
                << ". Stop" << endl;
                return -1;
            }
        
        
        // for every file
        for (int i = 0; i != inPathList.size(); ++i)
        {
            path inImagePath = path(inPathList[i]);
            
            cout << "working on file: " << inImagePath << "... " << flush;
            
            // open image
            Mat image;
            if ( !evg::loadImage (inImagePath.string(), image) )
            {
                cout << "failed to open." << endl;
                continue;
            }
            
            // undistort image
            if ( !evg::undistortImageBool (calibrPath.string(), image) )
            {
                cout << "failed to undistort." << endl;
                continue;
            }
            
            // save image
            path outImagePath = outDirPath / inImagePath.filename();
            if ( !imwrite(outImagePath.string(), image) )
            {
                cout << "failed to save image. Will stop now." << endl;
                return -1;
            };
            
            cout << "succeded." << endl;
        }
        
        cout << "Complete." << endl;
        return 0;
    } catch(...) {
        cerr << "Exception in undistortImages(). Stop." << endl;
        return -1;
    }
}
