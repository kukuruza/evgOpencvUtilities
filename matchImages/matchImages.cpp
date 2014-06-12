//
//  matchImages.cpp
//  matchImages. Vary detectors, extractors, matchers and parameters and write results
//
//  Created by Evgeny on 7/22/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>

#include <tclap/CmdLine.h>

#include <opencv2/imgproc/imgproc.hpp>

#include "mediaIO.h"
#include "featuresIO.h"



using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace TCLAP;


cv::FeatureDetector* newFeatureDetector (const std::string& featureType)
{
    FeatureDetector* detector (NULL);
    if (featureType == "sift")
        detector = new SIFT();
    else if (featureType == "surf")
        detector = new SURF();
    else if (featureType == "orb")
        detector = new ORB();
    else if (featureType == "brisk")
        detector = new BRISK();
    else assert(0);
    return detector;
}


cv::DescriptorExtractor* newDescriptorExtractor (const string& featureType)
{
    DescriptorExtractor* extractor (NULL);
    if (featureType == "sift")
        extractor = new SIFT();
    else if (featureType == "surf")
        extractor = new SURF();
    else if (featureType == "orb")
        extractor = new ORB();
    else if (featureType == "brisk")
        extractor = new BRISK();
    else assert(0);
    return extractor;
}


// hack to manage both float and integer type of descriptors in OpenCV
cv::DescriptorMatcher* newMatcher (const std::string& featureType, int verbose = 0)
{
    cv::DescriptorMatcher* matcher;
    if      (featureType == "sift" || featureType == "surf")
    {
        if (verbose) cout << "using FlannBasedMatcher on sift or surf" << endl;
        matcher = new FlannBasedMatcher();
    }
    else if (featureType == "orb" || featureType == "brisk")
    {
        if (verbose) cout << "using FlannBasedMatcher(LshIndexParams) on orb or brisk" << endl;
        matcher = new FlannBasedMatcher(new flann::LshIndexParams(8, 15, 2));
    }
    else
        assert (0);
    return matcher;
}




int main(int argc, const char * argv[])
{
    // parse input
    CmdLine cmd ("match a pair of images using specified features");
    
    vector<string> featureTypes;
    featureTypes.push_back("sift");
    featureTypes.push_back("surf");
    featureTypes.push_back("orb");
    featureTypes.push_back("brisk");
    ValuesConstraint<string> cmdFeatureTypes( featureTypes );
    ValueArg<string> cmdFeature("f", "feature", "feature type", true, "", &cmdFeatureTypes, cmd);
    
    ValueArg<string> cmd1st ("1", "1st", "1st image file path", true, "", "string", cmd);
    ValueArg<string> cmd2nd ("2", "2nd", "2nd image file path", true, "", "string", cmd);
    ValueArg<float> cmdThresh ("t", "threshold", "threshold for matching, 0-1, higher gives more matches", true, 3, "float", cmd);
    ValueArg<string> cmdOutM  ("o", "outmat", "file path for matches", false, "/dev/null", "string", cmd);
    SwitchArg cmdDisableImshow ("", "disable_image", "don't show image", cmd);
    MultiSwitchArg cmdVerbose ("v", "", "level of verbosity of output", cmd);
    
    cmd.parse(argc, argv);
    string           featureType    = cmdFeature.getValue();
    float            threshold      = cmdThresh.getValue();
    string           imageName1     = cmd1st.getValue();
    string           imageName2     = cmd2nd.getValue();
    string           outMName       = cmdOutM.getValue();
    bool             disableImshow  = cmdDisableImshow.getValue();
    int              verbose        = cmdVerbose.getValue();
    
    // file for output
    path outMPath = absolute(path(outMName));
    if (! exists(outMPath.parent_path()))
    {
        cerr << "parent path " << outMPath.parent_path() << " doesn't exist." << endl;
        return -1;
    }
    if (is_directory(outMPath))
    {
        cerr << "writeSimpleMatches: Need a filename, not a directory: " << outMPath << endl;
        return -1;
    }
    
    // load images
    Mat im1, im2;
    if (!evg::loadImage(imageName1, im1)) return 0;
    if (!evg::loadImage(imageName2, im2)) return 0;
    
    // setup detectors
    Ptr<FeatureDetector> detector = newFeatureDetector (featureType);
    Ptr<DescriptorExtractor> extractor = newDescriptorExtractor (featureType);
    Ptr<DescriptorMatcher> matcher = newMatcher (featureType, verbose);
    
    // match
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    vector< vector<DMatch> > matchesPairs;
    
    detector->detect (im1, keypoints1);
    detector->detect (im2, keypoints2);
    extractor->compute (im1, keypoints1, descriptors1);
    extractor->compute (im2, keypoints2, descriptors2);
    matcher->knnMatch (descriptors1, descriptors2, matchesPairs, 2);

    // filter based on relative distance to the two closest
    vector<DMatch> matches;
    matches.reserve (matchesPairs.size());
    for (int i = 0; i != matchesPairs.size(); ++i)
    {
        float ratio = matchesPairs[i][0].distance / matchesPairs[i][1].distance;
        if (ratio < threshold)
        {
            if (verbose >= 2) cout << ratio << " ";
            matchesPairs[i][0].distance = ratio;
            matches.push_back (matchesPairs[i][0]);
        }
    }
    if (verbose >= 2) cout << endl;

    
    // write results
    evg::writeSimpleMatches (outMPath.string(), imageName1, imageName2, keypoints1, keypoints2, matches);
    
    if (!disableImshow)
    {
        Mat im1gray, im2gray;
        cvtColor(im1, im1gray, CV_RGB2GRAY);
        cvtColor(im2, im2gray, CV_RGB2GRAY);
        float factor = float(1440) / im1gray.cols / 2;
        vector<KeyPoint> keypoints1im = keypoints1, keypoints2im = keypoints2;
        for (int i = 0; i != keypoints1im.size(); ++i)
        {
            keypoints1im[i].pt.x = keypoints1im[i].pt.x * factor;
            keypoints1im[i].pt.y = keypoints1im[i].pt.y * factor;
        }
        for (int i = 0; i != keypoints2im.size(); ++i)
        {
            keypoints2im[i].pt.x = keypoints2im[i].pt.x * factor;
            keypoints2im[i].pt.y = keypoints2im[i].pt.y * factor;
        }
        
        resize(im1gray, im1gray, Size(), factor, factor);
        resize(im2gray, im2gray, Size(), factor, factor);
        Mat imgMatches;
        drawMatches (im1gray, keypoints1im, im2gray, keypoints2im, matches, imgMatches,
                     Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imshow( "matches", imgMatches );
        if (waitKey(0) == 27) return 0;
    }
    
    return 0;
}

