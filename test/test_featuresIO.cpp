//
//  Created by Evgeny on 8/28/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//


#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "featuresIO.h"
#include "gtest/gtest.h"

using namespace std;
using namespace cv;
using namespace evg;

/*
bool equal (const cv::Mat mat1, const cv::Mat& mat2)
{
    if (mat1.size() != mat2.size()) return false;
    if (mat1.type() != mat2.type()) return false;
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    return !countNonZero(diff);
}
*/

int equalByRows (const cv::Mat mat1, const cv::Mat& mat2)
{
    if (mat1.size() != mat2.size()) return -1;
    if (mat1.type() != mat2.type()) return -2;
    for (int i = 0; i != mat1.rows; ++i)
    {
        Mat row1 = mat1.row(i);
        Mat row2 = mat2.row(i);
        cv::Mat diff;
        cv::compare(row1, row2, diff, cv::CMP_NE);
        if (countNonZero(diff))
        {
            cerr << "equalByRows: " << row1 << " vs " << row2 << endl;
            return i+1;
        }
    }
    return 0;
}

const float FLOAT_EPS = 0.00001f;
bool almostEqual (const float a, const float b)
{
    if (a == 0 && b == 0)
        return true;
    else if (a == 0 || b == 0)
        return abs(a - b) < FLOAT_EPS;
    else
        return abs(a - b) < FLOAT_EPS * (abs(a) + abs(b));
}


int equal (const cv::KeyPoint& key1, const cv::KeyPoint& key2)
{
    if (!almostEqual(key1.pt.x, key2.pt.x) || !almostEqual(key1.pt.y, key2.pt.y)) return -1;
    if (!almostEqual(key1.size, key2.size)) return -2;
    if (!almostEqual(key1.angle, key2.angle)) return -3;
    if (!almostEqual(key1.response, key2.response)) return -4;
    if (!almostEqual(key1.octave, key2.octave)) return -5;
    if (!almostEqual(key1.class_id, key2.class_id)) return -6;
    return 0;
}


class Data {
protected:
    vector<KeyPoint> _keypoints;
    Mat _descriptors;
public:
    Data () { };
    inline Data (const vector<KeyPoint>& keypoints, const Mat& descriptors)
    {
        assert (keypoints.size() == descriptors.rows);
        _keypoints = keypoints;
        _descriptors = descriptors;
    }
    inline vector<KeyPoint> getKeypoints() const { return _keypoints; }
    inline Mat getDescriptors() const { return _descriptors; }
    inline int getSize() const { assert(_keypoints.size() == _descriptors.rows); return _descriptors.rows; }
    inline int getDescrSize() const { return _descriptors.cols; }
    inline int matchesKeypoints (const vector<KeyPoint>& keypoints) const
    {
        if (_keypoints.size() != keypoints.size())
            return -7;
        for (int i = 0; i != keypoints.size(); ++i)
        {
            int aKeyEquals = equal(_keypoints[i], keypoints[i]);
            if (aKeyEquals != 0) return aKeyEquals;
        }
        return 0;
    }
    inline int matchesDescr (const Mat& descriptors) const
    {
        return equalByRows (_descriptors, descriptors);
    }
};


class AllFormats : public ::testing::Test {
protected:
    vector<KeyPoint> keypoints;
    Mat descriptors;
public:
    Data data;
};


/// =============================   VLFeat   ================================

class VLFeatFormat : public AllFormats {
public:
    VLFeatFormat()
    {
        vector<KeyPoint> keypoints;
        Mat descriptors;
        uchar arr1[] = {2, 1, 3, 98, 38, 14, 6, 2, 46, 31, 13, 45, 27, 26, 31, 14, 142, 17, 6, 15,
                      13, 8, 15, 24, 60, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 61, 81, 31, 18, 0, 25, 2, 1,
                      37, 142, 95, 79, 32, 142, 11, 0, 1, 7, 16, 49, 68, 73, 5, 0, 0, 0, 0, 0, 0,
                      4, 0, 0, 6, 31, 90, 142, 4, 44, 24, 27, 38, 135, 87, 31, 8, 142, 65, 15, 12,
                      9, 2, 2, 15, 64, 4, 0, 0, 0, 0, 0, 2, 14, 2, 26, 54, 40, 8, 23, 8, 39, 23, 40,
                      24, 5, 3, 32, 49, 142, 45, 17, 3, 0, 0, 11, 42, 48, 3, 0, 0, 0, 0, 0, 8 };
        uchar arr2[] = {0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 18, 48, 135, 40, 62, 0, 0, 0, 3, 10, 72,
                      140, 6, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 62, 10, 10, 11, 76, 158,
                      46, 55, 158, 25, 5, 2, 15, 50, 29, 158, 42, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
                      0, 0, 0, 0, 35, 72, 158, 34, 6, 32, 3, 3, 158, 158, 138, 8, 1, 8, 1, 11, 32,
                      10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 67, 0, 0, 0, 0, 0, 3,
                      45, 51, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0 };
        keypoints.push_back(KeyPoint(7.03005f, 112.668f, 1.19038f, 6.20909f));
        keypoints.push_back(KeyPoint(5.57416f, 117.696f, 3.62599f, 0.0700764f));
        descriptors = Mat(0, 128, CV_8U);
        Mat mat;
        mat = Mat(vector<uchar>(arr1, arr1 + 128)).t();
        descriptors.push_back( mat );
        mat = Mat(vector<uchar>(arr2, arr2 + 128)).t();
        descriptors.push_back( mat );
        data = Data (keypoints, descriptors);
    }
};


class VLFeatAscii : public VLFeatFormat {
protected:
    string pathVlfeatAsciiGood;
    inline VLFeatAscii() { pathVlfeatAsciiGood = "test/ioTestData/vlfeat_good.txt"; }
};

class VLFeatBin : public VLFeatFormat {
protected:
    string pathVlfeatBinGood;
    inline VLFeatBin() { pathVlfeatBinGood = "test/ioTestData/vlfeat_good.bin"; }
};



/// --------------------------  Read VLFeat Ascii --------------------------------


class ReadsVLFeatAscii : public VLFeatAscii {
public:
    bool result;
protected:
    ReadsVLFeatAscii() { result = readVLFeatFormat (pathVlfeatAsciiGood, keypoints, descriptors, false); }
};

TEST_F (ReadsVLFeatAscii, returnValue) {
    EXPECT_TRUE (result);
}

TEST_F (ReadsVLFeatAscii, correctNumPoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
}

TEST_F (ReadsVLFeatAscii, correctDescriptorSize) {
    EXPECT_EQ (data.getSize(), descriptors.rows);
    EXPECT_EQ (128, descriptors.cols);
}

TEST_F (ReadsVLFeatAscii, correctDescriptors) {
    EXPECT_EQ (0, data.matchesDescr(descriptors));
}

TEST_F (ReadsVLFeatAscii, correctKeypoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
    EXPECT_EQ (0, data.matchesKeypoints(keypoints));
}


/// ---------------------------- Write VLFeat Ascii -------------------------------

class WritesVLFeat : public VLFeatAscii {
protected:
    string pathVlfeatWriteGot;
    WritesVLFeat()
    {
        pathVlfeatWriteGot = "test/ioTestData/vlfeat_write_got.txt";
        result = writeVLFeatFormat (pathVlfeatWriteGot, data.getKeypoints(), data.getDescriptors(), false);
    }
public:
    bool result;
};


TEST_F (WritesVLFeat, returnSuccess) {
    EXPECT_TRUE (result);
}

TEST_F (WritesVLFeat, correctFile)
{
    ifstream ifsActual (pathVlfeatWriteGot.c_str());
    ifstream ifsExpected (pathVlfeatAsciiGood.c_str());
    assert (ifsActual && ifsExpected);
    
    float value;
    vector<float> valuesActual, valuesExpected;;
    while (ifsActual) { ifsActual >> value; valuesActual.push_back(value); }
    while (ifsExpected) { ifsExpected >> value; valuesExpected.push_back(value); }

    EXPECT_EQ (valuesActual.size(), valuesExpected.size());
    EXPECT_EQ (valuesActual, valuesExpected);
}


/// -------------------------------- Read VLFeat Bin ------------------------------------

class ReadsVLFeatBin : public VLFeatBin {
public:
    bool result;
protected:
    ReadsVLFeatBin() { result = readVLFeatFormat (pathVlfeatBinGood, keypoints, descriptors, true); }
};


TEST_F (ReadsVLFeatBin, returnValue) {
    EXPECT_TRUE (result);
}

TEST_F (ReadsVLFeatBin, correctNumPoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
}

TEST_F (ReadsVLFeatBin, correctDescriptorSize) {
    EXPECT_EQ (data.getSize(), descriptors.rows);
    EXPECT_EQ (data.getDescrSize(), descriptors.cols);
}

TEST_F (ReadsVLFeatBin, correctDescriptors) {
    EXPECT_EQ (0, data.matchesDescr(descriptors));
}

TEST_F (ReadsVLFeatBin, correctKeypoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
    EXPECT_EQ (0, data.matchesKeypoints(keypoints));
}



/// ---------------------------- Write VLFeat Bin -------------------------------

class WritesVLFeatBin : public VLFeatBin {
protected:
    string pathVlfeatWriteBinGot;
    WritesVLFeatBin()
    {
        pathVlfeatWriteBinGot = "test/ioTestData/vlfeat_write_got.bin";
        result = writeVLFeatFormat (pathVlfeatWriteBinGot, data.getKeypoints(), data.getDescriptors(), true);
    }
public:
    bool result;
};


TEST_F (WritesVLFeatBin, returnSuccess) {
    EXPECT_TRUE (result);
}

TEST_F (WritesVLFeatBin, correctFileSize)
{
    ifstream ifsActual (pathVlfeatWriteBinGot.c_str(), ios::binary);
    ifstream ifsExpected (pathVlfeatBinGood.c_str(), ios::binary);
    assert (ifsActual && ifsExpected);
    
    ifsActual.seekg(0, std::ifstream::end);
    ifsExpected.seekg(0, std::ifstream::end);
    
    EXPECT_EQ (ifsActual.tellg(), ifsExpected.tellg());
    
    ifsActual.close();
    ifsExpected.close();
}

TEST_F (WritesVLFeatBin, correctContents)
{
    ifstream ifsActual (pathVlfeatWriteBinGot.c_str(), ios::binary);
    ifstream ifsExpected (pathVlfeatBinGood.c_str(), ios::binary);
    assert (ifsActual && ifsExpected);
    
    while (ifsActual && ifsExpected)
    {
        const int HeaderSize = 4;
        double header1[HeaderSize], header2[HeaderSize];
        ifsActual.read   ((char*)&header1, sizeof(header1));
        ifsExpected.read ((char*)&header2, sizeof(header2));
        
        EXPECT_TRUE (almostEqual(float(header1[0]), float(header2[0])));
        EXPECT_TRUE (almostEqual(float(header1[1]), float(header2[1])));
        EXPECT_TRUE (almostEqual(float(header1[2]), float(header2[2])));
        EXPECT_TRUE (almostEqual(float(header1[3]), float(header2[3])));
        
        const int DescrSize = 128;
        char descr1[DescrSize], descr2[DescrSize];
        ifsActual.read   ((char *) &descr1, sizeof(descr1));
        ifsExpected.read ((char *) &descr2, sizeof(descr2));
    
        vector<char> descrVect1 (descr1, descr1 + DescrSize);
        vector<char> descrVect2 (descr2, descr2 + DescrSize);
        EXPECT_EQ (descrVect1, descrVect2);
    }
}


/// =============================     Ubc    ================================


class UbcFormat : public AllFormats {
};


class UbcAscii : public UbcFormat {
protected:
    string pathUbcAsciiGood;
public:
    inline UbcAscii()
    {
        pathUbcAsciiGood = "test/ioTestData/ubc_good.txt";

        vector<KeyPoint> keypoints;
        Mat descriptors;
        uchar arr1[] = { 2, 2, 6, 13, 40, 103, 5, 1, 44, 14, 31, 26, 30, 46, 12, 32, 140, 24, 16, 9,
                         13, 16, 6, 20, 62, 3, 0, 0, 0, 0, 0, 5, 0, 0, 24, 34, 81, 55, 0, 0,
                         25, 32, 78, 99, 142, 35, 1, 2, 142, 65, 49, 15, 7, 1, 0, 14, 73, 0, 0, 0,
                         0, 0, 0, 6, 4, 5, 142, 87, 32, 8, 0, 0, 46, 8, 31, 86, 131, 37, 28, 26,
                         142, 14, 2, 2, 9, 11, 14, 73, 60, 1, 0, 0, 0, 0, 0, 5, 16, 9, 21, 8,
                         42, 57, 28, 2, 40, 50, 31, 3, 6, 24, 40, 24, 142, 38, 9, 0, 0, 3, 16, 47,
                         44, 6, 0, 0, 0, 0, 0, 3 };
        keypoints.push_back(KeyPoint(7.03f, 112.67f, 1.19f, 0.092f));
        keypoints.push_back(KeyPoint(7.03f, 112.67f, 1.19f, 0.092f));
        descriptors = Mat(0, 128, CV_8U);
        Mat mat;
        mat = Mat(vector<uchar>(arr1, arr1 + 128)).t();
        descriptors.push_back( mat );
        descriptors.push_back( mat );
        data = Data (keypoints, descriptors);
    }
};


class UbcBin : public UbcFormat {
protected:
    string pathUbcBinGood;
public:
    inline UbcBin()
    {
        pathUbcBinGood  = "test/ioTestData/ubc_good.bin";

        vector<KeyPoint> keypoints;
        Mat descriptors;
        uchar arr1[] = { 11, 99, 178, 64, 90, 100, 235, 66, 113, 113, 113, 0, 81, 20, 104, 64, 118,
                         123, 143, 61, 2, 1, 3, 98, 38, 14, 6, 2, 46, 31, 13, 45, 27, 26, 31, 14,
                         142, 17, 6, 15, 13, 8, 15, 24, 59, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 61, 81,
                         31, 18, 0, 25, 2, 1, 37, 142, 95, 79, 32, 142, 11, 0, 1, 7, 16, 49, 68,
                         73, 5, 0, 0, 0, 0, 0, 0, 4, 0, 0, 6, 31, 90, 142, 4, 44, 24, 27, 38, 135,
                         87, 31, 8, 142, 65, 15, 12, 9, 2, 2, 15, 64, 4, 0, 0, 0, 0, 0, 2, 14, 2,
                         26, 54, 40, 8, 23, 8, 39, 23, 40, 24 };
        keypoints.push_back (KeyPoint(7.03007f, 112.668f, 1.19032f, 6.20913f));
        uchar arr2[] = { 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 18, 48, 135, 40, 62, 0, 0, 0, 3, 10,
                         73, 140, 6, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 62, 10, 10, 11,
                         76, 158, 46, 55, 158, 25, 5, 2, 15, 50, 29, 158, 42, 0, 0, 0, 0, 0, 0, 7,
                         0, 0, 0, 0, 0, 0, 0, 0, 35, 72, 158, 34, 6, 32, 3, 3, 158, 158, 138, 8, 1,
                         8, 1, 11, 32, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 67, 0,
                         0, 0, 0, 0, 3, 45, 51, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0 };
        keypoints.push_back (KeyPoint(0, 0, 0, 0));
        descriptors = Mat(0, 128, CV_8U);
        Mat mat;
        mat = Mat(vector<uchar>(arr1, arr1 + 128)).t();
        descriptors.push_back( mat );
        mat = Mat(vector<uchar>(arr2, arr2 + 128)).t();
        descriptors.push_back( mat );
        data = Data (keypoints, descriptors);
    }
};


/// ---------------------------- Write Ubc Ascii -------------------------------



class WritesUbcAscii : public UbcAscii {
protected:
    string pathUbcWriteAsciiGot;
    WritesUbcAscii()
    {
        pathUbcWriteAsciiGot = "test/ioTestData/ubc_write_got.txt";
        result = writeUbcFormat (pathUbcWriteAsciiGot, data.getKeypoints(), data.getDescriptors(), false);
    }
public:
    bool result;
};


TEST_F (WritesUbcAscii, returnSuccess) {
    EXPECT_TRUE (result);
}

TEST_F (WritesUbcAscii, correctFile)
{
    ifstream ifsActual (pathUbcWriteAsciiGot.c_str());
    ifstream ifsExpected (pathUbcAsciiGood.c_str());
    assert (ifsActual && ifsExpected);
    
    float value;
    vector<float> valuesActual, valuesExpected;;
    while (ifsActual) { ifsActual >> value; valuesActual.push_back(value); }
    while (ifsExpected) { ifsExpected >> value; valuesExpected.push_back(value); }

    EXPECT_EQ (valuesActual.size(), valuesExpected.size());
    EXPECT_EQ (valuesActual, valuesExpected);
}



/// --------------------------  Read Ubc Ascii --------------------------------


class ReadsUbcAscii : public UbcAscii {
public:
    bool result;
protected:
    ReadsUbcAscii() { result = readUbcFormat (pathUbcAsciiGood, keypoints, descriptors, false); }
};

TEST_F (ReadsUbcAscii, returnValue) {
    EXPECT_TRUE (result);
}

TEST_F (ReadsUbcAscii, correctNumPoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
}

TEST_F (ReadsUbcAscii, correctDescriptorSize) {
    EXPECT_EQ (data.getSize(), descriptors.rows);
    EXPECT_EQ (data.getDescrSize(), descriptors.cols);
}

TEST_F (ReadsUbcAscii, correctDescriptors) {
    EXPECT_EQ (0, data.matchesDescr(descriptors));
}

TEST_F (ReadsUbcAscii, correctKeypoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
    EXPECT_EQ (0, data.matchesKeypoints(keypoints));
}


/// ---------------------------- Write Ubc Bin -------------------------------

class WritesUbcBin : public UbcBin {
protected:
    string pathUbcWriteBinGot;
    WritesUbcBin()
    {
        pathUbcWriteBinGot = "test/ioTestData/ubc_write_got.bin";
        result = writeUbcFormat (pathUbcWriteBinGot, data.getKeypoints(), data.getDescriptors(), true);
    }
public:
    bool result;
};


TEST_F (WritesUbcBin, returnSuccess) {
    EXPECT_TRUE (result);
}

TEST_F (WritesUbcBin, correctFileSize)
{
    ifstream ifsActual (pathUbcWriteBinGot.c_str(), ios::binary);
    ifstream ifsExpected (pathUbcBinGood.c_str(), ios::binary);
    assert (ifsActual && ifsExpected);
    
    ifsActual.seekg(0, std::ifstream::end);
    ifsExpected.seekg(0, std::ifstream::end);
    
    EXPECT_EQ (ifsActual.tellg(), ifsExpected.tellg());
    
    ifsActual.close();
    ifsExpected.close();
}

TEST_F (WritesUbcBin, correctContents)
{
    ifstream ifsActual (pathUbcWriteBinGot.c_str(), ios::binary);
    ifstream ifsExpected (pathUbcBinGood.c_str(), ios::binary);
    assert (ifsActual && ifsExpected);
    
    while (ifsActual && ifsExpected)
    {
        const int HeaderSize = 4;
        double header1[HeaderSize], header2[HeaderSize];
        ifsActual.read   ((char*)&header1, sizeof(header1));
        ifsExpected.read ((char*)&header2, sizeof(header2));
        
        EXPECT_TRUE (almostEqual(float(header1[0]), float(header2[0])));
        EXPECT_TRUE (almostEqual(float(header1[1]), float(header2[1])));
        EXPECT_TRUE (almostEqual(float(header1[2]), float(header2[2])));
        EXPECT_TRUE (almostEqual(float(header1[3]), float(header2[3])));
        
        const int DescrSize = 128;
        char descr1[DescrSize], descr2[DescrSize];
        ifsActual.read   ((char *) &descr1, sizeof(descr1));
        ifsExpected.read ((char *) &descr2, sizeof(descr2));
    
        vector<char> descrVect1 (descr1, descr1 + DescrSize);
        vector<char> descrVect2 (descr2, descr2 + DescrSize);
        EXPECT_EQ (descrVect1, descrVect2);
    }
}



/// -------------------------------- Read Ubc Bin ------------------------------------

class ReadsUbcBin : public UbcBin {
public:
    bool result;
protected:
    ReadsUbcBin() { result = readUbcFormat (pathUbcBinGood, keypoints, descriptors, true); }
};


TEST_F (ReadsUbcBin, returnValue) {
    EXPECT_TRUE (result);
}

TEST_F (ReadsUbcBin, correctNumPoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
}

TEST_F (ReadsUbcBin, correctDescriptorSize) {
    EXPECT_EQ (data.getSize(), descriptors.rows);
    EXPECT_EQ (data.getDescrSize(), descriptors.cols);
}

TEST_F (ReadsUbcBin, correctDescriptors) {
    EXPECT_EQ (0, data.matchesDescr(descriptors));
}

TEST_F (ReadsUbcBin, correctKeypoints) {
    EXPECT_EQ (data.getSize(), keypoints.size());
    EXPECT_EQ (0, data.matchesKeypoints(keypoints));
}








int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}



