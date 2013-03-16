#include <iostream>
#include "angles3D.h"

using namespace std;
using namespace cv;


Matx33f evg::rotAxisX (const float angle)
{
    return Matx33f (1,  0,           0,
                    0,  cos(angle), -sin(angle),
                    0,  sin(angle),  cos(angle) );
}

Matx33f evg::rotAxisY (const float angle)
{
    return Matx33f (cos(angle),  0,  sin(angle),
                    0,           1,  0,
                   -sin(angle),  0,  cos(angle) );
}

Matx33f evg::rotAxisZ (const float angle)
{
    return Matx33f (cos(angle), -sin(angle),  0,
                    sin(angle),  cos(angle),  0,
                    0,           0,           1 );
}


//
// All functions of type evg::euler2R_xx and evg::euler2R_xx are
//   implementations of http://nghiaho.com/?page_id=846
//
// note: The order of rotations is always roll -> pitch -> yaw
//

// XY. axes: roll - X, pitch - Y, yaw - Z (jet is moving in XY plane along X)
Matx33f evg::euler2R_XY (const float yaw, const float pitch, const float roll)
{
    Matx33f X_roll  = evg::rotAxisZ (roll);
    Matx33f Y_pitch = evg::rotAxisX (pitch);
    Matx33f Z_yaw   = evg::rotAxisY (yaw);
    return Z_yaw * Y_pitch * X_roll;
}

void evg::R2euler_XY (const Matx33f& R, float& yaw, float& pitch, float& roll)
{
    float r11 = R(0, 0);
    float r21 = R(1, 0);
    float r31 = R(2, 0);
    float r32 = R(2, 1);
    float r33 = R(2, 2);
    roll = atan2(r32, r33);
    pitch = atan2(-r31, sqrtf(r32 * r32 + r33 * r33));
    yaw = atan2(r21, r11);
}


// XZ. axes: roll - X, pitch - Z, yaw - Y (jet is moving in XZ plane along X)
Matx33f evg::euler2R_XZ (const float yaw, const float pitch, const float roll)
{
    Matx33f X_roll  = evg::rotAxisZ (roll);
    Matx33f Z_pitch = evg::rotAxisX (pitch);
    Matx33f Y_yaw   = evg::rotAxisY (yaw);
    return Y_yaw * Z_pitch * X_roll;
}


// ZX. axes: roll - Z, pitch - X, yaw - Y (jet is moving in ZX plane along Z)
Matx33f evg::euler2R_ZX (const float yaw, const float pitch, const float roll)
{
    Matx33f Z_roll  = evg::rotAxisZ (roll);
    Matx33f X_pitch = evg::rotAxisX (pitch);
    Matx33f Y_yaw   = evg::rotAxisY (yaw);
    return Y_yaw * X_pitch * Z_roll;
}



// taken from http://planning.cs.uiuc.edu/node198.html
//   and from http://www.cprogramming.com/tutorial/3d/quaternions.html
cv::Matx33f evg::sampleR (const float u1, const float u2, const float u3)
{
    // check that numbers are [0 1)
    assert (u1 >= 0 && u1 < 1 && u2 >= 0 && u2 < 1 && u3 >= 0 && u3 < 1);
    
    // make a "random" unit-length quaternion
    float q[4];
    q[0] = sqrtf(1 - u1) * sinf(float(2 * CV_PI * u2));
    q[1] = sqrtf(1 - u1) * cosf(float(2 * CV_PI * u2));
    q[2] = sqrtf(u1) * sinf(float(2 * CV_PI * u3));
    q[3] = sqrtf(u1) * cosf(float(2 * CV_PI * u3));
    assert( abs(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] - 1) < 0.0001f);
    
    // quaternion to rotation matrix
    Matx33f R;
    R(0,0) = q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3];
    R(0,1) = 2*q[1]*q[2] - 2*q[0]*q[3];
    R(0,2) = 2*q[1]*q[3] + 2*q[0]*q[2];
    R(1,0) = 2*q[1]*q[2] + 2*q[0]*q[3];
    R(1,1) = q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3];
    R(1,2) = 2*q[2]*q[3] - 2*q[0]*q[1];
    R(2,0) = 2*q[1]*q[3] - 2*q[0]*q[2];
    R(2,1) = 2*q[2]*q[3] + 2*q[0]*q[1];
    R(2,2) = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    assert (abs(sum(R.inv() - R.t())[0]) < 0.0001f);
    return R;
}



