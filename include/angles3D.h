#ifndef EVG_ANGLES_3D
#define EVG_ANGLES_3D

#include <opencv2/core/core.hpp>

//
// Evgeny Toropov, 2012
// Some functions to work with rotations using Opencv
//
// Topics:
//   1. rotation around an axis
//   2. euler angles vs. rotation matrix
//   3. random sampling of rotation
//
// Library does not cover quaternions.
//
// Dependences:
//   opencv_core
//
// Notes:
//   everything is wrapped in namespace 'evg' for convenience
//   everything is float (common practice for 3D)
//   error policy: exceptions are promoted, asserts are used
//   links to algorithms are given in the .cpp file
//
// Common Notations:
//   R:     camera rotation matrix is [3 x 3] matrix of type  cv::Matx33f
//

namespace cv {
namespace evg {

    // rotation around axes
    cv::Matx33f     rotAxisX (const float angle);
    cv::Matx33f     rotAxisY (const float angle);
    cv::Matx33f     rotAxisZ (const float angle);
    
    
    
    //
    // from euler angles to rotation matrix and back
    //   just a few jet-to-axes orientation are implemented, hope you use one of them
    //
    
    // XY. axes: roll - X, pitch - Y, yaw - Z (jet is moving in XY plane along X)
    cv::Matx33f     euler2R_XY (const float yaw, const float pitch, const float roll);
    void            R2euler_XY (const cv::Matx33f& R, float& yaw, float& pitch, float& roll);
    
    // XZ. axes: roll - X, pitch - Z, yaw - Y (jet is moving in XZ plane along X)
    cv::Matx33f     euler2R_XZ (const float yaw, const float pitch, const float roll);

    // ZX. axes: roll - Z, pitch - X, yaw - Y (jet is moving in ZX plane along Z)
    cv::Matx33f     euler2R_ZX (const float yaw, const float pitch, const float roll);
    


    // Use sampleR() to get uniformly distributed (across SO3 space) rotations.
    //   {u1,u2,u3} should be random numbers from uniform distribution on [0 1)
    cv::Matx33f     sampleR (const float u1, const float u2, const float u3);
    
    
} // namespace evg
} // namespace cv

#endif // EVG_ANGLES_3D
