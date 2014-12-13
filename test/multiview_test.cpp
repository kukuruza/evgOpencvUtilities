#include "gtest/gtest.h"
#include "../src/multiview.h"
#include "../src/mediaIO.h"


using namespace std;
using namespace cv;

namespace {

class MultiviewTest : public ::testing::Test {
protected:
  const string data_dir = "multiviewData/";
  const string matches123_clear_name = "matches123-clear.txt";
  const string matches12_clear_name = "matches12-clear.txt";
  const string matches23_clear_name = "matches23-clear.txt";
  const string matches31_clear_name = "matches31-clear.txt";
  const string matches123_noisy_name = "matches123-noisy.txt";
  const string matches12_noisy_name = "matches12-noisy.txt";
  const string matches23_noisy_name = "matches23-noisy.txt";
  const string matches31_noisy_name = "matches31-noisy.txt";
  const string pose2_name = "pose2.txt";
  const string pose3_name = "pose3.txt";
  
  Mat matches123_clear, matches123_noisy;
  Mat matches12_clear, matches12_noisy;
  Mat matches23_clear, matches23_noisy;
  Mat matches31_clear, matches31_noisy;
  
  Matx44f pose2, pose3;

  MultiviewTest() {
    matches123_clear = evg::dlmread(data_dir + matches123_clear_name);
    matches12_clear = evg::dlmread(data_dir + matches12_clear_name);
    matches23_clear = evg::dlmread(data_dir + matches23_clear_name);
    matches31_clear = evg::dlmread(data_dir + matches31_clear_name);
    matches123_noisy = evg::dlmread(data_dir + matches123_noisy_name);
    matches12_noisy = evg::dlmread(data_dir + matches12_noisy_name);
    matches23_noisy = evg::dlmread(data_dir + matches23_noisy_name);
    matches31_noisy = evg::dlmread(data_dir + matches31_noisy_name);
    pose2 = evg::dlmread(data_dir + pose2_name);
    pose3 = evg::dlmread(data_dir + pose3_name);
  }
  
  ~MultiviewTest() { }
  virtual void SetUp() { }
  virtual void TearDown() { }
};


TEST_F (MultiviewTest, Epipolar) {
    for (int j = 0; j != matches12_clear.rows; ++j)
    {
        Matx14f r (matches12_clear.row(j));
        float error = evg::epipolarError(pose2, Matx31f(r(0),r(1),1), Matx31f(r(2),r(3),1));
        cout << "clear, j: " << j << ", epipolar error: " << error << endl;
        ASSERT_TRUE (error >= 0);         // error should be positive
        ASSERT_TRUE (error < 1.f);        // error should be small (still big, even for clear data)
    }
    
    for (int j = 0; j != matches12_noisy.rows; ++j)
    {
        Matx14f r (matches12_noisy.row(j));
        float error = evg::epipolarError(pose2, Matx31f(r(0),r(1),1), Matx31f(r(2),r(3),1));
        cout << "noisy, j: " << j << ", epipolar error: " << error << endl;
        ASSERT_TRUE (error >= 0);        // error should be positive
        ASSERT_TRUE (error < 1.f);       // error should be small
    }
}


TEST_F (MultiviewTest, TwoView) {
    for (int j = 0; j != matches12_clear.rows; ++j)
    {
        Matx14f r (matches12_clear.row(j));
        float error = evg::twoViewError(pose2, Matx31f(r(0),r(1),1), Matx31f(r(2),r(3),1));
        cout << "clear, j: " << j << ", two-view error: " << error << endl;
        ASSERT_TRUE (error >= 0);         // error should be positive
        ASSERT_TRUE (error < 0.00001f);   // error should _very_ close to 0
    }

    for (int j = 0; j != matches12_noisy.rows; ++j)
    {
        Matx14f r (matches12_noisy.row(j));
        float error = evg::twoViewError(pose2, Matx31f(r(0),r(1),1), Matx31f(r(2),r(3),1));
        cout << "noisy, j: " << j << ", two-view error: " << error << endl;
        ASSERT_TRUE (error >= 0);         // error should be positive
        ASSERT_TRUE (error < 0.1f);       // error should _very_ close to 0
    }
}


TEST_F (MultiviewTest, ThreeView) {
    for (int j = 0; j != matches123_clear.rows; ++j)
    {
        Matx16f r (matches123_clear.row(j));
        vector<float> errors = evg::threeViewErrors(pose2, pose3, Matx31f(r(0),r(1),1),
                                                    Matx31f(r(2),r(3),1), Matx31f(r(4),r(5),1));
        cout << "clear, j: " << j << ", three-view error: " << errors[0]
             << " " << errors[1] << " " << errors[2] << endl;
        ASSERT_TRUE (std::abs(errors[0]) < 0.0001f);   // error should _very_ close to 0
        ASSERT_TRUE (std::abs(errors[1]) < 0.0001f);   // error should _very_ close to 0
        ASSERT_TRUE (std::abs(errors[2]) < 0.0001f);   // error should _very_ close to 0
    }

    for (int j = 0; j != matches123_noisy.rows; ++j)
    {
        Matx16f r (matches123_noisy.row(j));
        vector<float> errors = evg::threeViewErrors(pose2, pose3, Matx31f(r(0),r(1),1),
                                                    Matx31f(r(2),r(3),1), Matx31f(r(4),r(5),1));
        cout << "noisy, j: " << j << ", three-view error: " << errors[0]
             << " " << errors[1] << " " << errors[2] << endl;
        ASSERT_TRUE (std::abs(errors[0]) < 0.5f);      // error should be small
        ASSERT_TRUE (std::abs(errors[1]) < 0.5f);      // error should be small
        ASSERT_TRUE (std::abs(errors[2]) < 0.5f);      // error should be small
    }
}




}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}