cmake_minimum_required(VERSION 2.8.4)
project(evgOpencvUtilities)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set( Boost_USE_STATIC_LIBS OFF )
set( Boost_USE_STATIC_RUNTIME OFF )
find_package( Boost REQUIRED COMPONENTS system filesystem )
find_package( OpenCV REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_features2d opencv_nonfree opencv_video opencv_flann opencv_calib3d opencv_highgui)

set(TCLAP_HEADERS
    src/tclap/Arg.h
    src/tclap/ArgException.h
    src/tclap/ArgTraits.h
    src/tclap/CmdLine.h
    src/tclap/CmdLineInterface.h
    src/tclap/CmdLineOutput.h
    src/tclap/Constraint.h
    src/tclap/DocBookOutput.h
    src/tclap/HelpVisitor.h
    src/tclap/IgnoreRestVisitor.h
    src/tclap/MultiArg.h
    src/tclap/MultiSwitchArg.h
    src/tclap/OptionalUnlabeledTracker.h
    src/tclap/StandardTraits.h
    src/tclap/StdOutput.h
    src/tclap/SwitchArg.h
    src/tclap/UnlabeledMultiArg.h
    src/tclap/UnlabeledValueArg.h
    src/tclap/ValueArg.h
    src/tclap/ValuesConstraint.h
    src/tclap/VersionVisitor.h
    src/tclap/Visitor.h
    src/tclap/XorHandler.h
    src/tclap/ZshCompletionOutput.h)

set(
    matchImages/matchImages.cpp
    test/test_featuresIO.cpp
    undistortImage/undistortImage.cpp
    undistortVideo/undistortVideo.cpp)

set(LIB_SOURCE_FILES
    src/angles3D.cpp
    src/angles3D.h
    src/featuresIO.cpp
    src/featuresIO.h
    src/geometry3D.cpp
    src/geometry3D.h
    src/mediaIO.cpp
    src/mediaIO.h
    src/multiview.cpp
    src/multiview.h
    src/miscellaneous.h)

add_library(evgOpencvUtilities ${LIB_SOURCE_FILES})

add_executable(matchImages src/matchImages.cpp)
add_executable(undistortImage src/undistortImage.cpp)
add_executable(undistortVideo src/undistortVideo.cpp)
add_executable(test_featuresIO test/test_featuresIO.cpp)

target_link_libraries(matchImages evgOpencvUtilities ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )
target_link_libraries(undistortImage evgOpencvUtilities ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )
target_link_libraries(undistortVideo evgOpencvUtilities ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )
target_link_libraries(test_featuresIO evgOpencvUtilities ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} opencv_ts)

install(TARGETS matchImages undistortImage undistortVideo DESTINATION "bin")
install(TARGETS evgOpencvUtilities DESTINATION "lib")
