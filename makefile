#############################################################################
# Makefile for building: DrowsyDriverQt.app/Contents/MacOS/DrowsyDriverQt
# Generated by qmake (2.01a) (Qt 4.8.6) on: Wed Apr 2 21:06:19 2014
# Project:  ../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverQt.pro
# Template: app
# Command: /usr/local/bin/qmake -o makefile ../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverQt.pro
#############################################################################

####### Compiler, tools and options

CC            = clang
CXX           = clang++
DEFINES       = -DDEFAULT_VIDEO_PATH=\"/Users/evgenytoropov/Documents/Xcode/DrowsyDriver/data/6.mpg\" -DKYLE_MODEL_PATH=\"/Users/evgenytoropov/Documents/Xcode/DrowsyDriver/DrowsyDriver_2.2/FaceTracker\" -DOPENCV_DATA_PATH=\"/usr/local/share/OpenCV\" -DCLASSIFIER_FACE_PATH=\"/Users/evgenytoropov/Documents/Xcode/DrowsyDriver/DrowsyDriver_2.2/classifier_model\" -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED
CFLAGS        = -pipe -mmacosx-version-min=10.7 -O2 -arch x86_64 -Wall -W $(DEFINES)
CXXFLAGS      = -pipe -stdlib=libc++ -mmacosx-version-min=10.7 -fpermissive -O2 -arch x86_64 -Wall -W $(DEFINES)
INCPATH       = -I/usr/local/Cellar/qt/4.8.5/mkspecs/unsupported/macx-clang-libc++ -I../../DrowsyDriver/DrowsyDriver_2.2rel -I/usr/local/Cellar/qt/4.8.5/lib/QtCore.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtCore.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtGui.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtGui.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/include -I../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include -I. -I. -I../../DrowsyDriver/DrowsyDriver_2.2rel -I. -F/usr/local/Cellar/qt/4.8.5/lib
LINK          = clang++
LFLAGS        = -headerpad_max_install_names -stdlib=libc++ -mmacosx-version-min=10.7 -arch x86_64
LIBS          = $(SUBLIBS) -F/usr/local/Cellar/qt/4.8.5/lib -L/usr/local/Cellar/qt/4.8.5/lib -lopencv_core -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_imgproc -lopencv_video -lopencv_flann -lboost_thread -lboost_filesystem -lboost_system -lvl -framework QtGui -L/opt/X11/lib -L/usr/local/Cellar/qt/4.8.5/lib -F/usr/local/Cellar/qt/4.8.5/lib -framework QtCore 
AR            = ar cq
RANLIB        = ranlib -s
QMAKE         = /usr/local/bin/qmake
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = cp -f
COPY_DIR      = cp -f -R
STRIP         = 
INSTALL_FILE  = $(COPY_FILE)
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = $(COPY_FILE)
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p
export MACOSX_DEPLOYMENT_TARGET = 10.7

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = ../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Tracker.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PDM.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PAW.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Patch.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/IO.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FDet.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FCheck.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/CLM.cc \
		../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.cpp \
		../../DrowsyDriver/DrowsyDriver_2.2rel/QtMain.cpp moc_QtMainwindow.cpp
OBJECTS       = CVmanager.o \
		Trigger.o \
		TrackerCompressive.o \
		TrackerOpticalFlow.o \
		TrackerCamshift.o \
		TrackerKyle.o \
		DetectorOpencv.o \
		DetectorKyleFace.o \
		Common.o \
		cluster_computation.o \
		feature_extractor.o \
		svm_classifier_face.o \
		histogram_generator_face.o \
		classifier_interface_face.o \
		Recognizer.o \
		classifier_interface_eyes.o \
		histogram_generator_eyes.o \
		svm_classifier_eyes.o \
		KyleGeneric.o \
		timers.o \
		mediaLoadSave.o \
		rectOperations.o \
		Tracker.o \
		PDM.o \
		PAW.o \
		Patch.o \
		IO.o \
		FDet.o \
		FCheck.o \
		CLM.o \
		QtMainwindow.o \
		QtMain.o \
		moc_QtMainwindow.o
DIST          = /usr/local/Cellar/qt/4.8.5/mkspecs/common/unix.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/mac.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/gcc-base.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/gcc-base-macx.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/clang.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/qconfig.pri \
		/usr/local/Cellar/qt/4.8.5/mkspecs/modules/qt_webkit_version.pri \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt_functions.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt_config.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/exclusive_builds.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/default_pre.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/default_pre.prf \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverCore.pri \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/release.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/default_post.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/default_post.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/x86_64.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/objective_c.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/shared.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/warn_on.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/unix/thread.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/moc.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/rez.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/sdk.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/resources.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/uic.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/yacc.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/lex.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/include_source_dir.prf \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverQt.pro
QMAKE_TARGET  = DrowsyDriverQt
DESTDIR       = 
TARGET        = DrowsyDriverQt.app/Contents/MacOS/DrowsyDriverQt

####### Custom Compiler Variables
QMAKE_COMP_QMAKE_OBJECTIVE_CFLAGS = -pipe \
		-O2 \
		-arch \
		x86_64 \
		-Wall \
		-W


first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: makefile DrowsyDriverQt.app/Contents/PkgInfo DrowsyDriverQt.app/Contents/Resources/empty.lproj DrowsyDriverQt.app/Contents/Info.plist $(TARGET)

$(TARGET): ui_mainwindow.h $(OBJECTS)  
	@$(CHK_DIR_EXISTS) DrowsyDriverQt.app/Contents/MacOS/ || $(MKDIR) DrowsyDriverQt.app/Contents/MacOS/ 
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

makefile: ../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverQt.pro  /usr/local/Cellar/qt/4.8.5/mkspecs/unsupported/macx-clang-libc++/qmake.conf /usr/local/Cellar/qt/4.8.5/mkspecs/common/unix.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/mac.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/gcc-base.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/gcc-base-macx.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/common/clang.conf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/qconfig.pri \
		/usr/local/Cellar/qt/4.8.5/mkspecs/modules/qt_webkit_version.pri \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt_functions.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt_config.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/exclusive_builds.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/default_pre.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/default_pre.prf \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverCore.pri \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/release.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/default_post.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/default_post.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/x86_64.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/objective_c.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/shared.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/warn_on.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/unix/thread.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/moc.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/rez.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/sdk.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/resources.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/uic.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/yacc.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/lex.prf \
		/usr/local/Cellar/qt/4.8.5/mkspecs/features/include_source_dir.prf \
		/usr/local/Cellar/qt/4.8.5/lib/QtGui.framework/QtGui.prl \
		/usr/local/Cellar/qt/4.8.5/lib/QtCore.framework/QtCore.prl
	$(QMAKE) -o makefile ../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverQt.pro
/usr/local/Cellar/qt/4.8.5/mkspecs/common/unix.conf:
/usr/local/Cellar/qt/4.8.5/mkspecs/common/mac.conf:
/usr/local/Cellar/qt/4.8.5/mkspecs/common/gcc-base.conf:
/usr/local/Cellar/qt/4.8.5/mkspecs/common/gcc-base-macx.conf:
/usr/local/Cellar/qt/4.8.5/mkspecs/common/clang.conf:
/usr/local/Cellar/qt/4.8.5/mkspecs/qconfig.pri:
/usr/local/Cellar/qt/4.8.5/mkspecs/modules/qt_webkit_version.pri:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt_functions.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt_config.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/exclusive_builds.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/default_pre.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/default_pre.prf:
../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverCore.pri:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/release.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/default_post.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/default_post.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/x86_64.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/objective_c.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/shared.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/warn_on.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/qt.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/unix/thread.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/moc.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/rez.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/mac/sdk.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/resources.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/uic.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/yacc.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/lex.prf:
/usr/local/Cellar/qt/4.8.5/mkspecs/features/include_source_dir.prf:
/usr/local/Cellar/qt/4.8.5/lib/QtGui.framework/QtGui.prl:
/usr/local/Cellar/qt/4.8.5/lib/QtCore.framework/QtCore.prl:
qmake:  FORCE
	@$(QMAKE) -o makefile ../../DrowsyDriver/DrowsyDriver_2.2rel/DrowsyDriverQt.pro

DrowsyDriverQt.app/Contents/PkgInfo: 
	@$(CHK_DIR_EXISTS) DrowsyDriverQt.app/Contents || $(MKDIR) DrowsyDriverQt.app/Contents 
	@$(DEL_FILE) DrowsyDriverQt.app/Contents/PkgInfo
	@echo "APPL????" >DrowsyDriverQt.app/Contents/PkgInfo
DrowsyDriverQt.app/Contents/Resources/empty.lproj: 
	@$(CHK_DIR_EXISTS) DrowsyDriverQt.app/Contents/Resources || $(MKDIR) DrowsyDriverQt.app/Contents/Resources 
	@touch DrowsyDriverQt.app/Contents/Resources/empty.lproj
	
DrowsyDriverQt.app/Contents/Info.plist: 
	@$(CHK_DIR_EXISTS) DrowsyDriverQt.app/Contents || $(MKDIR) DrowsyDriverQt.app/Contents 
	@$(DEL_FILE) DrowsyDriverQt.app/Contents/Info.plist
	@sed -e "s,@SHORT_VERSION@,1.0,g" -e "s,@TYPEINFO@,????,g" -e "s,@ICON@,,g" -e "s,@EXECUTABLE@,DrowsyDriverQt,g" -e "s,@TYPEINFO@,????,g" /usr/local/Cellar/qt/4.8.5/mkspecs/unsupported/macx-clang-libc++/Info.plist.app >DrowsyDriverQt.app/Contents/Info.plist
dist: 
	@$(CHK_DIR_EXISTS) .tmp/DrowsyDriverQt1.0.0 || $(MKDIR) .tmp/DrowsyDriverQt1.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) .tmp/DrowsyDriverQt1.0.0/ && $(COPY_FILE) --parents ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackingManager.h ../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.h ../../DrowsyDriver/DrowsyDriver_2.2rel/TriggerManager.h ../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.h ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.h ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.h ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.h ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.h ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectingManager.h ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.h ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.h ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h ../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h ../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h ../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.h ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.h ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/PDM.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/PAW.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Patch.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/IO.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/FDet.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/FCheck.h ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/CLM.h ../../DrowsyDriver/DrowsyDriver_2.2rel/QtOutStream.h ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.h .tmp/DrowsyDriverQt1.0.0/ && $(COPY_FILE) --parents ../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Common.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Tracker.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PDM.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PAW.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Patch.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/IO.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FDet.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FCheck.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/CLM.cc ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMain.cpp .tmp/DrowsyDriverQt1.0.0/ && $(COPY_FILE) --parents ../../DrowsyDriver/DrowsyDriver_2.2rel/mainwindow.ui .tmp/DrowsyDriverQt1.0.0/ && (cd `dirname .tmp/DrowsyDriverQt1.0.0` && $(TAR) DrowsyDriverQt1.0.0.tar DrowsyDriverQt1.0.0 && $(COMPRESS) DrowsyDriverQt1.0.0.tar) && $(MOVE) `dirname .tmp/DrowsyDriverQt1.0.0`/DrowsyDriverQt1.0.0.tar.gz . && $(DEL_FILE) -r .tmp/DrowsyDriverQt1.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) -r DrowsyDriverQt.app
	-$(DEL_FILE) makefile


check: first

mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

compiler_objective_c_make_all:
compiler_objective_c_clean:
compiler_moc_header_make_all: moc_QtMainwindow.cpp
compiler_moc_header_clean:
	-$(DEL_FILE) moc_QtMainwindow.cpp
moc_QtMainwindow.cpp: ../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TriggerManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/QtOutStream.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.h
	/usr/local/Cellar/qt/4.8.5/bin/moc $(DEFINES) $(INCPATH) -D__APPLE__ -D__GNUC__ ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.h -o moc_QtMainwindow.cpp

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_image_collection_make_all: qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) qmake_image_collection.cpp
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_rez_source_make_all:
compiler_rez_source_clean:
compiler_uic_make_all: ui_mainwindow.h
compiler_uic_clean:
	-$(DEL_FILE) ui_mainwindow.h
ui_mainwindow.h: ../../DrowsyDriver/DrowsyDriver_2.2rel/mainwindow.ui
	/usr/local/Cellar/qt/4.8.5/bin/uic ../../DrowsyDriver/DrowsyDriver_2.2rel/mainwindow.ui -o ui_mainwindow.h

compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: compiler_moc_header_clean compiler_uic_clean 

####### Compile

CVmanager.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TriggerManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o CVmanager.o ../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.cpp

Trigger.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Trigger.o ../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.cpp

TrackerCompressive.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o TrackerCompressive.o ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.cpp

TrackerOpticalFlow.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o TrackerOpticalFlow.o ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.cpp

TrackerCamshift.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o TrackerCamshift.o ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.cpp

TrackerKyle.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o TrackerKyle.o ../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.cpp

DetectorOpencv.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o DetectorOpencv.o ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.cpp

DetectorKyleFace.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o DetectorKyleFace.o ../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.cpp

Common.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/Common.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Common.o ../../DrowsyDriver/DrowsyDriver_2.2rel/Common.cpp

cluster_computation.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o cluster_computation.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.cpp

feature_extractor.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o feature_extractor.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.cpp

svm_classifier_face.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o svm_classifier_face.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.cpp

histogram_generator_face.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o histogram_generator_face.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.cpp

classifier_interface_face.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o classifier_interface_face.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.cpp

Recognizer.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Recognizer.o ../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.cpp

classifier_interface_eyes.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o classifier_interface_eyes.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.cpp

histogram_generator_eyes.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o histogram_generator_eyes.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.cpp

svm_classifier_eyes.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o svm_classifier_eyes.o ../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.cpp

KyleGeneric.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o KyleGeneric.o ../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.cpp

timers.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o timers.o ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.cpp

mediaLoadSave.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o mediaLoadSave.o ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.cpp

rectOperations.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o rectOperations.o ../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/rectOperations.cpp

Tracker.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Tracker.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Tracker.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Tracker.cc

PDM.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PDM.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o PDM.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PDM.cc

PAW.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PAW.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o PAW.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/PAW.cc

Patch.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Patch.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Patch.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/Patch.cc

IO.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/IO.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o IO.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/IO.cc

FDet.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FDet.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o FDet.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FDet.cc

FCheck.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FCheck.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o FCheck.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/FCheck.cc

CLM.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/CLM.cc 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o CLM.o ../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/src/lib/CLM.cc

QtMainwindow.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TriggerManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/QtOutStream.h \
		ui_mainwindow.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o QtMainwindow.o ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.cpp

QtMain.o: ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMain.cpp ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMainwindow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/CVmanager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Common.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorOpencv.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/DetectorKyleFace.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/KyleGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/FaceTracker/include/FaceTracker/Tracker.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackingManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerGeneric.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCompressive.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerOpticalFlow.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerCamshift.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TrackerKyle.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/TriggerManager.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Trigger.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/Recognizer.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/classifier_interface_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/svm_classifier_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/cluster_computation.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/feature_extractor.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_face/histogram_generator_face.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/classifier_interface_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/svm_classifier_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/classifier_eyes/histogram_generator_eyes.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/timers.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/evgUtilities/mediaLoadSave.h \
		../../DrowsyDriver/DrowsyDriver_2.2rel/QtOutStream.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o QtMain.o ../../DrowsyDriver/DrowsyDriver_2.2rel/QtMain.cpp

moc_QtMainwindow.o: moc_QtMainwindow.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_QtMainwindow.o moc_QtMainwindow.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:
