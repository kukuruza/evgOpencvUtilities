# Paths

LIB_COMMON   = -L/usr/local/lib/ -lboost_system -lboost_filesystem -lboost_timer -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_nonfree -lopencv_video -lopencv_flann -lopencv_calib3d -lopencv_highgui
INCLUDES     = -I/usr/local/include/ -I include/
CPPFLAGS     = -O2


all: build/matchImages build/undistortImage build/undistortVideo build/angles3D build/geometry3D build/mediaIO build/featuresIO



build/angles3D: build/angles3D.o
	g++ -fPIC -shared -g -o build/angles3D.so build/angles3D.o $(LIB_COMMON)

build/geometry3D: build/geometry3D.o build/angles3D.o
	g++ -fPIC -shared -g -o build/geometry3D.so build/geometry3D.o build/angles3D.o $(LIB_COMMON)

build/mediaIO: build/mediaIO.o
	g++ -fPIC -shared -g -o build/mediaIO.so build/mediaIO.o $(LIB_COMMON)

build/featuresIO: build/featuresIO.o build/mediaIO.o
	g++ -fPIC -shared -g -o build/featuresIO.so build/featuresIO.o build/mediaIO.o $(LIB_COMMON)

build/matchImages: build/matchImages.o build/mediaIO.o build/featuresIO.o
	g++ -o build/matchImages build/matchImages.o build/mediaIO.o build/featuresIO.o $(LIB_COMMON)

build/undistortImage: build/undistortImage.o build/mediaIO.o
	g++ -o build/undistortImage build/undistortImage.o build/mediaIO.o $(LIB_COMMON)

build/undistortVideo: build/undistortVideo.o build/mediaIO.o
	g++ -o build/undistortVideo build/undistortVideo.o build/mediaIO.o $(LIB_COMMON)





build/angles3D.o: src/angles3D.cpp include/angles3D.h
	g++ -fPIC -c -o build/angles3D.o $(CPPFLAGS) $(INCLUDES) src/angles3D.cpp

build/geometry3D.o: src/geometry3D.cpp include/geometry3D.h
	g++ -fPIC -c -o build/geometry3D.o $(CPPFLAGS) $(INCLUDES) src/geometry3D.cpp

build/mediaIO.o: src/mediaIO.cpp include/mediaIO.h
	g++ -fPIC -c -o build/mediaIO.o $(CPPFLAGS) $(INCLUDES) src/mediaIO.cpp 

build/featuresIO.o: src/featuresIO.cpp include/featuresIO.h
	g++ -fPIC -c -o build/featuresIO.o $(CPPFLAGS) $(INCLUDES) src/featuresIO.cpp 

build/matchImages.o: matchImages/matchImages.cpp include/featuresIO.h include/mediaIO.h
	g++ -fPIC -c -o build/matchImages.o $(CPPFLAGS) $(INCLUDES) matchImages/matchImages.cpp

build/undistortImage.o: undistortImage/undistortImage.cpp include/mediaIO.h
	g++ -fPIC -c -o build/undistortImage.o $(CPPFLAGS) $(INCLUDES) undistortImage/undistortImage.cpp

build/undistortVideo.o: undistortVideo/undistortVideo.cpp include/mediaIO.h
	g++ -fPIC -c -o build/undistortVideo.o $(CPPFLAGS) $(INCLUDES) undistortVideo/undistortVideo.cpp



clean:
	rm -rf build/*



install:
	mkdir -p                ~/local/include/evgOpencvUtilities
	cp include/*            ~/local/include/evgOpencvUtilities/
	cp build/angles3D.so    ~/local/lib/
	cp build/geometry3D.so  ~/local/lib/
	cp build/mediaIO.so     ~/local/lib/
	cp build/featuresIO.so  ~/local/lib/
	cp build/matchImages    ~/local/bin/
	cp build/undistortImage ~/local/bin/
	cp build/undistortVideo ~/local/bin/
