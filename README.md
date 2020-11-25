# creative-face  

## About the project  
Automatically detect a face, crop it, create multiple random filters and then keep showing them all for a couple of seconds. After that, repeat.  
Build in C++ and OpenCV. Free to use, but code is not clean and well designed.
![logo](src/logo.png)

## Build OpenCV (from official [tutorial](https://docs.opencv.org/4.5.0/d7/d9f/tutorial_linux_install.html))

``` bash
# install g++, cmake and make
sudo apt install g++ cmake make

# download OpenCV 4.5 into folder opencv
cd opencv
mkdir build && cd build

# build
cmake -D ENABLE_PRECOMPILED_HEADERS=OFF -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_GENERATE_PKGCONFIG=ON ..
make -j8

# install
sudo make install
```

## Build creative-face

``` bash
# download creative-face into folder creative-face
cd creative-face
mkdir build && cd build

# build
cmake ..
make
```

## Run creative-face

``` bash
./creative-face [-width=640] [-height=480] [-cameraWidth=640] [-cameraHeight=480] [-col=2] [-row=2] [-time=3]
```
>- width: width of the canvas (default: 640)
>- height: height of the canvas (default: 480)
>- cameraWidth: width of image captured by camera (default: 640)
>- cameraHeight: height of image captured by camera (default: 480)
>- col: number of filtered images in colum (default: 2)
>- row: number of filtered images in row (default: 2)
>- time: time in seconds between the images (default: 3)
>- faceFactor: face detection area multiplicator (default: 0.4)