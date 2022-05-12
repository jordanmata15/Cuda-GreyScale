#ifndef GREYSCALE_CUH
#define GREYSCALE_CUH

#include <string>
#include "CImg.h"
#include "DataManager.cuh"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
static const std::string fileSeparator="\\";
#else
static const std::string fileSeparator="/";
#endif

class GreyScale{
    private:
        cimg_library::CImg<u_char> img;
        std::string filePath;
        std::string fileName;
        size_t gridSize;
        size_t blockSize;
        DataManager dManager;

    public:
        GreyScale();
        void loadFile();
        void loadFile(std::string filePath, std::string fileName);
        void makeGreyScaleSerial();
        void makeGreyScaleParallel(int numBlocks);
        void display();
        void writeFile();
        double getTimeElapsed();
};

// kernel functions must be global
__global__ void mykernel(u_char* imageArr, int* devImageDims);

#endif // GREYSCALE_CUH