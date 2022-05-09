#ifndef GREYSCALE_HPP
#define GREYSCALE_HPP

#include <string>


#include "CImg.h"

class GreyScale{
    private:
        cimg_library::CImg<u_char> img;
        std::string filePath;
        std::string fileName;
        size_t gridSize;
        size_t blockSize;

    public:
        GreyScale();
        GreyScale(std::string filePath, std::string fileName);
        void loadFile();
        void makeGreyScaleSerial();
        void makeGreyScaleParallel(int numBlocks, int numGrids);
        void display();
        void writeFile();
};

// kernel functions must be global
__global__ void mykernel(u_char* imageArr, int* devImageDims);

#endif // GREYSCALE_HPP