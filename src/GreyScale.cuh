#ifndef GREYSCALE_HPP
#define GREYSCALE_HPP

#include <string>
//#include <filesystem>

#include "CImg.h"

class GreyScale{
    private:
        cimg_library::CImg<u_char> img;
        //std::filesystem::path filePath;
        std::string filePath;
        std::string fileName;
        size_t gridSize;
        size_t blockSize;

    public:
        GreyScale();
        GreyScale(std::string filePath, std::string fileName);
        void loadFile();
        void makeGreyScaleSerial();
        void makeGreyScaleParallel();
        void display();
        void writeFile();
};

// kernel functions must be global
__global__ void mykernel(u_char* imageArr, int* rgbOffset);

#endif // GREYSCALE_HPP