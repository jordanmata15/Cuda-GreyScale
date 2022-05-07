#ifndef GREYSCALE_HPP
#define GREYSCALE_HPP

#include <string>
#include <filesystem>

#include "CImg.h"

class GreyScale{
    private:
        cimg_library::CImg<u_char> img;
        std::filesystem::path filePath;
        std::string fileName;

    public:
        GreyScale();
        GreyScale(std::string filePath, std::string fileName);
        void loadFile();
        void makeGreyScaleSerial();
        void displayResult();
        void writeFile();
};

#endif // GREYSCALE_HPP