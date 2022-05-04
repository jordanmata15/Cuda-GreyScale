#ifndef GREYSCALE_HPP
#define GREYSCALE_HPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <filesystem>

class GreyScale{
    private:
        cv::Mat image;
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