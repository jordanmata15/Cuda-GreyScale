#include "GreyScale.hpp"
#include <iostream>


GreyScale::GreyScale(){
    std::filesystem::path dir ("..");
    this->filePath = dir / "data";
    this->fileName = "dino.jpg";
}

GreyScale::GreyScale(std::string filePath, std::string fileName){
    this->filePath = filePath;
    this->fileName = fileName;
}

void GreyScale::loadFile(){
    this->image = cv::imread(this->filePath / this->fileName);
    if (this->image.empty()){
        // TODO Handle errors
    }
}

void GreyScale::makeGreyScaleSerial(){
    cv::Mat image = this->image;
    //int nRows = image.rows;
    //int nCols = image.cols;

    uchar * arr = image.isContinuous()? image.data: image.clone().data;
    uint length = image.total()*image.channels();

    for (size_t i=0; i<length; i+=3){
        uchar R = arr[i+2];
        uchar G = arr[i+1];
        uchar B = arr[i];
        arr[i] = R*0.3 + G*0.59 + B*0.11;
        arr[i+1] = arr[i];
        arr[i+2] = arr[i];
    }
}

void GreyScale::displayResult(){
    cv::imshow("Greyscale " + this->fileName, image);
    cv::waitKey(0);
}

void GreyScale::writeFile(){
    std::string outputFileName = "GreyScale_" + fileName;
    std::filesystem::path outFilePath = this->filePath / outputFileName;
    cv::imwrite(outFilePath, this->image);
}