#include "GreyScale.hpp"
#include <iostream>
#include <cstring>

#include <cstdio>
#include <jpeglib.h>
#include <jerror.h>

using namespace cimg_library;

GreyScale::GreyScale(){
    std::filesystem::path dir ("..");
    this->filePath = dir / "data";
    this->fileName = "dino.bmp";
}

GreyScale::GreyScale(std::string filePath, std::string fileName){
    this->filePath = filePath;
    this->fileName = fileName;
}

void GreyScale::loadFile(){
    /**this->image = cv::imread(this->filePath / this->fileName);
    if (this->image.empty()){
        // TODO Handle errors
    }*/

    std::string fullFilePath = this->filePath / this->fileName;
    const char * filename_input = fullFilePath.c_str();
    this->img = CImg<u_char>(filename_input);
}

void GreyScale::makeGreyScaleSerial(){
    /*cv::Mat image = this->image;
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
    }*/
    
    int width = this->img.width(),
        height = this->img.height(), 
        depth = this->img.depth();

    std::cout << width << std::endl << height << std::endl << depth << std::endl;

    /* pointer to image pixels. Colors are rranged in contiguous memory. 
    Eg: {RRR...RRRGGG...GGGBBB...BBB} */
    auto arr = this->img.data(); 
    size_t length = width*height*depth;
    size_t rgbOffset = length;

    for (size_t i=0; i<length; i+=1){
        u_char R = arr[i+2*rgbOffset];
        u_char G = arr[i+1*rgbOffset];
        u_char B = arr[i];
        arr[i] = R*0.3 + G*0.59 + B*0.11;
        arr[i+1*rgbOffset] = arr[i];
        arr[i+2*rgbOffset] = arr[i];
    }
}

void GreyScale::displayResult(){
    this->img.display();
}

void GreyScale::writeFile(){
    std::string outputFileName = "GreyScale_" + fileName;
    std::filesystem::path outFilePath = this->filePath / outputFileName;
    const char * filename_output = outFilePath.c_str();
    this->img.save(filename_output);
}