#include "GreyScale.cuh"
#include <iostream>
#include <cstring>

#include <cstdio>
#include <jpeglib.h>
#include <jerror.h>

using namespace cimg_library;


GreyScale::GreyScale(){
    //std::filesystem::path dir ("..");
    //this->filePath = dir / "data";
    this->filePath = "../data";
    this->fileName = "dino.bmp";
}


GreyScale::GreyScale(std::string filePath, std::string fileName){
    this->filePath = filePath;
    this->fileName = fileName;
}


void GreyScale::loadFile(){
    //std::string fullFilePath = this->filePath / this->fileName;
    std::string fullFilePath = this->filePath + "/" + this->fileName;
    const char* filename_input = fullFilePath.c_str();
    this->img = CImg<u_char>(filename_input);
}


void GreyScale::makeGreyScaleSerial(){
    int width  = this->img.width(),
        height = this->img.height(), 
        depth  = this->img.depth();

    /* pointer to image pixels. Colors are rranged in contiguous memory. 
    Eg: {RRR...RRRGGG...GGGBBB...BBB} */
    auto arr = this->img.data(); 
    size_t length = width*height*depth;
    size_t rgbOffset = length;
    // pointers to the start of each of the color channels
    u_char* R = &arr[0*rgbOffset];
    u_char* G = &arr[1*rgbOffset];
    u_char* B = &arr[2*rgbOffset];

    for (size_t i=0; i<length; ++i){
        u_char greyScaleValue = R[i]*0.3 + G[i]*0.59 + B[i]*0.11;
        R[i] = greyScaleValue;
        G[i] = greyScaleValue;
        B[i] = greyScaleValue;
    }
}


void GreyScale::makeGreyScaleParallel(){
    /* pointer to image pixels. Colors are rranged in contiguous memory. 
    Eg: {RRR...RRRGGG...GGGBBB...BBB} */
    u_char *hostImageArr = this->img.data();
    u_char *devImageArr;
    int *devRGBOffset;

    size_t  channels = 3,
            length = this->img.width() * this->img.height() * this->img.depth() * channels,
            rgbOffset = length/channels;

    cudaMalloc((void**)&devImageArr, length*sizeof(u_char));
    cudaMalloc((void**)&devRGBOffset, sizeof(int));
    cudaMemcpy(devImageArr, hostImageArr, length*sizeof(u_char), cudaMemcpyHostToDevice);
    cudaMemcpy(devRGBOffset, &rgbOffset, sizeof(int), cudaMemcpyHostToDevice);

    mykernel<<<this->blockSize,this->gridSize>>>(devImageArr, devRGBOffset);

    cudaMemcpy(hostImageArr, devImageArr, length*sizeof(u_char), cudaMemcpyDeviceToHost);
    cudaFree(devImageArr);
    cudaFree(devRGBOffset);
}


__global__ void mykernel(u_char* imageArr, int* devRGBOffset){
    int rgbOffset = *devRGBOffset;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    u_char  *R = &imageArr[0*rgbOffset],
            *G = &imageArr[1*rgbOffset],
            *B = &imageArr[2*rgbOffset],
            greyScaleValue;

    if (index < rgbOffset){
        greyScaleValue = R[index]*0.3 + G[index]*0.59 + B[index]*0.11;
        R[index] = greyScaleValue;
        G[index] = greyScaleValue;
        B[index] = greyScaleValue;
    }
}


void GreyScale::display(){
    this->img.display();
}


void GreyScale::writeFile(){
    std::string outputFileName = "GreyScale_" + fileName;
    //std::filesystem::path outFilePath = this->filePath / outputFileName;
    std::string outFilePath = this->filePath + "/" + outputFileName;
    const char * filename_output = outFilePath.c_str();
    this->img.save(filename_output);
}