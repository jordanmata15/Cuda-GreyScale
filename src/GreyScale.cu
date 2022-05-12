#include "GreyScale.cuh"

using namespace cimg_library;

GreyScale::GreyScale() {
  this->filePath = ".." + fileSeparator + "data";
  this->fileName = "dino.bmp";
  this->dManager = DataManager();
}

void GreyScale::loadFile() {
  this->loadFile(this->filePath, this->fileName);
}

void GreyScale::loadFile(std::string filePath, std::string fileName) {
  this->filePath = filePath;
  this->fileName = fileName;
  std::string fullFilePath = this->filePath + fileSeparator + this->fileName;
  const char* filename_input = fullFilePath.c_str();
  this->img = CImg<u_char>(filename_input);
}


void GreyScale::makeGreyScaleSerial() {
  int width   = this->img.width(),
      height  = this->img.height(),
      depth   = this->img.depth(),
      length = width * height * depth;

  /* pointer to image pixels. Colors are arranged in contiguous memory.
  Eg: {RRR...RRRGGG...GGGBBB...BBB} */
  u_char  * arr = this->img.data(),
          * R = &arr[0 * length],
          * G = &arr[1 * length],
          * B = &arr[2 * length],
          greyScaleValue;

  dManager.startTimer();
  for (size_t i = 0; i < length; ++i) {
    greyScaleValue = R[i] * 0.3 + G[i] * 0.59 + B[i] * 0.11;
    R[i] = greyScaleValue;
    G[i] = greyScaleValue;
    B[i] = greyScaleValue;
  }
  dManager.stopTimer();
}


void GreyScale::makeGreyScaleParallel(int numBlocks) {
  /* pointer to image pixels. Colors are rranged in contiguous memory.
  Eg: {RRR...RRRGGG...GGGBBB...BBB} */
  u_char  * hostImageArr = this->img.data(),
          * devImageArr;

  int * devImageDims,
      channels = 3,
      length = this->img.width() * this->img.height() * this->img.depth() * channels,
      imageDims[2] = { this->img.width(), this->img.height() };


  dManager.startTimer(); /* start of CUDA procedures */

  cudaMalloc((void**)&devImageArr, length * sizeof(u_char));
  cudaMalloc((void**)&devImageDims, 2 * sizeof(unsigned long));
  cudaMemcpy(devImageArr, hostImageArr, length * sizeof(u_char), cudaMemcpyHostToDevice);
  cudaMemcpy(devImageDims, &imageDims, 2 * sizeof(unsigned long), cudaMemcpyHostToDevice);

  // grid and block numbers are inversely proportional. More blocks => smaller grid (and vice versa)
  dim3  dimBlock(numBlocks, numBlocks),
        dimGrid( (img.width()+numBlocks-1)/dimBlock.x, (img.height()+numBlocks-1)/dimBlock.y);
  mykernel << <dimGrid, dimBlock >> > (devImageArr, devImageDims);

  cudaMemcpy(hostImageArr, devImageArr, length * sizeof(u_char), cudaMemcpyDeviceToHost);
  cudaFree(devImageArr);
  cudaFree(devImageDims);

  dManager.stopTimer(); /* end of CUDA procedures */
}

// each core on the gpu runs this. Each handles making one pixel greyscale
__global__ void mykernel(u_char* imageArr, int* devImageDims) {
  int width     = devImageDims[0],
      height    = devImageDims[1],
      length    = width * height,
      x         = threadIdx.x + blockIdx.x * blockDim.x,
      y         = threadIdx.y + blockIdx.y * blockDim.y,
      pixelIdx  = x + y * blockDim.x * gridDim.x;

  // a single channel takes up "length" number of ints
  if (pixelIdx > length*3) return;
  // pointers to the start of each of the color channels
  u_char  * R = &imageArr[0 * length],
          * G = &imageArr[1 * length],
          * B = &imageArr[2 * length],
          greyScaleValue = 0;

  greyScaleValue = R[pixelIdx]*0.3 + G[pixelIdx]*0.59 + B[pixelIdx]*0.11;
  R[pixelIdx] = greyScaleValue;
  G[pixelIdx] = greyScaleValue;
  B[pixelIdx] = greyScaleValue;
}


void GreyScale::display() {
  this->img.display();
}


void GreyScale::writeFile() {
  std::string outputFileName = "GreyScale_" + fileName;
  std::string outFilePath = this->filePath + fileSeparator + outputFileName;
  const char* filename_output = outFilePath.c_str();
  this->img.save(filename_output);
}

double GreyScale::getTimeElapsed(){
  return dManager.getTimeElapsed();
}