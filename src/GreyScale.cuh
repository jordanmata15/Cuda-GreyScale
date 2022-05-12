#ifndef GREYSCALE_CUH
#define GREYSCALE_CUH

#include <string>
#include "CImg.h"
#include "DataManager.cuh"

// we can't use the filesystem library with c++11. And CUDA doesn't work with c++17
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
static const std::string fileSeparator = "\\";
#else
static const std::string fileSeparator = "/";
#endif

class GreyScale {
private:
  cimg_library::CImg<u_char> img;
  std::string filePath;
  std::string fileName;
  DataManager dManager;

public:
  /**
   * @brief Construct a new Grey Scale object
   * 
   */
  GreyScale();

  /**
   * @brief Loads in the default bmp file (dino.bmp) from the default directory (../data)
   * 
   */
  void loadFile();
  
  /**
   * @brief Loads a user defined image from a user defined folder.
   * 
   * @param filePath the directory where to find the image
   * @param fileName a bmp formatted image
   */
  void loadFile(std::string filePath, std::string fileName);
  
  /**
   * @brief Makes the loaded image greyscale using no parallelism.
   *        Used as a proof of concept to help develop the parallel version.
   * 
   */
  void makeGreyScaleSerial();
  
  /**
   * @brief Makes the loaded image greyscale using CUDA. This is the host
   *        function that calls myKernel to run on the device.
   * 
   * @param numBlocks The dimension of our block. The dimension of our grid is inversely
   *                  proportional to this value. Larger block dimensions means smaller
   *                  grid dimensions (and vice versa).
   */
  void makeGreyScaleParallel(int numBlocks);
  
  /**
   * @brief Displays the image currently loaded in memory to screen.
   * 
   */
  void display();
  
  /**
   * @brief Writes the image currently loaded in memory to screen.
   * 
   */
  void writeFile();
  
  /**
   * @brief Gets the total time elapsed of the last algorithm run.
   * 
   * @return double The time elapsed in seconds.
   */
  double getTimeElapsed();
};

/**
 * @brief Kernel function that each core executes. Each core gets one pixel index
 *        and uses the RGB values to make it greyscale.
 *
 * @param imageArr 1D u_char array holding our pixels arranged in {RRR...GGG...BBB} order.
 * @param devImageDims Dimensions of our image array (devImageDims[0] = width, devImageDims[1] height)
 */
__global__ void mykernel(u_char* imageArr, int* devImageDims);

#endif // GREYSCALE_CUH