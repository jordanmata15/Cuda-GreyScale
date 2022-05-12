#include "main.cuh"
#include <iostream>


int main(int argc, char** argv){
    ArgParser ap = ArgParser();
    Arguments* args = ap.parseArgs(argc, argv);

    GreyScale gs = GreyScale();
    gs.loadFile("../data", "tiger.bmp");
    gs.makeGreyScaleParallel(args->numBlocks);
    //gs.display();
    gs.writeFile();
    std::cout << std::endl << "Time Elapsed (seconds): " << gs.getTimeElapsed() << "\n" << std::endl;
}