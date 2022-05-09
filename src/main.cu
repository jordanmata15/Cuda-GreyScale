#include "main.cuh"
// TODO
#include <iostream>


int main(int argc, char** argv){
    ArgParser ap = ArgParser();
    Arguments* args = ap.parseArgs(argc, argv);

    GreyScale gs = GreyScale();
    gs.loadFile();
    //gs.display();
    //gs.makeGreyScaleSerial();
    gs.makeGreyScaleParallel(args->numBlocks, args->numGrids);
    gs.display();
    gs.writeFile();
}