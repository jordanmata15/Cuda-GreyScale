#include "main.cuh"
// TODO
#include <iostream>


int main(int argc, char** argv){
    GreyScale gs = GreyScale();
    gs.loadFile();
    //gs.display();
    //gs.makeGreyScaleSerial();
    gs.makeGreyScaleParallel();
    gs.display();
    gs.writeFile();
}