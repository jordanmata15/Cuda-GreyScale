#include "main.hpp"
// TODO
#include <iostream>


int main(int argc, char** argv){
    GreyScale gs = GreyScale();
    gs.loadFile();
    //gs.makeGreyScaleSerial();
    gs.displayResult();
    gs.writeFile();
}