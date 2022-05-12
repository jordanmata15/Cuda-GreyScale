#include "main.cuh"


int main(int argc, char** argv){
  ArgParser ap = ArgParser();
  Arguments* args = ap.parseArgs(argc, argv);

  GreyScale gs = GreyScale();
  gs.loadFile(args->dataDir, args->filename);
  
  if (args->displayBefore) gs.display();
  gs.makeGreyScaleParallel(args->numBlocks);
  if (args->displayAfter) gs.display();
  
  gs.writeFile();

  std::cout << std::endl << "Time Elapsed (seconds): " << gs.getTimeElapsed() << "\n" << std::endl;
}