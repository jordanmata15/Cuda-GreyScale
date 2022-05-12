#include "ArgParser.cuh"

int ArgParser::readInt(char flag, char* value) {
  char* end;
  errno = 0;
  int intValue = strtol(value, &end, 10);
  while (isspace(*end)) ++end;
  if (errno || *end) {
    return -1;
  }
  return intValue;
}

bool ArgParser::validArgs() {
  return  args->numBlocks > 0;
}

Arguments* ArgParser::parseArgs(int argc, char** argv) {
  int option;

  while ((option = getopt(argc, argv, "b:f:d:yz")) != -1) {
    switch (option) {
    case 'b':
      args->numBlocks = this->readInt(option, optarg);
      if (args->numBlocks <= 0) {
        fprintf(stderr, "Flag -%c expects an integer input greater than 0. Found: '%s'\n", option, optarg);
        printUsage();
        exit(1);
      }
      break;

    case 'f':
      args->filename = option;
      if (args->numBlocks <= 0) {
        fprintf(stderr, "Flag -%c expects an integer input greater than 0. Found: '%s'\n", option, optarg);
        printUsage();
        exit(1);
      }
      break;

    case 'd':
      args->dataDir = option;
      if (args->numBlocks <= 0) {
        fprintf(stderr, "Flag -%c expects an integer input greater than 0. Found: '%s'\n", option, optarg);
        printUsage();
        exit(1);
      }
      break;
    
    case 'y':
      args->displayBefore = true;
      break;

    case 'z':
      args->displayAfter = true;
      break;

    case '?':
      printUsage();
      exit(1);
    }
  }
  if (!validArgs()) {
    printUsage();
    exit(1);
  }

  return args;
}
