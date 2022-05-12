#ifndef ARG_PARSER_CUH
#define ARG_PARSER_CUH

#include <iostream>
#include <getopt.h>

#define DEFAULT_BLOCKS 10
#define DEFAULT_DATA_DIR "../data"
#define DEFAULT_BMP_FILE "dino.bmp"
#define INVALID -1
#define USAGE "\nusage: ./GreyScale <optional_flags>\n"\
              "\tOptional flags with arguments:\n"\
              "\t\t-b reads in NUM_BLOCKS\t\t(integer greater than 0)\n"\
              "\t\t-f reads in the name of the bmp filename (default is dino.bmp)\n"\
              "\t\t-d reads in the directory of the bmp file to read (default is ../data)\n"

/**
 * @brief Arguments object to hold the values of the desired inputs given by the user.
 */
class Arguments {
public:
  int numBlocks;
  std::string filename;
  std::string dataDir;

  /**
   * Basic constructor. sets default values.
   */
  Arguments() : numBlocks(DEFAULT_BLOCKS), filename(DEFAULT_BMP_FILE), dataDir(DEFAULT_DATA_DIR) {}
};


/**
 * @brief Class to read in command line arguments and validate them. Fills an Arguments
 *        object with the users desired input.
 */
class ArgParser {

private:
  /**
   * @brief Reads in an string and returns the converted integer.
   *
   * @param value String value to convert to an integer.
   * @return int The converted integer. Returns -1 if the string was not successfully converted.
   */
  int readInt(char flag, char* value);

  /**
   * @brief Prints out the usage of our program.
   *
   */
  void printUsage() { std::cout << USAGE << std::endl; }

  /**
   * @brief Validates whether our inputs are valid and we can populate the matrices.
   *
   * @return true if the populated arguments are valid.
   * @return false otherwise.
   */
  bool validArgs();

public:
  Arguments* args;

  /**
   * @brief Construct a new Arg Parser object. Dynamically allocate the memory.
   *
   */
  ArgParser() { args = new Arguments(); }

  /**
   * @brief Destroy the Arg Parser object and free the allocated memory.
   *
   */
  ~ArgParser() { delete args; }

  /**
   * @brief Method used to parse the flags from the user. Validates integer arguments
   * and exits if invalid.
   *
   * @param argc The argument count provided to main().
   * @param argv The argument list provided to main().
   * @return Arguments* A pointer to an argument structure from which the user inputs
   *         can be read.
   */
  Arguments* parseArgs(int argc, char** argv);
};

#endif // ARG_PARSER_CUH