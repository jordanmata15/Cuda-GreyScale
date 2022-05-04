SRCDIR=.
BUILD=../build
SCRIPTING=../scripting
EXTERNAL=../external

PROGRAM=GreyScale
CXX=g++
CXXFLAGS=-std=c++17 -g -Wall
INCLUDE=-I${EXTERNAL}/build/include/opencv4
LDLIBS=-L${EXTERNAL}/build/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_core

SRC:=$(shell find $(SRCDIR)/ -name "*.cpp") # all *.cpp files in src directory
OBJS:=$(SRC:%.cpp=$(BUILD)/%.o)				# all *.o files in the build directory

all: ${OBJS} $(BUILD)/${PROGRAM}

$(BUILD)/$(PROGRAM):	$(OBJS)
	$(CXX) $(CXXFLAGS) $(BUILD)/$(OBJS) ${INCLUDE} ${LDLIBS} -o $(BUILD)/$(PROGRAM)

$(BUILD)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/%.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${INCLUDE}

run: all
	$(BUILD)/$(PROGRAM)

clean:
	rm -rf $(BUILD)/core* ${BUILD}/$(OBJS) $(BUILD)/$(PROGRAM)
