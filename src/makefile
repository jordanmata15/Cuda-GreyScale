SRCDIR=.
BUILD=../build
SCRIPTING=../scripting
EXTERNAL=../external

PROGRAM=GreyScale
CXX=g++
CXXFLAGS=-std=c++17 -g -Wall
LIB=-I/usr/local/include/opencv4 -L${EXTERNAL}/build/lib -lstdc++ -lopencv_imgcodecs -lopencv_highgui \
-lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video

#LIB=-I/opt/opencv/include/opencv4 -L/opt/opencv/lib -lstdc++ -lopencv_imgcodecs -lopencv_highgui \
-lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video

SRC:=$(shell find $(SRCDIR)/ -name "*.cpp")
OBJS:=$(SRC:%.cpp=$(BUILD)/%.o)

all: ${OBJS} $(BUILD)/${PROGRAM}

$(BUILD)/$(PROGRAM):	$(OBJS)
	$(CXX) $(CXXFLAGS) $(BUILD)/$(OBJS) ${LIB} -o $(BUILD)/$(PROGRAM)

$(BUILD)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/%.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${LIB}

plot:
	python3 $(SCRIPTING)/plot.py

run:
	$(BUILD)/$(PROGRAM)

clean:
	rm -rf $(BUILD)/core* ${BUILD}/$(OBJS) $(BUILD)/$(PROGRAM)