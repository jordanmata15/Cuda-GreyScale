PACKAGE_DIR=$(shell dirname $(shell pwd))
SRCDIR=$(PACKAGE_DIR)/src
BUILD=$(PACKAGE_DIR)/build
SCRIPTING=$(PACKAGE_DIR)/scripting
EXTERNAL=$(PACKAGE_DIR)/external

PROGRAM=GreyScale
EXE=$(BUILD)/$(PROGRAM)
CXX=nvcc
CXXFLAGS=-std=c++11 -g# -Wall
INCLUDE=-I $(EXTERNAL)/CImg -I $(EXTERNAL)/CImg/plugins -I $(EXTERNAL)/jpeg-6b
LDLIBS=-L $(EXTERNAL)/CImg -L $(EXTERNAL)/CImg/plugins -Dcimg_display=1 -lm -lX11

SRC:=$(shell find $(SRCDIR)/ -name "*.cu") # all *.cpp files in src directory
OBJS:=$(SRC:$(SRCDIR)/%.cu=$(BUILD)/%.o)	# all *.o files in the build directory

all: $(OBJS) $(BUILD)/${PROGRAM}

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) ${INCLUDE} ${LDLIBS} -o $(EXE)

$(BUILD)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/%.cuh
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${INCLUDE}

run: all
	$(BUILD)/$(PROGRAM)

clean:
	rm -rf $(BUILD)/core* $(OBJS) $(BUILD)/$(PROGRAM)
