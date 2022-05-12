PACKAGE_DIR=$(shell dirname $(shell pwd))
SRCDIR=$(PACKAGE_DIR)/src
BUILD=$(PACKAGE_DIR)/build
SCRIPTING=$(PACKAGE_DIR)/scripting
EXTERNAL=$(PACKAGE_DIR)/external

PROGRAM=GreyScale
EXE=$(BUILD)/$(PROGRAM)

CXX=nvcc
CXXFLAGS=-std=c++11 -g# -Wall
INCLUDE=-I $(EXTERNAL)/CImg
LDLIBS=-L $(EXTERNAL)/CImg -Dcimg_display=1 -lm -lX11

SRC:=$(shell find $(SRCDIR)/ -name "*.cu") # all *.cpp files in src directory
OBJS:=$(SRC:$(SRCDIR)/%.cu=$(BUILD)/%.o)	# all *.o files in the build directory

# Optional flags the user can pass in to the make command
RUN_ARGS=

all: $(OBJS) $(BUILD)/${PROGRAM}

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) ${INCLUDE} ${LDLIBS} -o $(EXE)

$(BUILD)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/%.cuh
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${INCLUDE}

benchmarks: all
	$(BUILD)/$(PROGRAM) -b 1 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 2 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 4 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 8 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 12 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 16 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 20 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 24 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 28 $(RUN_ARGS)
	$(BUILD)/$(PROGRAM) -b 32 $(RUN_ARGS)

run: all
	$(BUILD)/$(PROGRAM) $(RUN_ARGS)	

clean:
	rm -rf $(BUILD)/core* $(OBJS) $(BUILD)/$(PROGRAM)
