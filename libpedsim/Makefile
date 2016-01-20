# Set os specific variables
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXX_ = g++
	LDFLAGS_ = -lpthread -fopenmp -shared -Wl,-soname,libpedsim.so
endif
ifeq ($(UNAME_S),Darwin)
	CXX_ = g++-5
	LDFLAGS_ = -lpthread -fopenmp -shared -Wl,-install_name,libpedsim.so
endif

NVCC = nvcc
SOURCES_CXX = ped_waypoint.cpp ped_agent.cpp ped_model.cpp ped_tree.cpp heatmap_seq.cpp
OBJECTS_CXX = $(SOURCES_CXX:.cpp=.cxx.o)

SOURCES_CU = tick_cuda.cu
OBJECTS_CU = $(SOURCES_CU:.cu=.cu.o)

CXXFLAGS_ = -lpthread -fopenmp -g -O3 -fno-inline -fPIC


# Workaround to make project compile on non-cuda capable machines.
ifeq ($(shell which $(NVCC)),)

$(warning Warning: Compiling without nvcc support. Compilation will proceed but CUDA code will not be included.)

CXX=$(CXX_)
CXXFLAGS = $(CXXFLAGS_)
LDFLAGS = $(LDFLAGS_)
OBJECTS =  $(OBJECTS_CXX)
else 
# Compile everything with nvcc. nvcc calls system default c++ compiler for the host code.
# Make shared library directly with default c+++ compiler, makes format compatability more likely, and life more easy.
CXX = $(NVCC)
CXXFLAGS = -arch sm_30 -rdc=true --compiler-options "$(CXXFLAGS_)"
LDFLAGS = --shared #--compiler-options "$(LDFLAGS_)"
OBJECTS =  prelinked_objects.o  $(OBJECTS_CU) $(OBJECTS_CXX)
endif



libpedsim.so: $(OBJECTS)
	$(CXX_) $(LDFLAGS_) $(OBJECTS) -o libpedsim.so 

%.cu.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@ 

%.cxx.o: %.cpp
	$(CXX) $(CXXFLAGS)  -c $< -o $@ 

# We need to do some linking beforehand, so that non-cuda compilers can link to the lib
# If the program were using nvcc, this step would be unneccesary
prelinked_objects.o:  $(OBJECTS_CU) $(OBJECTS_CXX)
	$(NVCC) $(CXXFLAGS) -dlink  $(OBJECTS_CU) $(OBJECTS_CXX) -o $@

# This removes a circular dependency from implicit make rules.
%.cu:

clean:
	rm -f *.o *.so *~ \#*
