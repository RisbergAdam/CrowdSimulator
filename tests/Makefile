all:


# You may need to add -I path to opencl headers to make this work.
opencl:
	$(CC_) helloWorld.c $(OPENCLFLAGS) -o helloWorldOpencl
	./helloWorldOpencl

cuda:
	nvcc helloWorld.cu -o helloWorldCuda
	./helloWorldCuda
omp:
	$(CXX_) helloOmp.cpp -fopenmp -o helloWorldOmp
	./helloWorldOmp

test.clean:
	rm -f helloWorldCuda
	rm -f helloWorldOmp
	rm -f hellowWorldOpencl
