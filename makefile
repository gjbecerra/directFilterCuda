# Builds the executable
all: directFilter.cu
	nvcc --compile -I/home/gbecerra/NVIDIA_CUDA-10.2_Samples/common/inc -arch=sm_62 -O3 directFilter.cu -o directFilter.o
	nvcc --link -lmatio -arch=sm_62 -O3 directFilter.o -o directFilter 
clean:
	rm directFilter.o directFilter
