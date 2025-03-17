CXX=g++
NVCC=nvcc
NVCC_LIB_PATH=/usr/lib/x86_64-linux-gnu
CXXFLAGS=-O3 -std=c++11 
NVCCFLAGS += -O3 -w -arch=sm_35 -rdc=true -Xptxas -dlcm=ca -Xcompiler -fopenmp --std=c++11 -m64 #-lineinfo #-g -G
NVCCLINKFLAGS = -L$(NVCC_LIB_PATH) -lcudart

all: ccdp_gpu

ccdp_gpu: main.cu util.o ccdp_gpu.o
	${NVCC} ${NVCCFLAGS} -o ccdp_gpu main.cu ccdp_gpu.o util.o $(NVCCLINKFLAGS)

ccdp_gpu.o: ccdp_gpu.cu util.o
	${NVCC} ${NVCCFLAGS} -c -o ccdp_gpu.o $<

util.o: util.h util.cpp
	${CXX} ${CXXFLAGS} -c -o util.o util.cpp

clean:
	rm -rf ccdp_gpu *.o 

