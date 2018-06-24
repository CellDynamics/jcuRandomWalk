//============================================================================
// Name        : solverTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <string>
#include <fstream>
#include <vector>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cusolver.h>
#include "cusparse.h"
#include "cusolverSp.h"


//using namespace std;

int main() {
	std::vector<int> rowIndA;
    int* h_csrRowPtrA;
    std::vector<int> colIndA;
    int* h_csrColIndA;
    std::vector<float> valA;
    float* h_csrValA;
    std::vector<float> bA;
    float* h_b;
    float *h_x = NULL;

    int* d_csrRowPtrA = NULL;
    int* d_csrColIndA = NULL;
    float* d_csrValA = NULL;
    float* d_b = NULL;
    float* d_x = NULL;

    int colsA;
    int rowsA;
    int nnzA;

	{
		std::ifstream ris("/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/rowInd.txt");
		int value;
		while(ris >> value) {
			rowIndA.push_back(value);
		}
		ris.close();
		std::cout << "row size " << rowIndA.size() << std::endl;
		h_csrRowPtrA = rowIndA.data();
	}
	{
		std::ifstream ris("/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/colInd.txt");
		int value;
		while(ris >> value) {
			colIndA.push_back(value);
		}
		ris.close();
		std::cout << "col size " << colIndA.size() << std::endl;
		h_csrColIndA = colIndA.data();
		colsA = (*std::max_element(colIndA.begin(), colIndA.end()))+1;
	}
	{
		std::ifstream ris("/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/val.txt");

		float value;
		while(ris >> value) {
			valA.push_back(value);
		}
		ris.close();
		std::cout << "val size " << valA.size() << std::endl;
		h_csrValA = valA.data();
		nnzA = valA.size();
	}
	{
		std::ifstream ris("/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/b.txt");
		float value;
		while(ris >> value) {
			bA.push_back(value);
		}
		ris.close();
		std::cout << "b size " << bA.size() << std::endl;
		h_b = bA.data();
	}
	assert(rowIndA.size() == (colsA + 1)); // colsA is maximum entry from colIndA
	assert(nnzA == colIndA.size());
	rowsA = colsA; // square
	std::cout << "rowsA: " << rowsA << " colsA: " << colsA << " nnzA: " << nnzA << std::endl;
    cusolverSpHandle_t handle = NULL;
    cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL;

    float tol = 1.e-12;
    int reorder = 0; // no reordering
    int singularity = 0; // -1 if A is invertible under tol.

    assert(cusolverSpCreate(&handle) == CUSOLVER_STATUS_SUCCESS);
    assert(cusparseCreate(&cusparseHandle) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseCreateMatDescr(&descrA) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)== CUSPARSE_STATUS_SUCCESS);

    h_x = (float*)malloc(sizeof(float)*colsA); // result
    assert(NULL!=h_x);

    checkCudaErrors(cudaMalloc((void**)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
    checkCudaErrors(cudaMalloc((void**)&d_csrColIndA, sizeof(int)*(nnzA)));
    checkCudaErrors(cudaMalloc((void**)&d_csrValA, sizeof(float)*(nnzA)));
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(float)*(rowsA)));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(float)*(colsA)));

    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int)*(nnzA), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(float)*(nnzA), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float)*(rowsA), cudaMemcpyHostToDevice));

    std::cout << "Solver start" << std::endl;
    assert(cusolverSpScsrlsvqr(
                handle, rowsA, nnzA,
                descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
                d_b, tol, reorder, d_x, &singularity) == CUSOLVER_STATUS_SUCCESS);

    std::cout << "Solver done" << std::endl;
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
    for(int i=0; i<30; i++) {
    	std::cout << h_x[i] << ", ";
    }

    if(h_x) {free(h_x);}
    if(d_csrRowPtrA) {checkCudaErrors(cudaFree(d_csrRowPtrA));}
    if(d_csrColIndA) {checkCudaErrors(cudaFree(d_csrColIndA));}
    if(d_csrValA) {checkCudaErrors(cudaFree(d_csrValA));}
    if(d_b) {checkCudaErrors(cudaFree(d_b));}
    if(d_x) {checkCudaErrors(cudaFree(d_x));}

    cudaDeviceReset();
	return 0;
}
