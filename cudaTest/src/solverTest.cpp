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
#include <iomanip>
#include <stdexcept>
#include <algorithm>

#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cusolver.h>
#include "cusparse.h"
#include <cublas.h>
#include "cusolverSp.h"


//using namespace std;

int main() {
	const double TOL = 1e-3;
	const size_t MAX_ITER = 1000;

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
	// get lower
	const size_t LOWER = 0.5*(nnzA + rowsA);
	float *h_val = (float *)malloc(LOWER*sizeof(float));
	int *h_col = (int *)malloc(LOWER*sizeof(int));
	int *h_row = (int *)malloc((rowsA + 1)*sizeof(int));
	//populate lower triangular column indices and row offsets for zero fill-in IC
	h_row[rowsA] = LOWER;
    int k = 0;
	for (int i = 0; i < rowsA; i++) {
		h_row[i] = k;
		int numRowElements = h_csrRowPtrA[i+1] - h_csrRowPtrA[i];
		int m = 0;
		for (int j = 0; j < numRowElements; j++) {
			if (!(h_csrColIndA[h_csrRowPtrA[i] + j] > i)) {
				h_col[h_row[i] + m] = h_csrColIndA[h_csrRowPtrA[i] + j];
				h_val[h_row[i] + m] = h_csrValA[h_csrRowPtrA[i] + j];
				k++; m++;
			}
		}
	}

    cusparseHandle_t cusparseHandle = NULL; // used in residual evaluation
    cublasHandle_t cublasHandle = NULL;
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL; // desc of A
    cusparseSolveAnalysisInfo_t infoA = NULL; // analysis result from csrsv_analysis

    h_x = (float*)calloc(colsA, sizeof(float)); // result
    assert(NULL!=h_x);

    assert(cusparseCreate(&cusparseHandle) == CUSPARSE_STATUS_SUCCESS);
    assert(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    // description of A (triangle)
    assert(cusparseCreateMatDescr(&descrA) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO) == CUSPARSE_STATUS_SUCCESS);

    // matrix A Lower
    checkCudaErrors(cudaMalloc((void**)&d_csrRowPtrA, sizeof(int)*(rowsA+1)));
    checkCudaErrors(cudaMalloc((void**)&d_csrColIndA, sizeof(int)*LOWER));
    checkCudaErrors(cudaMalloc((void**)&d_csrValA, sizeof(float)*LOWER));
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_row, sizeof(int)*(rowsA+1), cudaMemcpyHostToDevice)); // copy triangle lower
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_col, sizeof(int)*(LOWER), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA, h_val, sizeof(float)*(LOWER), cudaMemcpyHostToDevice));
    // copy valA on GPU - they will bo modified (not necessary)
    float* d_valChol = NULL;
    checkCudaErrors(cudaMalloc((void**)&d_valChol, sizeof(float)*LOWER));
    checkCudaErrors(cudaMemcpy(d_valChol, d_csrValA, sizeof(float)*LOWER, cudaMemcpyDeviceToDevice));
    // result x
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(float)*(colsA)));
    checkCudaErrors(cudaMemcpy(d_x, h_x, colsA*sizeof(float), cudaMemcpyHostToDevice));
    // RHS
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(float)*(rowsA)));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float)*(rowsA), cudaMemcpyHostToDevice));

    // analysis of A
    std::cout << "cusparseScsrsv_analysis" << std::endl;
    assert(cusparseCreateSolveAnalysisInfo(&infoA)==CUSPARSE_STATUS_SUCCESS);
    assert(cusparseScsrsv_analysis(cusparseHandle,
    		CUSPARSE_OPERATION_NON_TRANSPOSE,
    		rowsA,
    		LOWER,
    		descrA,
    		d_csrValA,
    		d_csrRowPtrA,
    		d_csrColIndA,
    		infoA)==CUSPARSE_STATUS_SUCCESS);

    std::cout << "cusparseScsric0" << std::endl;
    cusparseStatus_t cusparseStatus = cusparseScsric0(cusparseHandle,
    		CUSPARSE_OPERATION_NON_TRANSPOSE,
    		rowsA,
    		descrA,
    		d_valChol, // will modify this!
    		d_csrRowPtrA,
    		d_csrColIndA,
    		infoA);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cusparseScsric0 returned error code " << cusparseStatus << std::endl;
    }
    cusparseDestroySolveAnalysisInfo(infoA);

    // create info and analyse lower and upper triangular factors (nvidia paper)
    cusparseMatDescr_t descrL = 0;
    assert(cusparseCreateMatDescr(&descrL)==CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT) == CUSPARSE_STATUS_SUCCESS);
    assert(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO) == CUSPARSE_STATUS_SUCCESS);
    cusparseSolveAnalysisInfo_t infoL = 0;
    assert(cusparseCreateSolveAnalysisInfo(&infoL)==CUSPARSE_STATUS_SUCCESS);
    std::cout << "cusparseScsrsv_analysis L" << std::endl;
    assert(cusparseScsrsv_analysis(cusparseHandle,
    		CUSPARSE_OPERATION_NON_TRANSPOSE,
    		rowsA,
    		LOWER,
    		descrL,
    		d_valChol,
    		d_csrRowPtrA,
    		d_csrColIndA,
    		infoL)==CUSPARSE_STATUS_SUCCESS); // infoR

    cusparseSolveAnalysisInfo_t infoU = 0;
    assert(cusparseCreateSolveAnalysisInfo(&infoU)==CUSPARSE_STATUS_SUCCESS);
    std::cout << "cusparseScsrsv_analysis U" << std::endl;
    assert(cusparseScsrsv_analysis(cusparseHandle,
    		CUSPARSE_OPERATION_TRANSPOSE,
    		rowsA,
    		LOWER,
    		descrL,
    		d_valChol,
    		d_csrRowPtrA,
    		d_csrColIndA,
    		infoU)==CUSPARSE_STATUS_SUCCESS); // infoRt

    // 1. nvidia paper
    float *d_r; // residual
    float normr0, normr;
    float zero = 0.0;
    float one = 1.0, negone = -1.0;
    float rho;
    float rhop;
    float ptAp;
    float alpha, negalpha;
    float beta;
    float *d_t;
    float *d_p;
    float *d_q;
    float *d_z;
    checkCudaErrors(cudaMalloc((void **)&d_t, rowsA*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, rowsA*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_q, rowsA*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_z, rowsA*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, rowsA*sizeof(float)));
    std::cout << "cusparseScsrmv" << std::endl;
    assert(cusparseScsrmv(cusparseHandle,
    		CUSPARSE_OPERATION_NON_TRANSPOSE,
    		rowsA,
    		colsA,
    		LOWER,
    		&one,
    		descrA,
    		d_csrValA,
    		d_csrRowPtrA,
    		d_csrColIndA,
    		d_x,
    		&zero,
    		d_r)==CUSPARSE_STATUS_SUCCESS);
    std::cout << "cublasHandle" << std::endl;
    assert(cublasSscal(cublasHandle, rowsA, &negone, d_r, 1)==CUBLAS_STATUS_SUCCESS);
    std::cout << "cublasSaxpy" << std::endl;
    assert(cublasSaxpy(cublasHandle, rowsA, &one, d_b, 1, d_r, 1)==CUBLAS_STATUS_SUCCESS);
    std::cout << "cublasDdot" << std::endl;
    assert(cublasSdot(cublasHandle, rowsA, d_r, 1, d_z, 1, &rho)==CUBLAS_STATUS_SUCCESS);
    std::cout << "cublasSnrm2" << std::endl;
    assert(cublasSnrm2(cublasHandle, rowsA, d_r, 1, &normr0)==CUBLAS_STATUS_SUCCESS);
    normr = normr0;

    int i = 0;
    while (normr/normr0 > TOL && i < MAX_ITER) {
    	cusparseStatus = cusparseScsrsv_solve(cusparseHandle,
    			CUSPARSE_OPERATION_NON_TRANSPOSE,
    			rowsA,
    			&one,
    			descrL,
    			d_valChol,
    			d_csrRowPtrA,
    			d_csrColIndA,
    			infoL,
    			d_r,
    			d_t);
    	if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    	    	std::cerr << "cusparseScsrsv_solve returned error code " << cusparseStatus << std::endl;
    	        return 1;
    	}
    	cusparseStatus = cusparseScsrsv_solve(cusparseHandle,
    			CUSPARSE_OPERATION_TRANSPOSE,
    			rowsA,
    			&one,
    			descrL,
    			d_valChol,
    			d_csrRowPtrA,
    			d_csrColIndA,
    			infoU,
    			d_t,
    			d_z);
    	if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    	        std::cerr << "cusparseScsrsv_solve returned error code " << cusparseStatus << std::endl;
    	        return 1;
    	}
    	rhop = rho;
    	cublasSdot(cublasHandle, rowsA, d_r, 1, d_z, 1, &rho);
    	if (i == 0) {
    		cublasScopy(cublasHandle, rowsA, d_z, 1, d_p, 1);
    	} else {
    		beta = rho/rhop;
			cublasSaxpy(cublasHandle, rowsA, &beta, d_p, 1, d_z, 1);
			cublasScopy(cublasHandle, rowsA, d_z, 1, d_p, 1);
    	}
    	assert(cusparseScsrmv(cusparseHandle,
    			CUSPARSE_OPERATION_NON_TRANSPOSE,
    			rowsA,
    			rowsA,
    			LOWER,
    			&one,
    			descrA,
    			d_csrValA,
    			d_csrRowPtrA,
    			d_csrColIndA,
    			d_p,
    			&zero,
    			d_q)==CUSPARSE_STATUS_SUCCESS);
    	assert(cublasSdot(cublasHandle, rowsA, d_p, 1, d_q, 1, &ptAp)==CUBLAS_STATUS_SUCCESS);
    	alpha = rho / ptAp;
    	assert(cublasSaxpy(cublasHandle, rowsA, &alpha, d_p, 1, d_x, 1)==CUBLAS_STATUS_SUCCESS);
    	negalpha = -alpha;
    	assert(cublasSaxpy(cublasHandle, rowsA, &negalpha, d_q, 1, d_r, 1)==CUBLAS_STATUS_SUCCESS);
    	assert(cublasSnrm2(cublasHandle, rowsA, d_r, 1, &normr)==CUBLAS_STATUS_SUCCESS);
    	std::cout << "iter " << i << " err " << normr/normr0 << std::endl;
    	i++;

    }
    std::cout << "Saving" << std::endl;
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
	std::ofstream ofs("/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/x_out.txt");
	for(int i=0;i<rowsA;i++) {
		ofs << std::setprecision(7) << h_x[i] << std::endl;
	}
	ofs.close();

    cudaDeviceReset();
    if(h_x) {free(h_x);}
    free(h_val);
    free(h_col);
    free(h_row);
    std::cout << "bye" << std::endl;
	return 0;
}

/*
 * float tol = 1.e-12;
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
 */
