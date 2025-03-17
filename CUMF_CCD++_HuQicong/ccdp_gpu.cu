/**
 *
 * OHIO STATE UNIVERSITY SOFTWARE DISTRIBUTION LICENSE
 *
 * Parallel CCD++ on GPU (the “Software”) Copyright (c) 2017, The Ohio State
 * University. All rights reserved.
 *
 * The Software is available for download and use subject to the terms and
 * conditions of this License. Access or use of the Software constitutes acceptance
 * and agreement to the terms and conditions of this License. Redistribution and
 * use of the Software in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the capitalized paragraph below.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the capitalized paragraph below in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. The names of Ohio State University, or its faculty, staff or students may not
 * be used to endorse or promote products derived from the Software without
 * specific prior written permission.
 *
 * This software was produced with support from the National Science Foundation
 * (NSF) through Award 1629548. Nothing in this work should be construed as
 * reflecting the official policy or position of the Defense Department, the United
 * States government, Ohio State University.
 *
 * THIS SOFTWARE HAS BEEN APPROVED FOR PUBLIC RELEASE, UNLIMITED DISTRIBUTION. THE
 * SOFTWARE IS PROVIDED “AS IS” AND WITHOUT ANY EXPRESS, IMPLIED OR STATUTORY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF ACCURACY, COMPLETENESS,
 * NONINFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  ACCESS OR USE OF THE SOFTWARE IS ENTIRELY AT THE USER’S RISK.  IN
 * NO EVENT SHALL OHIO STATE UNIVERSITY OR ITS FACULTY, STAFF OR STUDENTS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  THE SOFTWARE
 * USER SHALL INDEMNIFY, DEFEND AND HOLD HARMLESS OHIO STATE UNIVERSITY AND ITS
 * FACULTY, STAFF AND STUDENTS FROM ANY AND ALL CLAIMS, ACTIONS, DAMAGES, LOSSES,
 * LIABILITIES, COSTS AND EXPENSES, INCLUDING ATTORNEYS’ FEES AND COURT COSTS,
 * DIRECTLY OR INDIRECTLY ARISING OUT OF OR IN CONNECTION WITH ACCESS OR USE OF THE
 * SOFTWARE.
 *
 */

/**
 *
 * Author:
 * 			Israt (nisa.1@osu.edu)
 *
 * Contacts:
 * 			Israt (nisa.1@osu.edu)
 * 			Aravind Sukumaran-Rajam (sukumaranrajam.1@osu.edu)
 * 			P. (Saday) Sadayappan (sadayappan.1@osu.edu)
 *
 */

#include "device_utilities.h"
#include "util.h"
#include <cuda.h>
#include <vector>

#include "helper_fusedR.h"
#include "helper_updateH.h"

inline cudaError_t checkCuda(cudaError_t result, int s) {

	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n",
				cudaGetErrorString(result), s);
		assert(result == cudaSuccess);
	}
	return result;
}

// Cyclic Coordinate Descent for Matrix Factorization
void ccdr1(SparseMatrix &R, MatData &W, MatData &H, TestData &T,
		Options &param) {

	int k = param.k, maxiter = param.maxiter, inneriter = param.maxinneriter,
			tileSize_H = param.tileSizeH, tileSize_W = param.tileSizeW;
	DTYPE lambda = param.lambda, *d_R_val, *d_R_val_t, *d_gArrU, *d_hArrU,
			*d_gArrV, *d_hArrV, *d_u, *d_v, oldobj = 0;
	int LB[NUM_THRDS], UB[NUM_THRDS], LB_Rt[NUM_THRDS], UB_Rt[NUM_THRDS],
			*d_R_colPtr, *d_R_rowPtr, *d_row_lim_R, *d_row_lim_Rt, sum = 0,
			*d_test_row, *d_test_col, i, j;
	DTYPE reg = 0, loss, *d_loss, *d_v_new, *d_Wt, *d_Ht, *d_W, *d_H,
			*d_test_val, v, *d_pred_v, *d_rmse, *d_fundec_col;
	unsigned *d_R_rowIdx, *d_R_colIdx;

	DTYPE *pred_v = (DTYPE*) malloc(T.nnz_ * sizeof(DTYPE));
	DTYPE *rmse = (DTYPE*) malloc(T.nnz_ * sizeof(DTYPE));

	//omp_set_num_threads(param.threads);
	// Create transpose view of R
	SparseMatrix Rt;
	Rt = R.get_shallow_transpose();
	// initial value of the regularization term
	// H is a zero matrix now.
	for (int t = 0; t < k; ++t)
		for (unsigned c = 0; c < R.cols_; ++c)
			H[t][c] = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float mili = 0, copyTime = 0;

	//**************************CUDA COPY************************

	checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), __LINE__);
	size_t RCols_memsize = (R.cols_) * sizeof(DTYPE);
	size_t RRows_memsize = (R.rows_) * sizeof(DTYPE);
	size_t d_memsize = 1 * sizeof(DTYPE);
	size_t R_colPtr_memsize = (R.cols_ + 1) * sizeof(int);
	size_t R_rowPtr_memsize = (R.rows_ + 1) * sizeof(int);
	size_t R_rowIdx_memsize = R.nnz_ * sizeof(unsigned);
	size_t R_val_memsize = R.nnz_ * sizeof(DTYPE);

	checkCuda(cudaMalloc((void**) &d_W, k * R.rows_ * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &d_H, k * R.cols_ * sizeof(DTYPE)), 1);
	checkCuda(cudaMalloc((void**) &d_Wt, R.rows_ * sizeof(DTYPE)), 0);
	checkCuda(cudaMalloc((void**) &d_Ht, R.cols_ * sizeof(DTYPE)), 1);
	checkCuda(cudaMalloc((void**) &d_u, RRows_memsize), 0);
	checkCuda(cudaMalloc((void**) &d_v, RCols_memsize), 1);
	checkCuda(cudaMalloc((void**) &d_v_new, RCols_memsize), 1);
	checkCuda(cudaMalloc((void**) &d_gArrU, RRows_memsize), 2);
	checkCuda(cudaMalloc((void**) &d_hArrU, RRows_memsize), 3);
	checkCuda(cudaMalloc((void**) &d_gArrV, RCols_memsize), 2);
	checkCuda(cudaMalloc((void**) &d_hArrV, RCols_memsize), 3);
	checkCuda(cudaMalloc((void**) &d_R_colPtr, R_colPtr_memsize), 4);
	checkCuda(cudaMalloc((void**) &d_R_rowPtr, R_rowPtr_memsize), 5);
	checkCuda(cudaMalloc((void**) &d_R_rowIdx, R_rowIdx_memsize), 6);
	checkCuda(cudaMalloc((void**) &d_R_colIdx, R_rowIdx_memsize), 8);
	checkCuda(cudaMalloc((void**) &d_R_val, R_val_memsize), 7);
	checkCuda(cudaMalloc((void**) &d_R_val_t, R_val_memsize), 7);
	checkCuda(cudaMalloc((void**) &d_loss, 1 * sizeof(DTYPE)), 11);
	checkCuda(cudaMalloc((void**) &d_test_row, (T.nnz_ + 1) * sizeof(int)), 7);
	checkCuda(cudaMalloc((void**) &d_test_col, (T.nnz_ + 1) * sizeof(int)), 7);
	checkCuda(cudaMalloc((void**) &d_test_val, (T.nnz_ + 1) * sizeof(DTYPE)),
			7);
	checkCuda(cudaMalloc((void**) &d_pred_v, (T.nnz_ + 1) * sizeof(DTYPE)), 7);
	checkCuda(cudaMalloc((void**) &d_rmse, (T.nnz_ + 1) * sizeof(DTYPE)), 7);

	checkCuda(cudaEventRecord(start), __LINE__);
	cudaMemcpy(d_R_colPtr, R.get_csc_col_ptr(), R_colPtr_memsize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_rowPtr, R.get_csr_row_ptr(), R_rowPtr_memsize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_rowIdx, R.get_csc_row_indx(), R_rowIdx_memsize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_colIdx, R.get_csr_col_indx(), R_rowIdx_memsize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_val, R.get_csc_val(), R_val_memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_val_t, R.get_csr_val(), R_val_memsize,
			cudaMemcpyHostToDevice);

	for (int t = 0; t < k; ++t)
		cudaMemcpy(d_W + t * R.rows_, &(W[t][0]), R.rows_ * sizeof(DTYPE),
				cudaMemcpyHostToDevice);
	cudaMemset(d_H, 0, k * R.cols_ * sizeof(DTYPE));
	//cpoying test
	cudaMemcpy(d_test_row, T.getTestRow(), (T.nnz_ + 1) * sizeof(int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_test_col, T.getTestCol(), (T.nnz_ + 1) * sizeof(int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_test_val, T.getTestVal(), (T.nnz_ + 1) * sizeof(DTYPE),
			cudaMemcpyHostToDevice);

	checkCuda(cudaEventRecord(stop), __LINE__);
	checkCuda(cudaDeviceSynchronize(), __LINE__);
	checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
	copyTime = mili;

	float ACSRTime = 0, textureACSRTime = 0, innerLoopTime = 0;
	float ACSRPreProcessTime;

	cudaStream_t streamT;
	checkCuda(cudaStreamCreate(&streamT), __LINE__);
	create_stream();

	//****************** preprrocessing TILING*************

	double t1 = seconds();
	int total_tileInRows = (R.rows_ + tileSize_H - 1) / tileSize_H;
	int total_tileInCols = (R.cols_ + tileSize_W - 1) / tileSize_W;

	MatInt row_lim_R = MatInt(total_tileInRows + 1, VecInt(R.cols_ + 1));
	MatInt row_lim_Rt = MatInt(total_tileInCols + 1, VecInt(R.rows_ + 1));
	MatInt row_lim_R_odd = MatInt(total_tileInRows + 1, VecInt(R.cols_ + 1));
	MatInt row_lim_Rt_odd = MatInt(total_tileInCols + 1, VecInt(R.rows_ + 1));

	make_tile(R, row_lim_R, tileSize_H);
	make_tile(Rt, row_lim_Rt, tileSize_W);

	//copying tiles limit rowPointers

	checkCuda(cudaEventRecord(start), __LINE__);

	checkCuda(
			cudaMalloc((void**) &d_row_lim_R,
					(total_tileInRows + 1) * (R.cols_ + 1) * sizeof(int)), 0);
	checkCuda(
			cudaMalloc((void**) &d_row_lim_Rt,
					(total_tileInCols + 1) * (R.rows_ + 1) * sizeof(int)), 0);

	checkCuda(
			cudaMemcpy(d_row_lim_R, R.get_csc_col_ptr(),
					(R.cols_ + 1) * sizeof(int), cudaMemcpyHostToDevice),
			__LINE__);
	checkCuda(
			cudaMemcpy(d_row_lim_Rt, R.get_csr_row_ptr(),
					(R.rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice),
			__LINE__);

	for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile +=
			tileSize_H) {
		int tile_no = tile / tileSize_H;	// - 1;
		checkCuda(
				cudaMemcpy(d_row_lim_R + tile_no * (R.cols_ + 1),
						&(row_lim_R[tile_no][0]), (R.cols_ + 1) * sizeof(int),
						cudaMemcpyHostToDevice), __LINE__);
	}
	for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile +=
			tileSize_W) {
		int tile_no = tile / tileSize_W;	// - 1;
		checkCuda(
				cudaMemcpy(d_row_lim_Rt + (tile_no * R.rows_) + tile_no,
						&(row_lim_Rt[tile_no][0]), (R.rows_ + 1) * sizeof(int),
						cudaMemcpyHostToDevice), __LINE__);
	}

	mili = cuda_timerEnd(start, stop, streamT);
	copyTime = mili;

	//******************PreProcess for TILED binning*******************************
	checkCuda(cudaEventRecord(start), __LINE__);
	int *tiled_count[total_tileInRows], *tiled_count_Rt[total_tileInCols];
	for (int i = 0; i < total_tileInRows; ++i)
		tiled_count[i] = (int*) malloc(NUM_THRDS * sizeof(int));
	for (int i = 0; i < total_tileInCols; ++i)
		tiled_count_Rt[i] = (int*) malloc(NUM_THRDS * sizeof(int));

	int *tiled_rowGroupPtr, *tiled_rowGroupPtr_Rt;

	// Extract CSR group info on CPU
	int *tiled_host_rowGroupPtr[total_tileInRows],
			*tiled_host_rowGroupPtr_Rt[total_tileInCols];

	for (int i = 0; i < total_tileInRows; ++i)
		tiled_host_rowGroupPtr[i] = (int*) malloc(
				NUM_THRDS * R.cols_ * sizeof(int));

	for (int i = 0; i < total_tileInCols; ++i)
		tiled_host_rowGroupPtr_Rt[i] = (int*) malloc(
				NUM_THRDS * R.rows_ * sizeof(int));

	for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile +=
			tileSize_H) {
		int tile_no = tile / tileSize_H - 1;
		tiled_binning(R, (tiled_host_rowGroupPtr[tile_no]), LB, UB,
				tiled_count[tile_no], row_lim_R, tile_no);
	}
	for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile +=
			tileSize_W) {
		int tile_no = tile / tileSize_W - 1;
		tiled_binning(Rt, (tiled_host_rowGroupPtr_Rt[tile_no]), LB_Rt, UB_Rt,
				tiled_count_Rt[tile_no], row_lim_Rt, tile_no);
	}
	checkCuda(
			cudaMalloc((void **) &tiled_rowGroupPtr,
					total_tileInRows * R.cols_ * sizeof(int)), __LINE__);
	checkCuda(
			cudaMalloc((void **) &tiled_rowGroupPtr_Rt,
					total_tileInCols * R.rows_ * sizeof(int)), __LINE__);
	int *test1 = (int*) malloc((R.cols_ + 1) * sizeof(int)); //del
	for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile +=
			tileSize_H) {
		int tile_no = tile / tileSize_H - 1;
		sum = 0;
		for (int i = 0; i < NUM_THRDS; i++) {
			if (tiled_count[tile_no][i] > 0) {
				checkCuda(
						cudaMemcpy(
								tiled_rowGroupPtr + (tile_no * R.cols_) + sum,
								&(tiled_host_rowGroupPtr[tile_no][i * R.cols_]),
								tiled_count[tile_no][i] * sizeof(int),
								cudaMemcpyHostToDevice), __LINE__);
				sum += tiled_count[tile_no][i];
			}
		}
	}

	for (int tile = tileSize_W; tile < (Rt.rows_ + tileSize_W - 1); tile +=
			tileSize_W) {
		int tile_no = tile / tileSize_W - 1;
		sum = 0;
		for (int i = 0; i < NUM_THRDS; i++) {
			if (tiled_count_Rt[tile_no][i] > 0) {
				checkCuda(
						cudaMemcpy(
								tiled_rowGroupPtr_Rt + (tile_no * Rt.cols_)
										+ sum,
								&(tiled_host_rowGroupPtr_Rt[tile_no][i * R.rows_]),
								tiled_count_Rt[tile_no][i] * sizeof(int),
								cudaMemcpyHostToDevice), __LINE__);
				sum += tiled_count_Rt[tile_no][i];
			}
		}
	}
	mili = cuda_timerEnd(start, stop, streamT);
	copyTime = mili;

	//********************STARTING CCD++ ALGORTIHM************************
	printf("tileSize_H,W: %d, %d k: %d lambda:  %f\n", tileSize_H, tileSize_W,
			k, lambda);

	float mergeR = 0, mergeRT = 0, updateR = 0, updateRT = 0;
	for (int oiter = 1; oiter <= maxiter; ++oiter) {
		int early_stop = 0, kk = 0;

		for (int tt = 0; tt < k; ++tt) {
			int t = tt;
			VecData &Wt = W[t], &Ht = H[t];
			cudaMemset(d_hArrU, 0, RRows_memsize);
			cudaMemset(d_gArrV, 0, RCols_memsize);
			cudaMemset(d_hArrV, 0, RCols_memsize);
			cudaMemset(d_gArrU, 0, RRows_memsize);

			//if (oiter > 1) 
			{
				//**************************Updating R with add true**********************************
				mergeR = 0;
				for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1);
						tile += tileSize_H) {
					int tile_no = tile / tileSize_H; //printf("*****tile no %d\n", tile_no );
					cuda_timerStart(start, streamT);
					if (t == 0)
						kk = t;
					else
						kk = t - 1;
					helper_UpdateR(
							d_row_lim_R + ((tile_no - 1) * (R.cols_ + 1)),
							d_row_lim_R + (tile_no * R.cols_) + tile_no,
							d_R_rowIdx, d_R_val, d_W + t * R.rows_,
							d_H + t * R.cols_, R.rows_, R.cols_, true,
							tiled_rowGroupPtr + ((tile_no - 1) * R.cols_),
							&(tiled_count[tile_no - 1][0]), lambda, d_gArrV,
							d_hArrV, d_W + t * R.rows_, d_W + kk * R.rows_,
							d_H + kk * R.cols_, t);

					mili = cuda_timerEnd(start, stop, streamT);
					ACSRTime += mili;
					mergeR += mili;
				}

				cuda_timerStart(start, streamT);
				assignment<<<(R.cols_ + 1023) / 1024, 1024>>>(d_R_colPtr,
						d_v_new, d_gArrV, d_hArrV, lambda, R.cols_);
				mili = cuda_timerEnd(start, stop, streamT);
				ACSRTime += mili;
				mergeR += mili;
				if (oiter == 1 && (t == 1))
					printf("time to merge R %f\n", mergeR);

				//**************************Updating RTranspose with add true**********************************
				mergeRT = 0;
				for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1);
						tile += tileSize_W) {
					int tile_no = tile / tileSize_W; //printf("tile_no from RT %d\n", tile_no);

					cuda_timerStart(start, streamT);
					if (t == 0)
						kk = t;
					else
						kk = t - 1;
					helper_UpdateR(
							d_row_lim_Rt + ((tile_no - 1) * (R.rows_ + 1)),
							d_row_lim_Rt + (tile_no * R.rows_) + (tile_no),
							d_R_colIdx, d_R_val_t, d_H + t * R.cols_,
							d_W + t * R.rows_, R.cols_, R.rows_, true,
							tiled_rowGroupPtr_Rt + ((tile_no - 1) * Rt.cols_),
							&(tiled_count_Rt[tile_no - 1][0]), lambda, d_gArrU,
							d_hArrU, d_v_new, d_H + kk * R.cols_,
							d_W + kk * R.rows_, t);
					mili = cuda_timerEnd(start, stop, streamT);
					ACSRTime += mili;
					mergeRT += mili;
					//printf("update R in GPU takes %f \n", mili );
				}
				cuda_timerStart(start, streamT);
				assignment<<<(R.cols_ + 1023) / 1024, 1024>>>(d_R_colPtr,
						d_H + t * R.cols_, d_gArrV, d_hArrV, lambda, R.cols_);
				assignment<<<(R.rows_ + 1023) / 1024, 1024>>>(d_R_rowPtr,
						d_W + t * R.rows_, d_gArrU, d_hArrU, lambda, R.rows_);
				mili = cuda_timerEnd(start, stop, streamT);
				ACSRTime += mili;
				mergeRT += mili;
				if (oiter == 1 && t == 1)
					printf("time to merge Rt %f\n", mergeRT);

			}
			int maxit = inneriter;

			float init = ACSRTime;
			int iter = 0;
			//*************************inner iter***

			// if(oiter > 1) iter = 2; 
			// else 
			iter = 1;  //maxit = inneriter;
			for (; iter < maxit; ++iter) {
				//*************************Update Ht***************  
				float updateR = 0;
				for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1);
						tile += tileSize_H) {
					int tile_no = tile / tileSize_H; //printf("*****tile no %d\n", tile_no );
					cuda_timerStart(start, streamT);
					helper_rankOneUpdate_v(
							d_row_lim_R + ((tile_no - 1) * R.cols_)
									+ (tile_no - 1),
							d_row_lim_R + (tile_no * R.cols_) + tile_no,
							d_R_rowIdx, d_R_val, d_W + t * R.rows_,
							d_H + t * R.cols_, R.rows_, R.cols_, true,
							tiled_rowGroupPtr + ((tile_no - 1) * R.cols_),
							&(tiled_count[tile_no - 1][0]), lambda, d_gArrV,
							d_hArrV, d_W + t * R.rows_);
					mili = cuda_timerEnd(start, stop, streamT);
					ACSRTime += mili;
					updateR += mili;

				}
				cuda_timerStart(start, streamT);
				assignment<<<(R.cols_ + 1023) / 1024, 1024>>>(d_R_colPtr,
						d_H + t * R.cols_, d_gArrV, d_hArrV, lambda, R.cols_);
				mili = cuda_timerEnd(start, stop, streamT);
				ACSRTime += mili;
				updateR += mili;

				//*************************Update Wt***************  

				float updateRT = 0;
				for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1);
						tile += tileSize_W) {
					int tile_no = tile / tileSize_W;//printf("tile_no from RT %d\n", tile_no);

					cuda_timerStart(start, streamT);
					helper_rankOneUpdate_v(
							d_row_lim_Rt + ((tile_no - 1) * R.rows_)
									+ (tile_no - 1),
							d_row_lim_Rt + (tile_no * R.rows_) + (tile_no),
							d_R_colIdx, d_R_val_t, d_H + t * R.cols_,
							d_W + t * R.rows_, R.cols_, R.rows_, true,
							tiled_rowGroupPtr_Rt + ((tile_no - 1) * Rt.cols_),
							&(tiled_count_Rt[tile_no - 1][0]), lambda, d_gArrU,
							d_hArrU, d_v_new);
					mili = cuda_timerEnd(start, stop, streamT);
					ACSRTime += mili;
					updateRT += mili;
				}
				cuda_timerStart(start, streamT);
				assignment<<<(R.rows_ + 1023) / 1024, 1024>>>(d_R_rowPtr,
						d_W + t * R.rows_, d_gArrU, d_hArrU, lambda, R.rows_);
				mili = cuda_timerEnd(start, stop, streamT);
				ACSRTime += mili;
				updateRT += mili;
				// if(oiter ==1 && t ==0 && iter == maxit-1)
				//  printf("time to update Wt %f\n", updateRT);
			}

			//**************************Updating R = R - Wt * Ht  *****************************

			updateR = 0;
			if (t == k - 1) {
				for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1);
						tile += tileSize_H) {
					int tile_no = tile / tileSize_H; //printf("tile no %d\n", tile_no );

					cuda_timerStart(start, streamT);
					helper_UpdateR(
							d_row_lim_R + ((tile_no - 1) * (R.cols_ + 1)),
							d_row_lim_R + (tile_no * R.cols_) + tile_no,
							d_R_rowIdx, d_R_val, d_W + t * R.rows_,
							d_H + t * R.cols_, R.rows_, R.cols_, false,
							tiled_rowGroupPtr + ((tile_no - 1) * R.cols_),
							&(tiled_count[tile_no - 1][0]), lambda, d_gArrU,
							d_hArrU, d_W + t * R.rows_, d_W + t * R.rows_,
							d_H + t * R.cols_, t);
					mili = cuda_timerEnd(start, stop, streamT);
					ACSRTime += mili;
					updateR += mili;
				}
			}

			//**************************Updating RT = RT - Wt * Ht  *****************************

			updateRT = 0;
			if (t == k - 1) {
				for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1);
						tile += tileSize_W) {
					int tile_no = tile / tileSize_W;
					cuda_timerStart(start, streamT);
					helper_UpdateR(
							d_row_lim_Rt + ((tile_no - 1) * (R.rows_ + 1)),
							d_row_lim_Rt + (tile_no * R.rows_) + (tile_no),
							d_R_colIdx, d_R_val_t, d_H + t * R.cols_,
							d_W + t * R.rows_, R.cols_, R.rows_, false,
							tiled_rowGroupPtr_Rt + ((tile_no - 1) * Rt.cols_),
							&(tiled_count_Rt[tile_no - 1][0]), lambda, d_gArrU,
							d_hArrU, d_H + t * R.cols_, d_H + t * R.cols_,
							d_W + t * R.rows_, t);
					mili = cuda_timerEnd(start, stop, streamT);
					ACSRTime += mili;
					updateRT += mili;
				}
			}
			// if(oiter ==1 && t == k-1)
			// 	printf("time to update Rt %f\n", updateRT);
			if (oiter == 1 && t == 2)
				printf("iter %d time for 1 feature: %f ms\n", oiter, ACSRTime);

		}

		//**************Check RMSE******************** 
		cudaMemset(d_rmse, 0, (T.nnz_ + 1) * sizeof(DTYPE));
		cudaMemset(d_pred_v, 0, (T.nnz_ + 1) * sizeof(DTYPE));

		GPU_rmse<<<(T.nnz_ + 1023) / 1024, 1024>>>(d_test_row, d_test_col,
				d_test_val, d_pred_v, d_rmse, d_W, d_H, T.nnz_, k, R.rows_,
				R.cols_);
		DTYPE tot_rmse = 0, f_rmse = 0;
		cudaMemcpy(&(rmse[0]), d_rmse, (T.nnz_ + 1) * sizeof(DTYPE),
				cudaMemcpyDeviceToHost);

#pragma omp parallel for reduction(+:tot_rmse)
		for (int i = 0; i < T.nnz_; ++i)
			tot_rmse += rmse[i];
		f_rmse = sqrt(tot_rmse / T.nnz_);
		printf("iter %d time %f RMSE %f\n", oiter, (ACSRTime / 1000), f_rmse);

	}

	for (int i = 0; i <= NUM_THRDS; i++)
		checkCuda(cudaStreamDestroy(stream[i]), __LINE__);
	checkCuda(cudaStreamDestroy(streamT), __LINE__);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_W);
	cudaFree(d_H);
	cudaFree(d_R_rowIdx);
	cudaFree(d_R_colPtr);
	cudaFree(d_R_val);
	cudaFree(d_R_colIdx);
	cudaFree(d_R_rowPtr);
	cudaFree(d_R_val_t);
	cudaFree(d_gArrU);
	cudaFree(d_gArrV);
	cudaFree(d_hArrU);
	cudaFree(d_hArrV);

}

