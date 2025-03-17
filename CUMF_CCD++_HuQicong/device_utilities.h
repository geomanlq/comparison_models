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

#include "util.h"
const int THREADLOAD = 2;

int NUM_THRDS = 10;
void cuda_timerStart(cudaEvent_t start, cudaStream_t streamT) {
	cudaEventRecord(start, streamT);
}
float cuda_timerEnd(cudaEvent_t start, cudaEvent_t stop, cudaStream_t streamT) {
	float mili = 0;
	cudaDeviceSynchronize();
	cudaEventRecord(stop, streamT);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mili, start, stop);
	return mili;

}
void copy_R(SparseMatrix &R, DTYPE *copy_R) //R to R copy
		{
	auto val_ptr = R.get_csr_val();
#pragma omp parallel for
	for (int c = 0; c < R.cols_; ++c) {
		for (int idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1];
				++idx)
			copy_R[idx] = val_ptr[idx];
	}
}

void copy_R1(DTYPE *copy_R, SparseMatrix &R) {
	auto val_ptr = R.get_csr_val();
#pragma omp parallel for
	for (int c = 0; c < R.cols_; ++c) {
		for (int idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1];
				++idx)
			val_ptr[idx] = copy_R[idx];
	}
}

void make_tile(SparseMatrix &R, MatInt &tiled_bin, const int TS) {
#pragma omp parallel for
	for (int c = 0; c < R.cols_; ++c) {
		long idx = R.get_csc_col_ptr()[c];
		tiled_bin[0][c] = idx;
		for (int tile = TS; tile < (R.rows_ + TS - 1); tile += TS) {
			int tile_no = tile / TS; // - 1;
			while (R.get_csc_row_indx()[idx] < tile
					&& idx < R.get_csc_col_ptr()[c + 1]) {
				idx++;
			}
			tiled_bin[tile_no][c] = idx;
		}
	}
}

void make_tile_odd(SparseMatrix &R, MatInt &tiled_bin, const int TS) {
#pragma omp parallel for
	for (int c = 0; c < R.cols_; ++c) {
		long idx = R.get_csc_col_ptr()[c];
		tiled_bin[0][c] = idx;
		for (int tile = TS + (TS / 2); tile < (R.rows_ + (TS + (TS / 2)) - 1);
				tile += TS) {
			int tile_no = tile / TS; // - 1;
			while (R.get_csc_row_indx()[idx] < tile
					&& idx < R.get_csc_col_ptr()[c + 1]) {
				idx++;
			}
			tiled_bin[tile_no][c] = idx;
		}
	}
}

void tiled_binning(SparseMatrix &R, int *host_rowGroupPtr, int *LB, int *UB,
		int *count, MatInt &tiled_bin, const int tile_no) {
	for (int i = 0; i < NUM_THRDS; i++) {
		count[i] = 0;
		UB[i] = (1 << i) * THREADLOAD;
		LB[i] = UB[i] >> 1;
	}
	LB[0] = 0;
	UB[NUM_THRDS - 1] = R.max_col_nnz_ + 1;
	// // // // //***********binned
	// omp_set_num_threads(NUM_THRDS);  // create as many CPU threads as there are # of bins
	// #pragma omp parallel
	// {
	// unsigned int cpu_thread_id = omp_get_thread_num();
	// int i = cpu_thread_id; count[i] = 0;
	// for (int col = 0; col < R.cols; col++){
	// //for (int col = tile_no_c*5*tileSize_H; col < ((tile_no_c+1)*5*tileSize_H) && col < R.cols ; col++){
	//         int NNZ = tiled_bin[tile_no+1][col] -  tiled_bin[tile_no][col]; // R.col_ptr[col + 1] - R.col_ptr[col];
	//         if (NNZ >= LB[i] && NNZ < UB[i]){
	//             host_rowGroupPtr[R.cols * i + count[i]++] = col;
	//          }
	//     }
	// }

	//*********non-binned
	int i = 6;
	count[i] = 0;
	for (int col = 0; col < R.cols_; col++) {
		host_rowGroupPtr[R.cols_ * i + count[i]++] = col;
	}

	//*********non-binned

	// int i = 6;
	// count[i] = 0;
	// for (int col = 0; col < R.cols; col++){
	//     int NNZ = R.col_ptr[col+1] -  R.col_ptr[col];
	//     host_rowGroupPtr[R.cols * i + count[i]++] = col;
	//     printf("%d %d\n",col, NNZ );
	//  }
	//  printf("done for R\n");
}

void binning(SparseMatrix &R, int *host_rowGroupPtr, int *LB, int *UB,
		int *count) {
	for (int i = 0; i < NUM_THRDS; i++) {
		count[i] = 0;
		UB[i] = (1 << i) * THREADLOAD + 1;
		LB[i] = UB[i] >> 1;

	}
	LB[0] = 0;
	UB[NUM_THRDS - 1] = R.max_col_nnz_ + 1;

	omp_set_num_threads(NUM_THRDS); // create as many CPU threads as there are # of bins
#pragma omp parallel
	{
		unsigned int cpu_thread_id = omp_get_thread_num();
		int i = cpu_thread_id;
		for (int col = 0; col < R.cols_; col++) {
			int NNZ = R.get_csc_col_ptr()[col + 1] - R.get_csc_col_ptr()[col];
			if (NNZ > LB[i] && NNZ < UB[i]) {
				host_rowGroupPtr[R.cols_ * i + count[i]++] = col;    ////changed
			}
		}
	}
}
__global__ void weighted_H_all(int const* __restrict__ R_colPtr,
DTYPE * __restrict__ H, DTYPE * __restrict__ temp_H, int m, int k) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < m) {
		int nnz = R_colPtr[c + 1] - R_colPtr[c];
		if (nnz != 0) {
			for (int t = 0; t < k; ++t)
				H[c * k + t] = temp_H[c * k + t] / nnz;
		}
	}
}

__global__ void weighted_H(int const* __restrict__ R_colPtr,
		int const* __restrict__ R_rowLim, DTYPE * __restrict__ H,
		DTYPE * __restrict__ temp_H, int m, int k) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < m) {
		int nnz = R_rowLim[c] - R_colPtr[c];    ////////////-R_colPtr[c];
		if (nnz != 0) {
			for (int t = 0; t < k; ++t)
				H[c * k + t] = temp_H[c * k + t] / nnz;
		}
	}
}

__global__ void assignment(int const* __restrict__ R_colPtr,
DTYPE * __restrict__ v, DTYPE * __restrict__ g, DTYPE *__restrict__ h,
DTYPE lambda, int m) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < m) {
		DTYPE gc = g[c], hc = h[c];
		if (hc == 0)
			v[c] = 0; //
		else
			v[c] = gc / hc;
	}
}

__global__ void GPU_rmse(int const* __restrict__ test_row,
		int const * __restrict__ test_col, DTYPE const * __restrict__ test_val,
		DTYPE * __restrict__ pred_v, DTYPE * __restrict__ rmse,
		DTYPE const * __restrict__ W, DTYPE const * __restrict__ H, int m,
		int k, int rows, int cols) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < m) {
		for (int t = 0; t < k; t++) {
			int i = test_row[c] + 1;
			int j = test_col[c] + 1;
			pred_v[c] += W[t * rows + (i - 1)] * H[t * cols + (j - 1)]; //W[i-1][t] * H[j-1][t];
		}
		rmse[c] = (pred_v[c] - test_val[c]) * (pred_v[c] - test_val[c]);
	}
}
