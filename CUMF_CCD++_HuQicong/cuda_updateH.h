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

template<bool UseCache, unsigned POWER, unsigned Nth_POWER>
__global__ void CudaRankOneUpdate_gen(int const* __restrict__ R_colPtr,
		int const* __restrict__ R_rowLim,
		const unsigned * __restrict__ R_rowIdx, DTYPE *R_val,
		const DTYPE * __restrict__ u, const DTYPE * __restrict__ v, int m,
		int n, bool add, int * __restrict__ rowGroupPtr, int numRowsPerGroup,
		DTYPE lambda, DTYPE * __restrict__ g_arr, DTYPE * __restrict__ h_arr,
		const DTYPE * __restrict__ u_new) {
	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & (POWER - 1);
	unsigned int c = (blockIdx.x * blockDim.x + tId) >> (Nth_POWER);

	if (c < numRowsPerGroup) {
		c = rowGroupPtr[c];
		DTYPE g = 0, h = 0;
		DTYPE vc = v[c];
		unsigned int colPtr = R_colPtr[c];
		unsigned int nnz_row = R_rowLim[c] - R_colPtr[c]; //nnz_row = R_colPtr[c+1] - colPtr;

		for (unsigned short i = laneId; i < nnz_row; i += POWER) {
			int ii = R_rowIdx[colPtr + i];
			DTYPE u_val = __ldg(&u[ii]);
			g += u_val * R_val[colPtr + i];
			h += u_val * u_val;
		}
#pragma unroll Nth_POWER
		for (unsigned i = (POWER) >> 1; i >= 1; i = i >> 1) {
			g += __shfl_down(g, i);
			h += __shfl_down(h, i);
		}
		if (laneId == 0) {
			h += lambda * (nnz_row);
			g_arr[c] += g;
			h_arr[c] += h;
		}
	}
}

template<bool UseCache>
__global__ void CudaRankOneUpdate_7(int const* __restrict__ R_colPtr,
		int const* __restrict__ R_rowLim, unsigned * R_rowIdx, DTYPE *R_val,
		DTYPE * u, const DTYPE * __restrict__ v, int m, int n, bool add,
		int * __restrict__ rowGroupPtr, int numRowsPerGroup, DTYPE lambda,
		DTYPE * __restrict__ g_arr, DTYPE * __restrict__ h_arr,
		const DTYPE * __restrict__ u_new) {
	unsigned int tId = threadIdx.x;
	unsigned int laneId = tId & 63;
	unsigned int c = (blockIdx.x * blockDim.x + tId) >> 6;
	extern __shared__   volatile DTYPE SD[];

	if (c < numRowsPerGroup) {
		c = rowGroupPtr[c];
		DTYPE g = 0, h = 0;
		DTYPE vc = v[c];
		unsigned int colPtr = R_colPtr[c], nnz_row = R_rowLim[c] - colPtr; //nnz_row = R_colPtr[c+1] - colPtr;
		for (long i = laneId; i < nnz_row; i += 64) {
			int ii = R_rowIdx[colPtr + i];
			DTYPE u_val = u[ii];
			g += u_val * R_val[colPtr + i];
			h += u_val * u_val;
		}
		DTYPE newvj = 0;
		g += __shfl_down(g, 16);
		g += __shfl_down(g, 8);
		g += __shfl_down(g, 4);
		g += __shfl_down(g, 2);
		g += __shfl_down(g, 1);

		h += __shfl_down(h, 16);
		h += __shfl_down(h, 8);
		h += __shfl_down(h, 4);
		h += __shfl_down(h, 2);
		h += __shfl_down(h, 1);

		if ((tId & 31) == 0) {
			SD[tId >> 5] = g;
			SD[blockDim.x / 32 + (tId >> 5)] = h;
		}
		__syncthreads();
		if (laneId == 0) {
			g += SD[(tId >> 5) + 1];
			h += SD[(blockDim.x / 32 + (tId >> 5)) + 1];
			h += lambda * (nnz_row);
			g_arr[c] += g;
			h_arr[c] += h;
		}
	}
}
