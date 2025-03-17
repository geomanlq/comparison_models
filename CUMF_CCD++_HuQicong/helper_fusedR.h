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

#include "cuda_fusedR.h"
#include "utils_extra.hpp"
#include <cmath>

const int maxThreadsPerBlock = 1024;
int BLOCKSIZE = 128;
cudaStream_t stream[10 + 1]; //hard coded
dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
void create_stream() {
	for (int i = 0; i < NUM_THRDS; i++) {
		cudaStreamCreate(&(stream[i]));
	}
	cudaStreamCreate(&(stream[NUM_THRDS]));
}

template<unsigned LB, unsigned UB>
struct RANK_LOOP {
	RANK_LOOP() = delete;
	RANK_LOOP(int& sum, const int * __restrict__ d_R_colPtr, int *d_row_lim,
			unsigned *d_R_rowIdx, DTYPE *d_R_val, const DTYPE *d_Wt,
			const DTYPE *d_Ht, int m, int n, bool add, int *rowGroupPtr,
			int *count, DTYPE lambda, DTYPE *d_gArrV, DTYPE *d_hArrV,
			DTYPE *v_new, DTYPE *Wt_p, DTYPE *Ht_p, int t) {
		static_assert(LB<UB,"Lower Bound should be less than Upper bound");
		if (count[LB] > 0) {
			constexpr unsigned BLOCKSIZE_V2 = 128;
			constexpr unsigned POWER = TMP_power<2, LB>::value;
			grid.x = (POWER * count[LB] + BLOCKSIZE_V2 - 1) / BLOCKSIZE_V2;
			updateR_gen<false, POWER, LB> <<<grid, BLOCKSIZE_V2, 0, stream[LB]>>>(
					d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val, d_Wt, d_Ht, m,
					n, add, rowGroupPtr + sum, count[LB], lambda, d_gArrV,
					d_hArrV, v_new, Wt_p, Ht_p, t);
		}
		sum += count[LB];
		RANK_LOOP<LB + 1, UB>(sum, d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val,
				d_Wt, d_Ht, m, n, add, rowGroupPtr, count, lambda, d_gArrV,
				d_hArrV, v_new, Wt_p, Ht_p, t);
	}
};

template<unsigned LB>
struct RANK_LOOP<LB, LB> {
	RANK_LOOP() = delete;
	RANK_LOOP(int& sum, const int * __restrict__ d_R_colPtr, int *d_row_lim,
			unsigned *d_R_rowIdx, DTYPE *d_R_val, const DTYPE *d_Wt,
			const DTYPE *d_Ht, int m, int n, bool add, int *rowGroupPtr,
			int *count, DTYPE lambda, DTYPE *d_gArrV, DTYPE *d_hArrV,
			DTYPE *v_new, DTYPE *Wt_p, DTYPE *Ht_p, int t) {
		//do nothing
	}
};

void helper_UpdateR(int *d_R_colPtr, int *d_row_lim, unsigned *d_R_rowIdx,
		DTYPE *d_R_val, DTYPE *d_Wt, DTYPE *d_Ht, int m, int n, bool add,
		int *rowGroupPtr, int *count, DTYPE lambda, DTYPE *d_gArrV,
		DTYPE *d_hArrV, DTYPE *v_new, DTYPE *Wt_p, DTYPE *Ht_p, int t) {
	int sum = 0;
	//loop from 0 to 5
	RANK_LOOP<0, 6>(sum, d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val, d_Wt, d_Ht,
			m, n, add, rowGroupPtr, count, lambda, d_gArrV, d_hArrV, v_new,
			Wt_p, Ht_p, t);

	if (count[6] > 0) {
		grid.x = (64 * count[6] + BLOCKSIZE - 1) / BLOCKSIZE;
		updateR_7<false> <<<grid, block, 2 * block.x / 32 * sizeof(DTYPE),
				stream[6]>>>(d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val, d_Wt,
				d_Ht, m, n, add, rowGroupPtr + sum, count[6], lambda, d_gArrV,
				d_hArrV, v_new, Wt_p, Ht_p, t);
	}
	sum += count[6];

	if (count[7] > 0) {
		grid.x = (64 * count[7] + BLOCKSIZE - 1) / BLOCKSIZE;
		updateR_7<false> <<<grid, block, 2 * block.x / 32 * sizeof(DTYPE),
				stream[7]>>>(d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val, d_Wt,
				d_Ht, m, n, add, rowGroupPtr + sum, count[7], lambda, d_gArrV,
				d_hArrV, v_new, Wt_p, Ht_p, t);
	}
	sum += count[7];

	if (count[8] > 0) {
		grid.x = (64 * count[8] + BLOCKSIZE - 1) / BLOCKSIZE;
		updateR_7<false> <<<grid, block, 2 * block.x / 32 * sizeof(DTYPE),
				stream[8]>>>(d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val, d_Wt,
				d_Ht, m, n, add, rowGroupPtr + sum, count[8], lambda, d_gArrV,
				d_hArrV, v_new, Wt_p, Ht_p, t);
	}
	sum += count[8];
	if (count[9] > 0) {
		grid.x = (64 * count[9] + BLOCKSIZE - 1) / BLOCKSIZE;
		updateR_7<false> <<<grid, block, 2 * block.x / 32 * sizeof(DTYPE),
				stream[9]>>>(d_R_colPtr, d_row_lim, d_R_rowIdx, d_R_val, d_Wt,
				d_Ht, m, n, add, rowGroupPtr + sum, count[9], lambda, d_gArrV,
				d_hArrV, v_new, Wt_p, Ht_p, t);
	}
}
