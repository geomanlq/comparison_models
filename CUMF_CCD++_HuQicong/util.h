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

#ifndef UTIL
#define UTIL

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <memory>
#include <limits>
#include <vector>
#include <omp.h>
#include <cassert>

inline double seconds() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#define DTYPE float
//#define DTYPE double

using std::vector;
using std::ifstream;

class TestData;
class SparseMatrix;
class Options;

using VecData = vector<DTYPE>;
using MatData = vector<VecData>;
using VecInt = vector<int>;
using MatInt = vector<VecInt>;

void ccdr1(SparseMatrix &R, MatData &W, MatData &H, TestData &T,
		Options &options);
void load_from_binary(const char* srcdir, SparseMatrix &R, TestData &data);
MatData load_mat_t(FILE *fp, bool row_major = true);
void init_random(MatData &X, long k, long n);

class SparseMatrix {

public:
	long rows_, cols_, nnz_, max_row_nnz_, max_col_nnz_;

	void read_binary_file(long rows, long cols, long nnz,
			std::string fname_csr_row_ptr,
			std::string fname_csr_col_indx, std::string fname_csr_val,
			std::string fname_csc_col_ptr, std::string fname_csc_row_indx,
			std::string fname_csc_val);

	SparseMatrix get_shallow_transpose();

	int* get_csc_col_ptr() const {
		return csc_col_ptr_.get();
	}

	unsigned int* get_csc_row_indx() const {
		return csc_row_indx_.get();
	}

	DTYPE* get_csc_val() const {
		return csc_val_.get();
	}

	unsigned int* get_csr_col_indx() const {
		return csr_col_indx_.get();
	}

	int* get_csr_row_ptr() const {
		return csr_row_ptr_.get();
	}

	DTYPE* get_csr_val() const {
		return csr_val_.get();
	}

private:

	void read_compressed(std::string fname_cs_ptr, std::string fname_cs_indx,
			std::string fname_cs_val, std::shared_ptr<int>&cs_ptr,
			std::shared_ptr<unsigned int> &cs_indx,
			std::shared_ptr<DTYPE>& cs_val, long num_elems_in_cs_ptr,
			long &max_nnz_in_one_dim);

	std::shared_ptr<int> csc_col_ptr_, csr_row_ptr_, col_nnz_, row_nnz_;
	std::shared_ptr<DTYPE> csr_val_, csc_val_;
	std::shared_ptr<unsigned int> csc_row_indx_, csr_col_indx_;
};

class TestData {

public:
	long rows_ { 0 }, cols_ { 0 }, nnz_ { 0 };

	void read_binary_file(long rows, long cols, long nnz,
		std::string fname_coo_row, std::string fname_coo_col,
		std::string fname_coo_val);
	
	// void read(long rows, long cols, long nnz, std::string filename) {
	// 	this->rows_ = rows;
	// 	this->cols_ = cols;
	// 	this->nnz_ = nnz;

	// 	test_row = std::unique_ptr<unsigned int[]>(new unsigned int[nnz]);
	// 	test_col = std::unique_ptr<unsigned int[]>(new unsigned int[nnz]);
	// 	test_val = std::unique_ptr<DTYPE[]>(new DTYPE[nnz]);

	// 	ifstream fp(filename);
	// 	for (long idx = 0; idx < nnz; ++idx) {
	// 		fp >> test_row[idx] >> test_col[idx] >> test_val[idx];
	// 	}
	// }

	unsigned int* getTestCol() const {
		return test_col.get();
	}

	unsigned int* getTestRow() const {
		return test_row.get();
	}

	DTYPE* getTestVal() const {
		return test_val.get();
	}

private:

	void read_compressed(std::string fname_coo_row,
		std::string fname_coo_col, std::string fname_coo_val,
		std::unique_ptr<unsigned int>&coo_row, std::unique_ptr<unsigned int> &coo_col,
		std::unique_ptr<DTYPE>& coo_val);

	std::unique_ptr<unsigned int> test_row, test_col;
	std::unique_ptr<DTYPE> test_val;

};

class Options {
public:
	int k = 10;
	int maxiter = 5;
	int maxinneriter = 1;DTYPE lambda = .05;
	int tileSizeW = 499999999;
	int tileSizeH = 499999999;

	void print() {
		std::cout << "k = " << k << '\n';
		std::cout << "iterinner = " << maxinneriter << '\n';
		std::cout << "outeriter = " << maxiter << '\n';
		std::cout << "tsw = " << tileSizeW << '\n';
		std::cout << "tsh = " << tileSizeH << '\n';
		std::cout << "lambda = " << lambda << '\n';
	}
};
#endif

