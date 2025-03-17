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
#define type DTYPE
#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

// load utility
void load_from_binary(const char* srcdir, SparseMatrix &R, TestData &data) {
	printf("Loading from binary file...\n");
	char filename[1024], buf[1024];

	char binary_test_coo_filename_row[1024],
			binary_test_coo_filename_col[1024], binary_test_coo_filename_val[1024],
			demo[1024];
	char binary_filename_rowptr[1024], binary_filename_colidx[1024],
			binary_filename_csrval[1024];
	char binary_filename_colptr[1024], binary_filename_rowidx[1024],
			binary_filename_cscval[1024];
	sprintf(filename, "%s/meta_modified_all", srcdir);
	FILE *fp = fopen(filename, "r");
	long m, n, nnz, test_nnz;
	fscanf(fp, "%ld %ld", &m, &n);
	fscanf(fp, "%ld", &nnz);
	fscanf(fp, "%ld", &test_nnz);

	// read names of train datasets
	fscanf(fp, "%s", buf);
	sprintf(binary_filename_rowptr, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_filename_colidx, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_filename_csrval, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_filename_colptr, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_filename_rowidx, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_filename_cscval, "%s/%s", srcdir, buf);

	R.read_binary_file(m, n, nnz, binary_filename_rowptr, binary_filename_colidx,
		binary_filename_csrval, binary_filename_colptr,
		binary_filename_rowidx, binary_filename_cscval);

	// read names of test datasets
	fscanf(fp, "%s", buf);
	sprintf(binary_test_coo_filename_row, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_test_coo_filename_col, "%s/%s", srcdir, buf);
	fscanf(fp, "%s", buf);
	sprintf(binary_test_coo_filename_val, "%s/%s", srcdir, buf);
	// fscanf(fp, "%s", buf);
	// sprintf(demo, "%s/%s", srcdir, buf);

	data.read_binary_file(m, n, test_nnz, binary_test_coo_filename_row, binary_test_coo_filename_col,
		binary_test_coo_filename_val);
	fclose(fp);

	return;
}

void init_random(MatData &X, long k, long n) {
	X = MatData(k, VecData(n));
	srand48(0L);
	for (long i = 0; i < n; ++i)
		for (long j = 0; j < k; ++j)
			X[j][i] = 0.1 * drand48();
}


void SparseMatrix::read_binary_file(long rows, long cols, long nnz,
		std::string fname_csr_row_ptr, std::string fname_csr_col_indx,
		std::string fname_csr_val, std::string fname_csc_col_ptr,
		std::string fname_csc_row_indx, std::string fname_csc_val) {
	this->rows_ = rows;
	this->cols_ = cols;
	this->nnz_ = nnz;

	/// read csr
	this->read_compressed(fname_csr_row_ptr, fname_csr_col_indx, fname_csr_val,
			this->csr_row_ptr_, this->csr_col_indx_, this->csr_val_, rows + 1,
			this->max_row_nnz_);

	/// read csc
	this->read_compressed(fname_csc_col_ptr, fname_csc_row_indx, fname_csc_val,
			this->csc_col_ptr_, this->csc_row_indx_, this->csc_val_, cols + 1,
			this->max_col_nnz_);

}

void SparseMatrix::read_compressed(std::string fname_cs_ptr,
		std::string fname_cs_indx, std::string fname_cs_val,
		std::shared_ptr<int>&cs_ptr, std::shared_ptr<unsigned int> &cs_indx,
		std::shared_ptr<DTYPE>& cs_val, long num_elems_in_cs_ptr,
		long &max_nnz_in_one_dim) {

	cs_ptr = std::shared_ptr<int>(new int[num_elems_in_cs_ptr],
			std::default_delete<int[]>());
	cs_indx = std::shared_ptr<unsigned int>(new unsigned int[this->nnz_],
			std::default_delete<unsigned int[]>());
	cs_val = std::shared_ptr<DTYPE>(new DTYPE[this->nnz_],
			std::default_delete<DTYPE[]>());

	std::ifstream f_indx(fname_cs_indx, std::ios::binary);
	std::ifstream f_val(fname_cs_val, std::ios::binary);

	for (std::size_t i = 0; i < this->nnz_; i++) {
		f_indx.read((char*) &cs_indx.get()[i], sizeof(unsigned int));
		f_val.read((char *) &cs_val.get()[i], sizeof(float));
	}

	std::ifstream f_ptr(fname_cs_ptr, std::ios::binary);

	int prev, cur = 0;
	max_nnz_in_one_dim = std::numeric_limits<long>::min();

	for (std::size_t i = 0; i < num_elems_in_cs_ptr; i++) {
		prev = cur;
		f_ptr.read((char*) &cur, sizeof(int));
		cs_ptr.get()[i] = cur;

		if (i > 0)
			max_nnz_in_one_dim = std::max<long>(max_nnz_in_one_dim, cur - prev);
	}
}

SparseMatrix SparseMatrix::get_shallow_transpose() {
	SparseMatrix shallow_transpose;
	shallow_transpose.cols_ = rows_;
	shallow_transpose.rows_ = cols_;
	shallow_transpose.nnz_ = nnz_;
	shallow_transpose.csc_val_ = csr_val_;
	shallow_transpose.csr_val_ = csc_val_;
	shallow_transpose.csc_col_ptr_ = csr_row_ptr_;
	shallow_transpose.csr_row_ptr_ = csc_col_ptr_;
	shallow_transpose.csr_col_indx_ = csc_row_indx_;
	shallow_transpose.csc_row_indx_ = csr_col_indx_;
	shallow_transpose.max_col_nnz_ = max_row_nnz_;
	shallow_transpose.max_row_nnz_ = max_col_nnz_;

	return shallow_transpose;
}
//Change shared to unique

void TestData::read_compressed(std::string fname_coo_row,
	std::string fname_coo_col, std::string fname_coo_val,
	std::unique_ptr<unsigned int>&coo_row, std::unique_ptr<unsigned int> &coo_col,
	std::unique_ptr<DTYPE>& coo_val) {

	test_row = std::unique_ptr<unsigned int>(new unsigned int[this->nnz_]);
	test_col = std::unique_ptr<unsigned int>(new unsigned int[this->nnz_]);
	test_val = std::unique_ptr<DTYPE>(new DTYPE[this->nnz_]);

	std::ifstream f_row(fname_coo_row, std::ios::binary);
	std::ifstream f_col(fname_coo_col, std::ios::binary);
	std::ifstream f_val(fname_coo_val, std::ios::binary);

	for (std::size_t i = 0; i < this->nnz_; i++) {
		f_row.read((char*) &test_row.get()[i], sizeof(unsigned int));
		f_col.read((char*) &test_col.get()[i], sizeof(unsigned int));
		f_val.read((char *) &test_val.get()[i], sizeof(float));
	}
}

void TestData::read_binary_file(long rows, long cols, long nnz,
		std::string fname_coo_row, std::string fname_coo_col,
		std::string fname_coo_val) {
	this->rows_ = rows;
	this->cols_ = cols;
	this->nnz_ = nnz;

	/// read coo
	this->read_compressed(fname_coo_row, fname_coo_col, fname_coo_val,
			this->test_row, this->test_col, this->test_val);

}



