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
#include <cuda.h>
#include <cstring>
#include <time.h>
#include <sys/time.h>

void print_help_and_exit() {
	printf("options:\n\
	    -k rank/feature : set the rank (default 10)\n\
        -l lambda : set the regularization parameter lambda (default 0.05)\n\
        -a tile size: set tile size for input matrix R (default 499999999)\n\
        -b tile size: set tile size for input matrix R Transpose (default 499999999)\n\
        -t max_iter: number of iterations (default 5)\n\
        -T max_iter: number of inner iterations (default 1)\n");
	exit(1);
}

Options parse_cmd_options(int argc, char **argv, char *train_file_name,
		char *test_file_name) {
	Options param;
	int i;
	//handle options
	for (i = 1; i < argc; i++) {
		if (argv[i][0] != '-')
			break;
		if (++i >= argc)
			print_help_and_exit();
		switch (argv[i - 1][1]) {
		case 'k':
			param.k = atoi(argv[i]);
			break;
		case 'l':
			param.lambda = atof(argv[i]);
			break;

		case 't':
			param.maxiter = atoi(argv[i]);
			break;

		case 'T':
			param.maxinneriter = atoi(argv[i]);
			break;

		case 'a':
			param.tileSizeH = atoi(argv[i]);
			break;

		case 'b':
			param.tileSizeW = atoi(argv[i]);
			break;

		default:
			fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
			print_help_and_exit();
			break;
		}
	}

	if (i >= argc)
		print_help_and_exit();

	strcpy(train_file_name, argv[i]);
	// strcpy(test_file_name, argv[i + 1]);
	return param;
}

void run_ccdr1(Options &param, const char* train_file_name,
		char* test_file_name) {

	//cudaEvent_t start, stop; float elapsedTime;
	clock_t start, end;
	double cpu_time_used;
	DTYPE *h_a, *d_W, *h_c, *d_H, *d_R;
	struct timeval t1, t2;
	SparseMatrix R;
	MatData W, H;
	TestData testdata;

	load_from_binary(train_file_name, R, testdata);
	// W, H  here are k*m, k*n
	init_random(W, param.k, R.rows_);
	init_random(H, param.k, R.cols_);

	puts("starts!");
	double t0 = seconds();
	ccdr1(R, W, H, testdata, param);
	printf("\nTotal seconds: %.3f for F= %d\n\n", seconds() - t0, param.k);
	return;
}

int main(int argc, char* argv[]) {
	char train_file_name[1024];
	char test_file_name[1024];
	Options options = parse_cmd_options(argc, argv, train_file_name,
			test_file_name);

	options.print();

	run_ccdr1(options, train_file_name, test_file_name);
	return 0;
}

