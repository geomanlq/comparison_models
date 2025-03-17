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

#ifndef UTILS_EXTRA_HPP_
#define UTILS_EXTRA_HPP_

template<int A, int B>
struct TMP_power {
	static const int value = A * TMP_power<A, B - 1>::value;
};
template<int A>
struct TMP_power<A, 0> {
	static const int value = 1;
};

template<class T>
inline constexpr T cexp_pow(const T base, unsigned const exponent) {
	// (parentheses not required in next line)
	return (exponent == 0) ? 1 : (base * pow(base, exponent - 1));
}

template<unsigned LB, unsigned UB>
struct LOOP {
	template<typename FUNCTOR>
	LOOP(FUNCTOR &&fun) {
		static_assert(LB<UB,"Lower Bound should be less than Upper bound");
		fun(LB);
		LOOP<LB + 1, UB>(std::forward < FUNCTOR > (fun));
	}
};

template<unsigned LB>
struct LOOP<LB, LB> {
	template<typename FUNCTOR>
	LOOP(FUNCTOR /*fun*/) {
		//do nothing
	}
};

#endif /* UTILS_EXTRA_HPP_ */
