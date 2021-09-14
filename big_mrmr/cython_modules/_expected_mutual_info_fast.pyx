# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#
# Authors: Robert Layton <robertlayton@gmail.com>
#           Corey Lynch <coreylynch9@gmail.com>
# License: BSD 3 clause
# Comment: Taken originally from https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/metrics/cluster/_expected_mutual_info_fast.pyx
# All credits to the authors above.

from libc.math cimport exp, lgamma
#from libc.stdio cimport printf
from scipy.special import gammaln
import numpy as np
cimport numpy as np
cimport cython

np.import_array()
ctypedef np.float64_t DOUBLE


def expected_mutual_information(np.ndarray[np.int32_t] a, np.ndarray[np.int32_t] b, int n_samples, int R, int C):
    """Calculate the expected mutual information for two labelings."""
    cdef DOUBLE N, gln_N, emi, term2, term3, gln
    cdef np.ndarray[DOUBLE] gln_a, gln_b, gln_Na, gln_Nb, gln_nij, log_Nnij
    cdef np.ndarray[DOUBLE] nijs, term1
    cdef np.ndarray[DOUBLE] log_a, log_b    

    #cdef np.ndarray[int, ndim=2] start, end
    N = <DOUBLE>n_samples
    a = np.ravel(a.astype(np.int32, copy=False))
    b = np.ravel(b.astype(np.int32, copy=False))
    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')

    nijs[0] = 1  # Stops divide by zero warnings. As its not used, no issue.
    # term1 is nij / N
    term1 = nijs / N
    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    log_a = np.log(a)
    log_b = np.log(b)
    # term2 uses log(N * nij) = log(N) + log(nij)
    log_Nnij = np.log(N) + np.log(nijs)
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = gammaln(a + 1)
    gln_b = gammaln(b + 1)
    gln_Na = gammaln(N - a + 1)
    gln_Nb = gammaln(N - b + 1)
    gln_N = gammaln(N + 1)
    gln_nij = gammaln(nijs + 1)
    # last line executed before memory explosion
    # start and end values for nij terms for each summation.
        
    end = np.minimum(np.resize(a, (C, R)).T, np.resize(b, (R, C))) + 1
    # emi itself is a summation over the various values.
    emi = 0.0
    cdef Py_ssize_t i, j, nij
    for i in range(R):
        for j in range(C):
            for nij in range(np.maximum(a[i] - N + b[j], 1), end[i,j]):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                     - gln_N - gln_nij[nij] - lgamma(a[i] - nij + 1)
                     - lgamma(b[j] - nij + 1)
                     - lgamma(N - a[i] - b[j] + nij + 1))
                term3 = exp(gln)
                emi += (term1[nij] * term2 * term3)   
    return emi