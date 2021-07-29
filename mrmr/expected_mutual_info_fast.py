import math
import numpy as np
from scipy.special import gammaln

def expected_mutual_information_py(a, b, n_samples, R, C):
    """Calculate the expected mutual information for two labelings."""

    #cdef np.ndarray[int, ndim=2] start, end
    N = n_samples
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
    #printf("%s\n", "BEFORE FOR")
    emi = 0.0
    
    for i in range(R):
        for j in range(C):
            for nij in range(np.maximum(a[i] - N + b[j], 1), end[i,j]):
                term2 = log_Nnij[nij] - log_a[i] - log_b[j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                     - gln_N - gln_nij[nij] - math.lgamma(a[i] - nij + 1)
                     - math.lgamma(b[j] - nij + 1)
                     - math.lgamma(N - a[i] - b[j] + nij + 1))
                term3 = np.exp(gln)
                emi += (term1[nij] * term2 * term3)   
    #printf("%s\n", "AFTER FOR")
    return emi
