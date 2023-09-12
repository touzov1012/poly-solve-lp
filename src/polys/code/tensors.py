import itertools
import numpy as np

def get_monomial_indices(n, k):
    '''
    get a dictionary mapping vectors of monomial coefficients in
    n variables up to degree k to their corresponding ordering index.
    Each monomial is represented as a tuple (., ., ...)
    
    n:   the number of variables
    k:   the maximum degree to consider
    '''
    
    # get all combinations of vectors of length n with values 0..k
    vectors = list(itertools.product(range(k+1), repeat=n))
    
    # filter out vectors whose sum is more than k
    vectors = [vector for vector in vectors if sum(vector) <= k]
    
    # Sort the vectors in lexicographical order
    vectors.sort()
    
    # create a dictionary mapping vectors to their lexicographic index
    vector_dict = {vector: index for index, vector in enumerate(vectors)}
    
    return vector_dict

def get_localized(full_mono_lookup, pairwise_sums):
    '''
    generate the localizing moment operator which acts on a sequence of monomials
    
    full_mono_lookup:   this is dictionary of monomials mapped to order,
                        should have at least num_slices monomials
    pairwise_sums:      the base matrix of monomial products with a possible local term
    '''

    array_2d = np.apply_along_axis(lambda x: full_mono_lookup.get(tuple(x), -1), -1, pairwise_sums)

    # number of monomials
    num_slices = len(full_mono_lookup)
    
    # get the spare array of tensor slices
    array_3d = np.zeros((num_slices + 1,) + array_2d.shape)

    # create index arrays for the first and second dimensions
    range_of_half_k = np.arange(array_2d.shape[0])
    x_idx, y_idx = np.meshgrid(range_of_half_k, range_of_half_k)

    # set the positions specified by the 2D array to 1
    array_3d[array_2d, x_idx, y_idx] = 1
    
    return array_3d[:-1]

def get_moment_operator(n, k_half, k_max=None, monomials=None, localizer=None):
    '''
    create the moment operator with optional localizer polynomial provided
    as a list of (coef, (powers))
    
    n:           the number of variables
    k_half:      the dictator of the base polynomial domain to consider, without
                 localizing, the full supported degree would be 2 * k_half
    k_max:       the upper bound on the degrees we work with in the problem, should be
                 large enough to support the localizer matrix upper degree
    monomials:   the dictionary of monomials to consider, by default this will be
                 generated using k_max as the upper bound on degree. Either k_max
                 or monomials should be provided
    localizer:   the localizing polynomials for this operator. Should be supplied as
                 a list of monomials which are given by (coef, (degrees))
    '''
    
    # add the base localizer if needed
    if localizer is None:
        localizer = [(1, [0] * n)]
    
    # get the monomial lookup
    if monomials is None:
        monomials = get_monomial_indices(n, k_max)
    
    # get the half degree monomials
    half_k_monomials = [key for key in monomials.keys() if sum(key) <= k_half]
    
    # get the half space monomials as a numpy array
    half_monos = np.array(half_k_monomials)
    
    array1 = half_monos[:, np.newaxis, :]
    array2 = half_monos[np.newaxis, :, :]

    # broadcast to find the pairwise sums
    pairwise_sums = array1 + array2
    
    # initialize with none
    num_slices = len(monomials)
    A = np.zeros((num_slices,) + pairwise_sums.shape[:2])
    
    # iterate over each localizer
    for coef, local in localizer:
        
        # localize the monomials
        localized = pairwise_sums + np.array(local)
        
        # generate the monomial localizer
        A += coef * get_localized(monomials, localized)
    
    # return the localized operator
    return A