'''
math helper functions

Franz Ludwig Kostelezky, 2021
'''

import numpy as np
from itertools import product

def finite_difference_derivate_3_point(series, _):
    derivate = - np.roll(series, 1) + np.roll(series, -1)
    derivate = derivate / 2
    
    return derivate

def finite_difference_derivate_5_point(series, _):
    derivate = - np.roll(series, -2) + 8 * np.roll(series, -1) - 8 * np.roll(series, 1) + np.roll(series, 2)
    derivate = derivate / 12
    
    return derivate

def finite_difference_derivate_7_point(series, _):
    derivate = - np.roll(series, 3) + 9 * np.roll(series, 2) - 45 * np.roll(series, 1) + 45 * np.roll(series, -1) \
               - 9 * np.roll(series, -2) + np.roll(series, -3)
    derivate = derivate / 60
    
    return derivate

def finite_difference_derivate_9_point(series, _):
    derivate = - 3 * np.roll(series, 4) - 32 * np.roll(series, 3) + 168 * np.roll(series, 2) - 672 * np.roll(series, 1) \
               + 672 * np.roll(series, -1) - 168 * np.roll(series, -2) + 32 * np.roll(series, -3) - \
               3 * np.roll(series, -4)
    derivate = derivate / 840
    
    return derivate

def second_order_upwind(series, dx=1):
    '''Returns the 1D second order upwind derivate of a one dimensional
    time series using reflecting boundary conditions.
    '''

    series = np.array(series)
    #dx = 1
    d_pos = (- 3 * series \
             + 4 * np.roll(series, shift=-1, axis=0) \
             - np.roll(series, shift=-2, axis=0)
            ) / (2 * dx)
    d_neg = (+ 3 * series \
             - 4 * np.roll(series, shift=1, axis=0) \
             + np.roll(series, shift=2, axis=0)
            ) / (2 * dx)
    derivate = d_pos
    derivate[-3::] = d_neg[-3::]

    return derivate

def first_order_upwind(series, dx=1):
    ''' first order upwind first order 1d derivate
    '''
    series = np.array(series)
    #dx = 1
    
    d_pos = (series - np.roll(series, shift=1, axis=0)) / dx
    d_neg = (np.roll(series, shift=-1, axis=0) - series) / dx

    derivate = d_pos
    derivate[-2::] = d_neg[-2::]
    
    return derivate

def third_order_upwind(series, dx=1):
    ''' third order upwind third order 1d derivate
    '''
    series = np.array(series)
    #dx = 1
    
    d_pos = (- 2 * np.roll(series, shift=1, axis=0) \
             - 3 * series \
             + 6 * np.roll(series, shift=-1, axis=0) \
             - np.roll(series, shift=-2, axis=0)
            ) / (6 * dx)
    d_neg = (+ 2 * np.roll(series, shift=-1, axis=0) \
             + 3 * series \
             - 6 * np.roll(series, shift=1, axis=0) \
             + np.roll(series, shift=2, axis=0)
            ) / (6 * dx)
    
    derivate = d_pos
    derivate[-4::] = d_neg[-4::]

    return derivate

def polynominal(dimension, grade):
    ''' returns the exponents of a polynominal
        of a given dimension to a given grade.
    '''
    # terminal condition
    if grade == 1:
        return np.identity(dimension)
        
    # get all possible combinations of grade x dimension
    tmp = product(range(grade + 1), repeat=dimension)
    tmp = list(tmp)
    
    # remove all which do not match grade
    tmp_ = []
    for i in range(len(tmp)):
        if np.sum(tmp[i]) == grade:
            tmp_.append(list(tmp[i]))
    
    # convert to full numpy array
    tmp_ = np.asarray([np.asarray(el) for el in tmp_])
    
    return np.append(polynominal(dimension, grade - 1), tmp_.T, axis=1)

#def _polynominal(dimension, grade):
#    ''' This function is obsolete. it was used to verify polynominal.
#    '''
#    if dimension != 3:
#        return polynominal(dimension, grade)
#    
#    assert(dimension == 3)
#    assert(grade <= 4)
#    print('using 3d static - warning: obsolete')
#
#    if grade == 1:
#        r = [[1., 0., 0.],
#             [0., 1., 0.],
#             [0., 0., 1.]]
#    if grade == 2:  #   v
#        r = [[1., 0., 0., 2., 1., 1., 0., 0., 0.],
#             [0., 1., 0., 0., 1., 0., 2., 0., 1.],
#             [0., 0., 1., 0., 0., 1., 0., 2., 1.]]
#    if grade == 3:  #                           v
#        r = [[1., 0., 0., 2., 1., 1., 0., 0., 0., 3., 2., 2., 1., 1., 1., 0., 0., 0., 0.],
#             [0., 1., 0., 0., 1., 0., 2., 0., 1., 0., 1., 0., 2., 0., 1., 3., 0., 1., 2.],
#             [0., 0., 1., 0., 0., 1., 0., 2., 1., 0., 0., 1., 0., 2., 1., 0., 3., 2., 1.]]
#    if grade == 4:  #                                                                   v
#        r = [[1., 0., 0., 2., 1., 1., 0., 0., 0., 3., 2., 2., 1., 1., 1., 0., 0., 0., 0., 4., 3., 3., 2., 2., 2., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
#             [0., 1., 0., 0., 1., 0., 2., 0., 1., 0., 1., 0., 2., 0., 1., 3., 0., 1., 2., 0., 1., 0., 2., 0., 1., 3., 0., 2., 1., 3., 1., 2., 4., 0.],
#             [0., 0., 1., 0., 0., 1., 0., 2., 1., 0., 0., 1., 0., 2., 1., 0., 3., 2., 1., 0., 0., 1., 0., 2., 1., 0., 3., 1., 2., 1., 3., 2., 0., 4.]]
#    
#    return np.asarray(r)