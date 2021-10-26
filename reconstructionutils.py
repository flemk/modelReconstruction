'''
model reconstuction of time series by ode's utility

retireve a model of system by converting time series x into system of
ode's. n-dimensional system will be expressed as:

\dot{y}_1 = f(\vec{x};\vec{p}_1)
...
\dot{y}_n = f(\vec{x};\vec{p}_n)

Franz Ludwig Kostelezky, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
import cutility as cu
from scipy.integrate import solve_ivp

class Model:
    def __init__(self, series: list, grade: int) -> None:
        '''
        parametes:
            - <nd-array> series: each element in this array 
                represents one of system's channel time series
            - <int> grade: highest grade for the fit-polynominals
        '''
        #assert(np.shape(series) == (*, ), 'unexpected series shape, expected (*, )')

        self.series = series
        self.dimension = len(series)
        self.grade = grade
        self.fit_coefficients = []
        self.fit_functions = []

    def _retrieve_fit_coefficients(self, z: list):
        '''
        parameters:
            - <1d-array> z: target time series to be fitted to
        '''   
        polynominal_exponents = cu.polynominal(self.dimension, self.grade)
    
        len_polynominal = len(polynominal_exponents[0])
    
        a = np.ones((len_polynominal, len_polynominal))
        for i in range(len_polynominal):
            for j in range(len_polynominal):
                for k in range(self.dimension):
                    y = self.series[k]
                    a[i][j] *= np.sum(y ** (polynominal_exponents[k][j] + polynominal_exponents[k][i]))
    
        b = np.ones((len_polynominal, 1))
        for i in range(len_polynominal):
            for k in range(self.dimension):
                    y = self.series[k]
                    b[i] *= np.sum(z * y ** polynominal_exponents[k][i])

        return np.linalg.solve(a, b)

    def _convert_fit_coefficients_to_function(self, p):
        '''
        '''
        if type(p) != np.ndarray: return print('Wrong coefficient type:', type(p), 'Expected numpy.ndarray.')

        y_poly = cu.polynominal(self.dimension, self.grade)

        def func(y):
            assert(len(y) == self.dimension)
            assert(len(y) == len(y_poly))

            res = 0
            for i in range(len(p)):
                for k in range(self.dimension):
                    res += p[i] * y[k] ** y_poly[k][i]

            return res

        return func

    def _create_model(self):
        '''
        '''
        if len(self.fit_coefficients) > 0:
            print('fit coefficients already set ...replacing with new one')
            self.fit_coefficients = []
        for el in self.series:
            p = self._retrieve_fit_coefficients(cu.five_point_derivate_periodic(el))
            self.fit_coefficients.append(p)

        if len(self.fit_functions) > 0:
            print('fit coefficients already set ...replacing with new one')
            self.fit_functions = []
        for el in self.fit_coefficients:
            p_ = self._convert_fit_coefficients_to_function(el)
            self.fit_functions.append(p_)

        def func(t, x, fit_functions):
            '''
            '''
            y = np.zeros(len(fit_functions))

            for i in range(len(fit_functions)):
                el = fit_functions[i]
                y[i] = el(x)

            return y
        
        self.model = func

    def _solve_model(self, length=None, ivp=None):
        '''
        '''
        if length is None:
            T = len(self.series[0])

        if ivp is None:
            ivp = []
            for el in self.series:
                ivp.append(el[0])

        sol = solve_ivp(self.model, [0, T], ivp, dense_output=True, args=[self.fit_functions])

        self.solution = sol

    def _evaluate_solution(self, t=None):
        '''
        '''
        if t is None:
            T = len(self.series[0])
            t = np.linspace(0, T, T * 2)
        
        series_solution = self.solution.sol(t)

        return series_solution

    def evaluate(self, length=None, ivp=None, t=None):
        '''
        '''
        self._create_model()
        self._solve_model(length=length, ivp=ivp)

        return self._evaluate_solution(t=t)