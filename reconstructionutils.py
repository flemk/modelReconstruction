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
from scipy.integrate import odeint, solve_ivp

class Model:
    def __init__(self, series: list, grade: int, derivate: list=None, weighting:list=None, dx:float=1) -> None:
        '''
        parametes:
            - <nd-array> series: each element in this array 
                represents one of system's channel time series
            - <int> grade: highest grade for the fit-polynominals
            - <nd-array> derivate: each element in this is a derivate for the
                corresponding element in series
            - <nd-array> weighting: weighting for each element used in ls fit
        '''
        #assert(np.shape(series) == (*, ), 'unexpected series shape, expected (*, )')

        for el in series:
            assert(len(el) == len(series[0]))
        self.series = series
        
        if derivate is not None:
            for el in derivate:
                assert(len(el) == len(derivate[0]))
            self.derivate = derivate
        else:
            self.derivate = []
            for el in self.series:
                self.derivate.append(
                    cu.first_order_upwind(el, dx)) # TODO make derivate option selectable

        if weighting is not None:  # setting weighting function
            assert(len(weighting) == len(series[0]))
            self.weighting = np.asarray(weighting)
        else:
            # if no weighting function is defined, the weight will be 1:
            # the series stay the same, multiplied by 1.
            self.weighting = np.ones(np.shape(self.series[0]))
        
        self.dimension = len(series)
        self.grade = grade
        self.dx = dx
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
                tmp = np.ones(np.shape(self.series[0]))
                for k in range(self.dimension):
                    y = self.series[k]
                    tmp *= y ** (polynominal_exponents[k][j] + polynominal_exponents[k][i])
                tmp *= self.weighting  # DEBUG: weighting here
                a[i][j] *= np.sum(tmp)

        b = np.ones((len_polynominal, 1))
        for i in range(len_polynominal):
            tmp = np.ones(np.shape(self.series[0]))
            for k in range(self.dimension):
                y = self.series[k]
                tmp *= y ** polynominal_exponents[k][i]
            tmp *= self.weighting  # DEBUG: weighting here
            b[i] *= np.sum(z * tmp)
            
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
                tmp = 1
                for k in range(self.dimension):
                    tmp *= y[k] ** y_poly[k][i]
                res += p[i] * tmp

            return res

        return func

    def _create_model(self):
        '''
        '''
        if len(self.fit_coefficients) > 0:
            print('fit coefficients already set ...replacing with new one')
            self.fit_coefficients = []
        for el in self.derivate:
            p = self._retrieve_fit_coefficients(el) #TODO 
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
        else:
            T = length

        if ivp is None:
            print('no initial values defined.')
            ivp = []
            for el in self.series:
                ivp.append(el[0])

        #sol = solve_ivp(self.model, [0, T], ivp, dense_output=True, args=[self.fit_functions])
        f = 1024  # TODO make factor f selectable as param
        t = np.linspace(0, T, T * f)
        sol, infodict = odeint(self.model, ivp, t, args=(self.fit_functions,), tfirst=True, full_output=True, printmessg=True)
        
        self.solution = sol

    def _evaluate_solution(self, t=None):
        '''
        '''
        # since using odeint, self.solution has no attribute sol
        # TODO make evaluate solution variable, i.e. possibility to change the time interval
        # /w t in _solve_model
        
        #if t is None:
        #    T = len(self.series[0])
        #    t = np.linspace(0, T, T * 2)
        #
        #series_solution = self.solution.sol(t)

        #return series_solution

        return self.solution

    def evaluate(self, length=None, ivp=None, t=None):
        '''
        '''
        self._create_model()
        self._solve_model(length=length, ivp=ivp)

        return self._evaluate_solution(t=t)