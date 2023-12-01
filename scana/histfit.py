
import numpy
import scipy.optimize

__all__ = ['HistogramFit']


class HistogramFit:

    def __init__(self, *, order=5):
        self._order = max(1, int(order))

    @property
    def order(self):
        return self._order

    def exp_func(self, x, mu, sigma, rho):
        return rho*numpy.exp(-numpy.power((x-mu)*sigma, 2.0))

    def exp_func_gauss(self, x, mu, sigma, rho):
        return rho/numpy.sqrt(2.0*numpy.pi*sigma*sigma)*numpy.exp(-0.5*numpy.power((x-mu)/sigma, 2.0))

    def exp_func_sum(self, x, *params):
        assert (len(params) == 3*self._order)
        func_values = numpy.zeros(x.size, dtype=float)
        for i in range(self._order):
            mu, sigma, rho = params[3*i:3*(i+1)]
            func_values += self.exp_func(x, mu, sigma, rho)
        return func_values

    def fit(self, centers, values, *, loss_func='cauchy', tol=1.0e-6, num_fev=5000, verbose_level=0):
        values = numpy.array(values, dtype=float)

        range_min, range_max = numpy.min(centers), numpy.max(centers)
        delta_mu = (range_max - range_min) / self._order
        initial_mu_values = numpy.linspace(range_min, range_max, num=self._order, endpoint=False) + delta_mu * 0.5
        initial_sigma_values = [0.1] * self._order
        initial_rho_values = [numpy.mean(values)] * self._order
        initial_values, param_bounds = list(), list()

        for i, (mu, sigma, rho) in enumerate(zip(initial_mu_values, initial_sigma_values, initial_rho_values)):
            initial_values += [mu, sigma, rho]
            param_bounds += [(range_min, range_max), (0.0, numpy.inf), (0.0, numpy.max(values))]

        initial_values = numpy.array(initial_values, dtype=float)
        param_bounds = numpy.array(param_bounds, dtype=float).T
        try:
            params, _ = scipy.optimize.curve_fit(self.exp_func_sum, centers, values, p0=initial_values,
                                                 check_finite=True, method='trf', bounds=param_bounds,
                                                 jac='3-point', max_nfev=num_fev, loss=loss_func,
                                                 ftol=tol, xtol=tol, gtol=tol, verbose=verbose_level)
        except ValueError as ex:
            print('ValueError: {}'.format(ex))
            return None
        except RuntimeError as ex:
            print('RuntimeError: {}'.format(ex))
            return None
        except scipy.optimize.OptimizeWarning as ex:
            print('OptimizeWarning: {}'.format(ex))
            return None
        else:
            return params
