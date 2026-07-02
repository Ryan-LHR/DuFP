"""
--------------------The implementation of other authors--------------------
Module Name: stable_kde
Description: Wrapper around scipys GaussianKDE
Author's paper: Simple-tip
"""

import warnings

import numpy as np
from numpy import sqrt
from scipy.stats import gaussian_kde

class StableGaussianKDE(gaussian_kde):
    """
    A version of scipy's Gaussian KDE, mitigating some numerical instability problems.

    Read the nice summary about why this is needed here: https://stackoverflow.com/a/66902455
    Note that this class is *NOT* a copy of the one given in the stackoverflow answer,
    but follows a similar approach.
    """

    MAX_INCREMENT = 1e-5

    def __init__(self, dataset, bw_method=None, weights=None):
        super(StableGaussianKDE, self).__init__(
            dataset.astype(np.float64), bw_method, weights
        )

    # docstr-coverage: inherited
    def _compute_covariance(self):
        self.factor = self.covariance_factor()

        # Cache covariance and inverse covariance of the data
        if not hasattr(self, "_data_inv_cov"):
            self._data_covariance = np.atleast_2d(
                np.cov(self.dataset, rowvar=1, bias=False, aweights=self.weights)
            )

            # !!!!!
            # The following lines make sure that the covariance matrix is (numerically)
            # positive definite.
            # !!!!!
            self._data_covariance = self._stabilize_covariance(self._data_covariance)
            if self.prepare_failed:
                return
            try:
                # self._data_inv_cov = np.linalg.inv(self._data_covariance)  # bugged

                # ~~~~~
                # modified lines for our experiment
                L = np.linalg.cholesky(self._data_covariance)
                I = np.eye(L.shape[0], dtype=self._data_covariance.dtype)
                Linv = np.linalg.solve(L, I)
                self._data_inv_cov = Linv.T @ Linv
                self._data_inv_cov = 0.5 * (self._data_inv_cov + self._data_inv_cov.T)
                # ~~~~~
            except np.linalg.LinAlgError:
                self.prepare_failed = True
                self._data_inv_cov = None
                return

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2  # for Win
        L = np.linalg.cholesky(self.covariance * 2 * np.pi)
        self.log_det = 2 * np.log(np.diag(L)).sum()
        self._norm_factor = sqrt(np.linalg.det(2 * np.pi * self.covariance))

    def _stabilize_covariance(self, covariance):
        """
        Add constant values to data covariance matrix to make it positive definite.
        """
        # !!!!!
        # The following line is the only line that differs from the
        # original scipy KDE implementation.
        # It makes sure that the covariance matrix is (numerically)
        # positive definite.
        # !!!!!
        increment = 1e-10
        while np.any(np.linalg.eigh(covariance * self.factor**2)[0] <= 0):
            # np.fill_diagonal(covariance, increment)  # bug?
            covariance[np.diag_indices_from(covariance)] += increment
            if increment > self.MAX_INCREMENT:
                warnings.warn(
                    "Was not able to fix numerical imprecision in covariance matrix."
                    "Failing silently. All likelihoods will be reported as 0."
                )
                self.prepare_failed = True
                return None
            increment += increment
        self.prepare_failed = False
        return covariance

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        if self.prepare_failed:
            return np.zeros(points.shape[1])

        # try:
        #     return super(StableGaussianKDE, self).evaluate(points)
        # except (np.linalg.LinAlgError, ValueError) as e:
        #     if "-th leading minor of the array is not positive definite" in str(
        #             e
        #     ) or "numerical imprecision in covariance matrix" in str(e):

        points = np.asarray(points, dtype=np.float64)
        return super(StableGaussianKDE, self).evaluate(points)
