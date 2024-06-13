from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
import pandas as pd


class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, expected_return):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        expected_return : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """

        raise NotImplementedError()

    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """

        raise NotImplementedError()

    def _get_risk(
        self,
        weights,
        factor_betas,
        expected_return_index,
        factor_cov_matrix,
        idiosyncratic_var_vector,
    ):
        f = factor_betas.loc[expected_return_index].values.T @ weights
        X = factor_cov_matrix
        S = np.diag(
            idiosyncratic_var_vector.loc[expected_return_index].values.flatten()
        )

        return cp.quad_form(f, X) + cp.quad_form(weights, S)

    def find(
        self, expected_return, factor_betas, factor_cov_matrix, idiosyncratic_var_vector
    ):
        weights = cp.Variable(len(expected_return))
        risk = self._get_risk(
            weights,
            factor_betas,
            expected_return.index,
            factor_cov_matrix,
            idiosyncratic_var_vector,
        )

        obj = self._get_obj(weights, expected_return)
        constraints = self._get_constraints(
            weights, factor_betas.loc[expected_return.index].values, risk
        )

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)

        optimal_weights = np.asarray(weights.value).flatten()
        return pd.DataFrame(data=optimal_weights, index=expected_return.index)
