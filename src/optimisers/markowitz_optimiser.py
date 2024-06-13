from optimisers.base_optimiser import AbstractOptimalHoldings
import cvxpy as cp


class OptimalHoldings(AbstractOptimalHoldings):
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
        assert len(expected_return.columns) == 1

        return cp.Minimize(-expected_return.T.values[0] @ weights)

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
        assert len(factor_betas.shape) == 2

        return [
            risk <= self.risk_cap**2,
            factor_betas.T @ weights <= self.factor_max,
            factor_betas.T @ weights >= self.factor_min,
            # sum(weights) == 1,
            weights >= self.weights_min,
            weights <= self.weights_max,
        ]

    def __init__(
        self,
        risk_cap=1,
        factor_max=10.0,
        factor_min=-10.0,
        weights_max=1,
        weights_min=0,
    ):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min
