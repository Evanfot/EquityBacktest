from optimisers.markowitz_optimiser import OptimalHoldings


import cvxpy as cp


class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert len(alpha_vector.columns) == 1

        target = (alpha_vector - alpha_vector.mean()) / alpha_vector.abs().sum()
        target = target.values.flatten()
        obj = cp.Minimize(cp.norm(weights - target, p=2, axis=None))

        return obj
