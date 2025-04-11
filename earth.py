import numpy as np

# ---------------------------
class EarthRegressor:
    def __init__(self, max_terms=10, min_samples_leaf=10, max_knots=None):
        """
        max_terms: maximum number of basis functions (including intercept)
        min_samples_leaf: minimum number of samples required for a candidate hinge to be considered
        max_knots: maximum number of candidate knots per feature. If None, use all unique values.
        """
        self.max_terms = max_terms
        self.min_samples_leaf = min_samples_leaf
        self.max_knots = max_knots
        self.basis_terms = []  # list of (feature_index, knot, direction) tuples

    def _hinge(self, x, knot, direction):
        """Compute the hinge function."""
        if direction == 'right':
            return np.maximum(0, x - knot)
        elif direction == 'left':
            return np.maximum(0, knot - x)
        else:
            raise ValueError("direction must be 'right' or 'left'")

    def fit(self, X, y):
        n, d = X.shape
        # Start with intercept only
        X_design = np.ones((n, 1))
        self.basis_terms = []  # reset basis terms

        # Forward pass: add basis functions until reaching max_terms - intercept.
        for term in range(self.max_terms - 1):
            best_rss = np.inf
            best_candidate = None
            best_new_feature = None

            # For each predictor variable
            for j in range(d):
                xj = X[:, j]
                # Use unique values as candidate knots
                candidate_knots = np.unique(xj)
                # If max_knots is specified and there are too many candidates, select evenly spaced quantiles.
                if self.max_knots is not None and len(candidate_knots) > self.max_knots:
                    percentiles = np.linspace(0, 100, self.max_knots)
                    candidate_knots = np.percentile(xj, percentiles)
                    
                for knot in candidate_knots:
                    for direction in ['right', 'left']:
                        candidate_feature = self._hinge(xj, knot, direction)
                        # Skip if too few samples are non-zero in the candidate feature
                        if np.sum(candidate_feature > 0) < self.min_samples_leaf:
                            continue
                        # Create candidate design matrix by adding this new feature
                        candidate_design = np.column_stack([X_design, candidate_feature])
                        # Solve for linear regression coefficients (least squares)
                        coef, residuals, rank, s = np.linalg.lstsq(candidate_design, y, rcond=None)
                        y_pred_candidate = candidate_design.dot(coef)
                        rss = np.sum((y - y_pred_candidate)**2)
                        if rss < best_rss:
                            best_rss = rss
                            best_candidate = (j, knot, direction)
                            best_new_feature = candidate_feature

            # If no candidate improved the model, break out
            if best_candidate is None:
                break
            # Add the best candidate feature to the design matrix and record its details
            X_design = np.column_stack([X_design, best_new_feature])
            self.basis_terms.append(best_candidate)

        # Fit the final linear model on the selected basis functions
        self.coef_, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
        self.X_design_ = X_design  # store design matrix for reference

    def predict(self, X_new):
        n_new = X_new.shape[0]
        # Start with the intercept column
        X_design_new = np.ones((n_new, 1))
        # For each selected basis term, generate the corresponding feature for new data
        for (j, knot, direction) in self.basis_terms:
            new_feature = self._hinge(X_new[:, j], knot, direction)
            X_design_new = np.column_stack([X_design_new, new_feature])
        return X_design_new.dot(self.coef_)