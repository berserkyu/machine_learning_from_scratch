class FeatureTransformer:
    def __init__(self, method='z-score'):
        self.means = None
        self.stds = None
        self.max_val = None
        self.min_val = None

    def fit(self, X):
        self.means = X.mean()
        self.stds = X.std()

    def transform(self, X):
        if self.means is None or self.stds is None:
            raise ValueError("The normalizer must be fitted before calling transform.")
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)