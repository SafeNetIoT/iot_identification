from dataclasses import dataclass
from scipy.stats import randint
from config import settings

@dataclass
class TuneConfig:
    """Configuration for randomized hyperparameter tuning."""
    n_iter: int = 5
    cv: int = 5
    estimator_range: tuple[int, int] = (100, 500)
    max_depth_range: tuple[int, int, int] = (5, 31, 5)
    min_samples_split_range: tuple[int, int] = (2, 20)
    min_samples_leaf_range: tuple[int, int] = (1, 10)
    max_features: tuple[str, ...] = ("sqrt", "log2", None)
    scoring: str = "accuracy"
    n_jobs: int = -1
    verbose: int = 2
    random_state: int = settings.random_state

    def param_distributions(self) -> dict:
        """Return the parameter distributions for RandomizedSearchCV."""
        return {
            "n_estimators": randint(*self.estimator_range),
            "max_depth": [None] + list(range(*self.max_depth_range)),
            "min_samples_split": randint(*self.min_samples_split_range),
            "min_samples_leaf": randint(*self.min_samples_leaf_range),
            "max_features": list(self.max_features),
        }

    def random_search_kwargs(self) -> dict:
        """Return arguments (excluding model + param_dist) for RandomizedSearchCV."""
        return {
            "n_iter": self.n_iter,
            "cv": self.cv,
            "scoring": self.scoring,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }
