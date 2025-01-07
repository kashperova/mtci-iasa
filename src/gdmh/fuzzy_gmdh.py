from itertools import combinations
from typing import Callable, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from gdmh.mia import PolynomType


class FuzzyGMDH:
    def __init__(
        self,
        polynomial_type: PolynomType,
        criterion: Callable,
        limit: Optional[float] = 0.01,
        test_size: Optional[float] = 0.2,
        k_best: Optional[int] = 5,
        seed: Optional[int] = 42,
        conf_th: Optional[float] = 0.85,
        max_iters: Optional[int] = None,
        metrics: dict[str, Callable] = None,
    ):
        """
        hyperparams:
        - limit: minimum required quality improvement to continue
        - k_best: best models count to form complex models on the next iteration
        - polynomial_type: polynomial type (only available at enum)
        - conf_th: confidence for fuzzy intervals (0 to 1)
        """

        self.test_size = test_size
        self.k_best = k_best
        self.polynomial_type = polynomial_type
        self.criterion = criterion
        self.limit = limit
        self.conf_th = conf_th
        self.max_iters = max_iters
        self.metrics = metrics or []
        self.__seed = seed
        self.__scaler = StandardScaler()

        self.history = []
        self.best_models = []
        self.metrics_cache = dict(train_r2_score=[], train_mse=[], test_r2_score=[], test_mse=[])
        self.__iters = 0

    def fit(self, x: np.array, y: np.array):
        x = self.__scaler.fit_transform(x)

        tr_size = int(len(x) * (1 - self.test_size))
        x_tr, x_test = x[:tr_size], x[tr_size:]
        y_train, y_test = y[:tr_size], y[tr_size:]

        current_tr_features = x_tr
        current_test_features = x_test
        previous_quality = float("inf")

        while True:
            if self.max_iters is not None and self.__iters >= self.max_iters:
                break

            self.__iters += 1

            # generate models for the current layer
            models = self.__get_models(current_tr_features, y_train)

            # evaluate models
            for model in models:
                x_test_transformed = self.__transform_features(current_test_features, model["features"])
                y_pred_test = x_test_transformed @ model["coefficients"]
                model["test_quality"] = self.criterion(y_test, y_pred_test)
                model["y_pred_test"] = y_pred_test

                for name, func in self.metrics.items():
                    model[f"test_{name}"] = func(y_test, y_pred_test)

            models = sorted(models, key=lambda m: m["test_quality"])
            best_models: list[dict] = models[: self.k_best]
            self.best_models.append(best_models)

            history_item = {"iter": self.__iters}
            for model in best_models:
                for k, v in model.items():
                    # count avg metrics (all metrics starts with "test" str)
                    if "test" in k:
                        history_item[f"avg_{k}"] = round(np.mean([m[k] for m in best_models]), 4)

            self.history.append(history_item)
            self.metrics_cache["test_r2_score"].append(history_item["avg_test_r2_score"])
            self.metrics_cache["test_mse"].append(history_item["avg_test_quality"])

            print(f"Iteration {self.__iters} completed: {history_item}", flush=True)

            if previous_quality - history_item[f"avg_test_quality"] < self.limit:
                print("Stopping: improvement below limit.")
                break

            previous_quality = history_item[f"avg_test_quality"]

            # generate new features for the next layer
            current_tr_features, current_test_features = self.__get_next_features()

    def __transform_features(
        self, x: np.ndarray, feature_indices: tuple[int, int]
    ) -> np.ndarray:
        x1 = x[:, feature_indices[0]]
        x2 = x[:, feature_indices[1]]

        if self.polynomial_type == PolynomType.LINEAR:
            # linear + bias
            return np.column_stack((x1, x2, np.ones_like(x1)))
        elif self.polynomial_type == PolynomType.PARTIAL_QUADRATIC:
            # partial quadratic + bias
            return np.column_stack((x1, x2, x1 * x2, x1 ** 2, np.ones_like(x1)))
        elif self.polynomial_type == PolynomType.QUADRATIC:
            # quadratic + bias
            return np.column_stack((x1, x2, x1 * x2, x1**2, x2**2, np.ones_like(x1)))

    def __get_models(
        self, x: np.ndarray, y: np.ndarray
    ) -> list[dict[str, Union[tuple[int, int], np.ndarray]]]:
        models = []
        n_features = x.shape[1]

        train_r2_scores = []
        train_mses = []

        for pair in combinations(range(n_features), 2):
            try:
                z = self.__transform_features(x, pair)
                coefficients = self.__solve_lp(z, y)
                y_pred_train = z @ coefficients
                models.append({"features": pair,
                               "coefficients": coefficients,
                               "y_pred_train": y_pred_train})

                train_r2_scores.append(
                    r2_score(y, y_pred_train)
                )
                train_mses.append(
                    mean_squared_error(y, y_pred_train)
                )

            except Exception as e:
                print(e)
                continue

        self.metrics_cache["train_r2_score"].append(sum(train_r2_scores) / len(train_r2_scores))
        self.metrics_cache["train_mse"].append(sum(train_mses) / len(train_mses))

        return models

    def __get_next_features(self) -> tuple[np.ndarray, np.ndarray]:
        train_outs, test_outs = [], []
        for model in self.best_models[-1]:
            test_outs.append(model['y_pred_test'])
            train_outs.append(model['y_pred_train'])
        return np.column_stack(train_outs), np.column_stack(test_outs)

    @staticmethod
    def __solve_lp(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        notes: y_k = alpha_k * z_k +/- c.T * z_k (5th slide, 5th formula)
        """
        n_samples, n_features = z.shape
        z_abs = np.abs(z)

        c = np.zeros(2 * n_features)
        # objective: minimize fuzzy intervals
        c[n_features:] = np.sum(z_abs, axis=0).ravel()

        # inequality constraints
        ineq_A_1 = np.column_stack((z, -1 * z_abs))
        ineq_A_2 = np.column_stack((-1 * z, -1 * z_abs))
        ineq_A = np.vstack((ineq_A_1, ineq_A_2))
        ineq_y = np.concatenate([y, -1 * y])

        # create bounds for lp
        # alphas (centers) can be any real numbers
        # c (widths) should be not negative
        bounds = [(None, None)] * n_features + [(0, None)] * n_features

        result = linprog(
            c, A_ub=ineq_A, b_ub=ineq_y,
            bounds=bounds,
            method='highs'
        )
        if not result.success:
            raise ValueError("Failed to solve lp")

        return result.x[:n_features]

    def plot_metrics(self):
        for k, v in self.metrics_cache.items():
            plt.plot(v)
            plt.title(k)
            plt.xlabel('Iters')
            plt.ylabel('Value')
            plt.grid(True)
            plt.show()

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = self.__scaler.transform(x)
        current_inputs = x

        for layer_models in self.best_models:
            next_inputs = []

            for model in layer_models:
                z = self.__transform_features(x, model["features"])
                y_pred = z @ model['coefficients']
                next_inputs.append(y_pred)

            current_inputs = np.column_stack(next_inputs)

        return current_inputs[:, 0]
