from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor
from itertools import combinations


class PolynomType(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    PARTIAL_QUADRATIC = "partial_quadratic"


class Mia:
    def __init__(
        self,
        polynomial_type: PolynomType,
        criterion: Callable,
        test_size: Optional[float] = 0.2,
        p_average: Optional[int] = 3,
        limit: Optional[float] = 0.01,
        k_best: Optional[int] = 5,
        seed: Optional[int] = 42,
        max_iters: Optional[int] = None,
    ):
        """
        hyperparams:
        - p_average: best models count for calculating the average quality
        - limit: minimum required quality improvement to continue
        - k_best: best models count to form complex models on the next iteration
        - polynomial_type: polynomial type (only available at enum)
        """

        self.polynomial_type = polynomial_type
        self.criterion = criterion
        self.test_size = test_size
        self.p_average = p_average
        self.limit = limit
        self.k_best = k_best
        self.max_iters = max_iters
        self.__seed = seed

        self.history = []
        self.best_models = []

    def fit(self, x: Tensor, y: Tensor):
        x_tr, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.__seed
        )
        current_features = x_tr
        iters = 0
        previous_quality = float("inf")

        while True:
            if self.max_iters is not None and iters >= self.max_iters:
                break

            iters += 1

            # generate models for the current layer
            models = self.__get_models(current_features, y_train)

            # evaluate models
            for model in models:
                x_test_transformed = self.__transform_features(x_test, model["features"])
                y_pred_test = x_test_transformed @ model["coefficients"]
                model["test_quality"] = self.criterion(y_pred_test, y_test)

            models = sorted(models, key=lambda m: m["test_quality"])
            best_models = models[: self.k_best]
            self.best_models.extend(best_models)

            # compute the average quality of the top p_average models
            top_p_models: list[dict] = models[: self.p_average]
            average_criterion = np.mean([m["test_quality"] for m in top_p_models])

            self.history.append({"iter": iters, "average_criterion": average_criterion})
            print(
                f"Iteration {iters} completed: average criterion {average_criterion}",
                flush=True,
            )

            if previous_quality - average_criterion < self.limit:
                print("Stopping: improvement below limit.")
                break

            previous_quality = average_criterion

            # generate new features for the next layer
            current_features = self.__get_next_features(x_tr, best_models)

    def __get_models(
        self, x: Tensor, y: Tensor
    ) -> list[dict[str, Union[tuple[int, int], np.ndarray]]]:
        models = []
        n_features = x.shape[1]

        for pair in combinations(range(n_features), 2):
            x_transformed = self.__transform_features(x, pair)
            # solve least squares to get coefficients
            coefficients = (
                np.linalg.pinv(x_transformed.T @ x_transformed) @ x_transformed.T @ y
            )

            models.append({"features": pair, "coefficients": coefficients})

        return models

    def __transform_features(
        self, x: Tensor, feature_indices: tuple[int, int]
    ) -> np.ndarray:
        x1 = x[:, feature_indices[0]]
        x2 = x[:, feature_indices[1]]

        if self.polynomial_type == PolynomType.LINEAR:
            # linear + bias
            return np.column_stack((x1, x2, np.ones_like(x1)))
        elif self.polynomial_type == PolynomType.QUADRATIC:
            # quadratic + bias
            return np.column_stack((x1, x2, x1 * x2, x1**2, x2**2, np.ones_like(x1)))

    def __get_next_features(self, x: Tensor, models: list[dict]) -> np.ndarray:
        outputs = []
        for model in models:
            x_transformed = self.__transform_features(x, model["features"])
            y_pred = x_transformed @ model["coefficients"]
            outputs.append(y_pred)
        return np.column_stack(outputs)

    def predict(self, x: Tensor) -> Tensor:
        # average predictions from the top p_average models
        top_models = sorted(self.best_models, key=lambda m: m["test_quality"])[
            : self.p_average
        ]
        predictions = np.mean(
            [
                self.__transform_features(x, model["features"]) @ model["coefficients"]
                for model in top_models
            ],
            axis=0,
        )
        return predictions
