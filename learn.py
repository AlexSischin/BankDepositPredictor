from dataclasses import dataclass
from numbers import Number

import numpy as np


def isfinite(*args):
    for a in args:
        if not np.isfinite(a).all():
            return False
    return True


def zero_params(feature_count: int):
    return np.zeros(feature_count), 0.


def sigmoid(z: np.float64):
    return 1 / (1 + np.exp(-z))


def arr_to_float64(a: np.ndarray) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a.astype(np.float64)
    else:
        raise ValueError(f'Expected numpy array. Got: {type(a)}')


def num_to_float64(n: Number) -> np.float64:
    if isinstance(n, Number):
        return np.float64(n)
    else:
        raise ValueError(f'Expected numpy array. Got: {type(n)}')


@dataclass(frozen=True)
class LearningHistPoint:
    cost: float
    dj_dw: np.ndarray
    dj_db: float
    w: np.ndarray
    b: float


class LogisticRegressor:
    def __init__(self,
                 w: np.ndarray | None = None,
                 b: Number | None = None
                 ):
        self._w = arr_to_float64(w) if w is not None else None
        self._b = num_to_float64(b) if b is not None else None
        self._x_mean = None
        self._x_sd = None

    # Parameters:
    #   x - 2D array with shape (m, n), where m - number of training examples, n - number of features
    #   y - 1D array of targets with shape of m
    #   it - number of iterations over the whole training set
    #   a - learning rate alpha
    #   l_ - regularization parameter lambda
    # Returns:
    #   list[LearningHistPoint] - the history of learning, if debug set to True.
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            it: int,
            a: Number,
            l_: Number
            ) -> list[LearningHistPoint]:
        x = arr_to_float64(x)
        y = arr_to_float64(y)
        a = num_to_float64(a)
        l_ = num_to_float64(l_)
        if not isfinite(x, y, it, a, l_):
            raise ValueError(f'Input params must be finite numbers')
        if self._w is None or self._b is None:
            self._init_params(x)

        self._init_scales(x)
        x = self._scale(x)

        hist = []
        for i in range(it):
            dj_dw, dj_db = self._gradient(x, y, l_)
            self._w -= a * dj_dw
            self._b -= a * dj_db
            hist.append(LearningHistPoint(
                cost=self._cost(x, y, l_),
                dj_dw=dj_dw,
                dj_db=dj_db,
                w=np.copy(self._w),
                b=self._b
            ))
        return hist

    # Parameters:
    #   x - 2D array with shape (m, n), where m - number of training examples, n - number of features
    # Returns:
    #   np.ndarray - 1D array of estimates
    def predict(self, x: np.ndarray) -> np.ndarray:
        x = self._scale(arr_to_float64(x))
        return self._predict(x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self._w + self._b)

    # Parameters:
    #   x - 2D array with shape (m, n), where m - number of training examples, n - number of features
    #   y - 1D array of targets with shape of m
    #   l_ - regularization parameter
    # Returns:
    #   float - cost of model for x.
    def cost(self,
             x: np.ndarray,
             y: np.ndarray,
             l_: Number
             ) -> np.float64:
        x = self._scale(arr_to_float64(x))
        y = arr_to_float64(y)
        l_ = num_to_float64(l_)
        return self._cost(x, y, l_)

    def _cost(self,
              x: np.ndarray,
              y: np.ndarray,
              l_: np.float64
              ) -> np.float64:
        mean_loss = np.mean(self._loss(x, y))
        w_regularizer = l_ / (2 * x.shape[0]) * np.sum(self._w ** 2)
        return mean_loss + w_regularizer

    def _loss(self,
              x: np.ndarray,
              y: np.ndarray
              ) -> np.ndarray:
        y_hat = self._predict(x)
        return -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    def _gradient(self,
                  x: np.ndarray,
                  y: np.ndarray,
                  l_: np.float64
                  ) -> tuple[np.ndarray, np.float64]:
        error = self._predict(x) - y
        dj_dw = (error @ x + l_ * self._w) / x.shape[0]
        dj_db = np.float64(np.mean(error))
        return dj_dw, dj_db

    def _init_params(self, x: np.ndarray):
        self._w, self._b = zero_params(x.shape[1])

    def _init_scales(self, x: np.ndarray) -> None:
        self._x_mean = np.mean(x, axis=0)
        self._x_sd = np.std(x, axis=0)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self._x_mean) / self._x_sd
