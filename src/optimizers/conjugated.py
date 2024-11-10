from copy import deepcopy
from typing import Optional

from torch import Tensor

from models.base import BaseModel
from models.losses.base import BaseLoss
from optimizers.base import BaseOptimizer


class ConjugatedOptimizer(BaseOptimizer):
    """
    ================== notes ===================
    y_hat = X * W + b
    loss = (1 / 2) * (X * W - Y) * (X * W - Y).T =
    = (1 / 2) * (W.T * X.T * X * W  -  2 * W.T * X.T * Y  +  Y.T * Y)
    grad(loss) = X.T * X * W - X.T * Y

    H(loss) = grad(X.T * (X * W - Y)) = X.T * X

    A * x = b  -> X.T * X * weights = X.T * Y
    A -> X.T * X
    b -> X.T * Y
    ================== notes ===================
    """

    def __init__(
        self,
        model: BaseModel,
        loss_fn: BaseLoss,
        tol: Optional[float] = 1e-6,
        ls_iters: Optional[int] = 10,
        ls_tol: Optional[float] = 1e-4,
        grad_clip: Optional[float] = 1e-2,
        patience: Optional[int] = 2,
        factor: Optional[float] = 0.1,
        init_alpha: Optional[float] = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.loss_fn = loss_fn
        self.tol = tol
        self.ls_iters = ls_iters
        self.ls_tol = ls_tol
        self.grad_clip = grad_clip
        self.patience = patience
        self.factor = factor
        self.runs = 0
        self.prev_grad = None
        self.direction = None
        self.prev_alpha = init_alpha
        self.num_bad_epochs = 0

    def line_search(self, x: Tensor, y: Tensor, init_loss: float):
        alpha = 0.001
        old_params = deepcopy(self.model.parameters())

        # line search loop
        for _ in range(self.ls_iters):
            self.update_params(alpha)
            new_loss =  self.loss_fn.loss(y=y, y_hat=self.model(x))

            # if the new loss is smaller, we accept this step size
            grads = deepcopy(self.model.gradients())
            grad_dir_product = sum((d * g).sum() for d, g in zip(self.direction, grads))
            if new_loss < init_loss - self.ls_tol * alpha * grad_dir_product:
                return alpha

            # reduce alpha if loss didn't decrease
            alpha *= 0.5
            self.model.set_params(old_params)

        return alpha

    def update_params(self, alpha: float):
        for param, d in zip(self.model.parameters(), self.direction):
            param += alpha * d

    def update_alpha(self) -> float:
        alpha = self.prev_alpha

        if self.num_bad_epochs >= self.patience:
            alpha *= self.factor
            self.num_bad_epochs = 0

        return alpha

    def run(self, x: Tensor, y: Tensor) -> float:
        # forward pass
        outputs = self.model.forward(x)
        loss = self.loss_fn.loss(y=y, y_hat=outputs)

        if self.runs == 0:
            # initially set direction to anti-gradient
            loss_grad = self.loss_fn.backward(y=y, y_hat=outputs)
            self.model.zero_grad()
            self.model.backward(loss_grad, lr=0, momentum=0)
            self.prev_grad = deepcopy(self.model.gradients())
            self.direction = [-g.clone() for g in self.prev_grad]
        else:
            # update parameters in the direction of descent
            # alpha = self.line_search(x, y, loss)
            # alpha = self.update_alpha()
            alpha = 0.001
            self.update_params(alpha)

            # recompute loss and grad
            loss = self.loss_fn.loss(y=y, y_hat=self.model.forward(x))
            loss_grad = self.loss_fn.backward(y=y, y_hat=outputs)
            self.model.zero_grad()
            self.model.backward(loss_grad, lr=0, momentum=0)
            new_grads = [grad.clamp_(-self.grad_clip, self.grad_clip) for grad in deepcopy(self.model.gradients())]

            if loss < self.tol:
                return loss

            # Polak-Ribiere update
            numerator = sum((ng * (ng - g)).sum() for ng, g in zip(new_grads, self.prev_grad))
            denominator = sum((g * g).sum() for g in self.prev_grad)
            beta = numerator / (denominator + 1e-8)
            beta = max(beta, 0)

            self.direction = [-ng + beta * d for ng, d in zip(new_grads, self.direction)]

            # update vars for the next step
            self.prev_grad = new_grads
            self.prev_alpha = alpha

        self.runs += 1

        return loss

