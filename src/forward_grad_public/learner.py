import torch
from forward_grad_public.objective import Objective

class Learner:
    def init_x(self, d: int):
        self.shape = [d, 1]
        self.x = torch.zeros(self.shape)
        self.x[0] = 1

    def forward(self, f: Objective) -> torch.Tensor:
        return f.forward(self.x)

    def step(self, f: Objective):
        raise NotImplementedError()

class BackProp(Learner):
    def __init__(
        self,
        d:  int,
        lr: float,
    ):
        self.init_x(d)
        self.lr = lr

    def step(self, f: Objective):
        g = f.grad(self.x)
        self.x -= self.lr * g

class ForwardGrad(Learner):
    def __init__(
        self,
        d:  int,
        lr: float,
    ):
        self.init_x(d)
        self.lr = lr

    def step(self, f: Objective):
        fg, _, _ = self.get_forward_grad(f)
        self.x -= self.lr * fg

    def get_forward_grad(
        self,
        f: Objective,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = torch.normal(0, 1, self.shape)
        g = f.grad(self.x)
        d = (g.mT @ v)
        fg = d * v
        return fg, d, v

class ForwardGradAdam(ForwardGrad):
    def __init__(
        self,
        d:  int,
        lr: float,
    ):
        self.init_x(d)
        self.opt = torch.optim.Adam([self.x], lr=lr)

    def step(self, f: Objective):
        fg, _, _ = self.get_forward_grad(f)
        self.x.grad = fg
        self.opt.step()

class ForwardGradFunctional(ForwardGrad):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  (float | None)=None,
        bias:   bool=True,
    ):
        if lr_cv is None:
            lr_cv = lr

        self.init_x(d)
        self.linear = torch.nn.Linear(d, d, bias=bias)
        with torch.no_grad():
            self.init_fsg_weights()
            if self.linear.bias is not None:
                self.linear.bias.zero_()

        self.opt = self.get_cv_optimiser(lr_cv)
        self.lr = lr

    def get_cv_optimiser(self, lr_cv: float) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.linear.parameters(), lr_cv)

    def init_fsg_weights(self):
        self.linear.weight.zero_()

    def step(self, f: Objective):
        if torch.isnan(self.x).any():
            return

        fg, _, v = self.get_forward_grad(f)
        sg = self.linear.forward(self.x.mT).mT

        self.opt.zero_grad()
        (fg - sg).square().sum().backward()
        self.opt.step()

        cv = sg - (sg.mT @ v) * v
        self.x -= self.lr * (fg + cv).detach()

class ForwardGradMinDd(ForwardGradFunctional):
    def step(self, f: Objective):
        if torch.isnan(self.x).any():
            return

        fg, d, v = self.get_forward_grad(f)
        sg = self.linear.forward(self.x.mT).mT

        loss_dd = 0.5 * (d - (sg.mT @ v)).square()

        self.opt.zero_grad()
        loss_dd.backward()
        self.opt.step()

        cv = sg - (sg.mT @ v) * v
        self.x -= self.lr * (fg + cv).detach()

class ForwardGradDiagonalFunctional(ForwardGrad):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  (float | None)=None,
    ):
        if lr_cv is None:
            lr_cv = lr

        self.init_x(d)
        self.grad_func = torch.zeros(self.shape, requires_grad=True)
        self.opt = torch.optim.SGD([self.grad_func], lr_cv)
        self.lr = lr

    def step(self, f: Objective):
        if torch.isnan(self.x).any():
            return

        fg, d, v = self.get_forward_grad(f)
        sg = self.grad_func * self.x

        loss_dd = 0.5 * (d - (sg.mT @ v)).square()

        self.opt.zero_grad()
        loss_dd.backward()
        self.opt.step()

        cv = sg - (sg.mT @ v) * v
        self.x -= self.lr * (fg + cv).detach()

class ForwardGradCvBias(ForwardGrad):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  (float | None)=None,
    ):
        if lr_cv is None:
            lr_cv = lr

        self.init_x(d)
        self.bias = torch.zeros(self.shape, requires_grad=True)
        self.opt = torch.optim.SGD([self.bias], lr_cv)
        self.lr = lr

    def step(self, f: Objective):
        if torch.isnan(self.x).any():
            return

        fg, d, v = self.get_forward_grad(f)
        sg = self.bias

        cv = sg - (sg.mT @ v) * v
        self.x -= self.lr * (fg + cv).detach()

        loss_dd = 0.5 * (d - (sg.mT @ v)).square()

        self.opt.zero_grad()
        loss_dd.backward()
        self.opt.step()

class ForwardGradCvDiagBias(ForwardGrad):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  (float | None)=None,
    ):
        if lr_cv is None:
            lr_cv = lr

        self.init_x(d)
        self.diag = torch.zeros(self.shape, requires_grad=True)
        self.bias = torch.zeros(self.shape, requires_grad=True)
        self.opt = torch.optim.SGD([self.diag, self.bias], lr_cv)
        self.lr = lr

    def step(self, f: Objective):
        if torch.isnan(self.x).any():
            return

        fg, d, v = self.get_forward_grad(f)
        sg = self.diag * self.x + self.bias

        loss_dd = 0.5 * (d - (sg.mT @ v)).square()

        self.opt.zero_grad()
        loss_dd.backward()
        self.opt.step()

        cv = sg - (sg.mT @ v) * v
        self.x -= self.lr * (fg + cv).detach()

class SyntheticGrad(Learner):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  float,
    ):
        self.init_x(d)
        self.m = torch.zeros(self.shape)
        self.lr = lr
        self.lr_cv = lr_cv

    def step(self, f: Objective):
        g = f.grad(self.x)
        self.m += self.lr_cv * (g - self.m)
        self.x -= self.lr * self.m

class ForwardGradIdealSg(ForwardGrad):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  float,
    ):
        self.init_x(d)
        self.m = torch.zeros(self.shape)
        self.lr = lr
        self.lr_cv = lr_cv

    def step(self, f: Objective):
        g = f.grad(self.x)
        self.m += self.lr_cv * (g - self.m)

        v = torch.normal(0, 1, self.shape)
        d = g.mT @ v
        d_sg = self.m.mT @ v

        self.x -= self.lr * ((d - d_sg) * v + self.m)

class BiasedForwardGrad(Learner):
    def __init__(
        self,
        d:      int,
        lr:     float,
        lr_cv:  float,
    ):
        self.init_x(d)
        self.m = torch.zeros(self.shape)
        self.lr = lr
        self.lr_cv = lr_cv

    def step(self, f: Objective):
        g = f.grad(self.x)
        self.m += self.lr_cv * (g - self.m)
        d = (g.mT @ self.m)
        fg = (d / self.m.square().sum()) * self.m
        self.x -= self.lr * fg
