import math
import torch
from jutility import plotting, util, cli
from forward_grad_public.objective import Quadratic
from forward_grad_public.learner import (
    Learner,
    ForwardGradIdealSg,
    ForwardGrad,
    SyntheticGrad,
    BiasedForwardGrad,
)

def main():

    nds = plotting.NoisySweep(log_y=True)
    for name in ["FG+SG", "FG", "SG", "BFG"]:
        nds.update(name, 0, 0)

    mp = plotting.MultiPlot(
        make_subplot(0.001, 5000),
        make_subplot(0.1, 500),
        legend=plotting.FigureLegend(
            *nds.plot(),
            plotting.Line(c="k", ls=":", label="Theory"),
            loc="outside center right",
            num_rows=None,
        ),
        figsize=[8, 2.7],
        h_pad=0.03,
    )
    mp.save(
        "bfg",
        dir_name="paper/figures/img",
    )

def make_subplot(
    beta:   float,
    steps:  int,
):
    lr_min = 1e-3
    lr_max = 3e-1
    alpha_list = util.log_range(lr_min, lr_max, 20)
    repeats = 5
    D = 100

    nds = plotting.NoisySweep(log_y=True)

    table = util.Table(
        util.TimeColumn(),
        util.CountColumn(),
        util.Column("learner"),
        util.Column("alpha",   ".5f"),
        util.Column("y",    ".9f"),
    )

    for alpha in alpha_list:
        for _ in range(repeats):

            name = "FG+SG"
            learner = ForwardGradIdealSg(D, alpha, beta)
            y = train_learner(learner, steps)
            nds.update(name, alpha, y)
            table.update(learner=name, alpha=alpha, y=y)

            name = "FG"
            learner = ForwardGrad(D, alpha)
            y = train_learner(learner, steps)
            nds.update(name, alpha, y)
            table.update(learner=name, alpha=alpha, y=y)

            name = "SG"
            learner = SyntheticGrad(D, alpha, beta)
            y = train_learner(learner, steps)
            nds.update(name, alpha, y)
            table.update(learner=name, alpha=alpha, y=y)

            name = "BFG"
            learner = BiasedForwardGrad(D, alpha, beta)
            y = train_learner(learner, steps)
            nds.update(name, alpha, y)
            table.update(learner=name, alpha=alpha, y=y)

    y0 = 0.5
    lr_list_theory = util.log_range(lr_min, lr_max, 100).tolist()
    theory_lines = [
        FgSg(D, beta).plot(y0, steps, lr_list_theory),
        Fg(D, 1).plot(y0, steps, lr_list_theory),
        Sg(beta).plot(y0, steps, lr_list_theory),
    ]

    return plotting.Subplot(
        *nds.plot(),
        *theory_lines,
        xlabel="$\\alpha$",
        ylabel="$y_T$",
        log_x=True,
        log_y=True,
        ylim=[None, 1],
        title="$\\beta=%s, T=%i$" % (beta, steps),
    )

def train_learner(
    learner:    Learner,
    steps:      int,
) -> float:
    f = Quadratic()

    for _ in range(steps):
        learner.step(f)

    return learner.forward(f).item()

class TheoryPlotter:
    def convergence_rate(self, lr: float) -> float:
        raise NotImplementedError()

    def get_yt(self, y0: float, steps: int, lr: float) -> float:
        log_yt = steps*math.log(self.convergence_rate(lr)) + math.log(y0)
        yt = math.exp(min(log_yt, 100))
        return yt

    def plot(
        self,
        y0:         float,
        steps:      int,
        lr_list:    list[float],
    ) -> plotting.Line:
        return plotting.Line(
            lr_list,
            [self.get_yt(y0, steps, lr) for lr in lr_list],
            c="k",
            ls=":",
            z=100,
        )

class FgSg(TheoryPlotter):
    def __init__(self, d: int, m: float):
        self.d = d
        self.m = m

    def convergence_rate(self, lr: float) -> float:
        A_list = [
            [
                (
                    ((1 - lr*self.m)**2)
                    - 2*lr*(1 - self.m)*(1 - lr*self.m)
                    + (lr**2)*((1 - self.m)**2)*(self.d + 2)
                ),
                -2*(lr**2)*((1 - self.m)**2)*(self.d + 1),
                (lr**2)*((1 - self.m)**2)*(self.d + 1),
            ],
            [
                self.m*(1 - lr),
                (1 - lr)*(1 - self.m),
                0,
            ],
            [
                self.m**2,
                2*self.m*(1 - self.m),
                (1 - self.m)**2,
            ],
        ]

        A = torch.tensor(A_list)
        e = torch.linalg.eigvals(A)
        c = torch.abs(e).max().item()

        return c

class Fg(TheoryPlotter):
    def __init__(self, d: int, s: int):
        self.d = d
        self.s = s

    def convergence_rate(self, lr: float) -> float:
        return ((1 - lr)**2) + (lr**2)*(self.d + 1)/self.s

class Sg(TheoryPlotter):
    def __init__(self, m: float):
        self.m = m

    def convergence_rate(self, lr: float) -> float:
        A_list = [
            [
                (1 - lr*self.m)**2,
                -2*lr*(1 - self.m)*(1 - lr*self.m),
                (lr**2)*((1 - self.m)**2),
            ],
            [
                self.m*(1 - lr*self.m),
                (1 - self.m)*(1 - 2*lr*self.m),
                -lr*((1 - self.m)**2),
            ],
            [
                self.m**2,
                2*self.m*(1 - self.m),
                (1 - self.m)**2,
            ],
        ]

        A = torch.tensor(A_list)
        e = torch.linalg.eigvals(A)
        c = torch.abs(e).max().item()

        return c

if __name__ == "__main__":
    with util.Timer("main"):
        main()
