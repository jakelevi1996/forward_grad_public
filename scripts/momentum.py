import math
import torch
from jutility import plotting, util, cli
from forward_grad_public.objective import Quadratic
from forward_grad_public.learner import (
    Learner,
    BackProp,
    ForwardGrad,
    ForwardGradMomentum,
    ForwardGradAdam,
)

def main():
    D = 100
    S = 1

    alpha_list = util.log_range(0.001, 0.02, 100)
    beta_list  = util.log_range(1e-5, 1, 100)

    alpha_best = 1 / (D + 2)
    rho_best = (D + 1) / (D + 2)

    ns_m = plotting.NoisySweep()

    for alpha in alpha_list:
        for beta in beta_list:
            rho = momentum_convergence_rate(alpha, beta, D)
            ns_m.update(beta, alpha, rho)

    rho_best_sweep = min(ns_m)
    print(rho_best, alpha_best)
    print(rho_best_sweep, ns_m.inverse(rho_best_sweep))
    util.hline()

    ns_a = plotting.NoisySweep(log_y=True)
    table = util.Table(
        util.TimeColumn(),
        util.CountColumn(),
        util.Column("alpha", ".5f"),
        util.Column("repeat"),
        util.Column("y_bp", ".5f"),
        util.Column("y_fg", ".5f"),
        util.Column("y_fgb1", ".5f"),
        util.Column("y_fgb2", ".5f"),
        util.Column("y_adam", ".5f"),
    )

    torch.manual_seed(0)
    lr_min = 5e-4
    lr_max = 2e-2
    lr_list_adam = util.log_range(lr_min, lr_max, 20)
    m1 = 0.02
    m2 = 0.005
    repeats = 5
    steps = 5000

    for alpha in lr_list_adam.tolist():
        for i in range(repeats):
            learner = BackProp(D, alpha)
            y_bp = train_learner(learner, steps)
            ns_a.update("BP", alpha, y_bp)

            learner = ForwardGrad(D, alpha)
            y_fg = train_learner(learner, steps)
            ns_a.update("FG", alpha, y_fg)

            learner = ForwardGradMomentum(D, alpha, m1)
            y_fgb1 = train_learner(learner, steps)
            ns_a.update("FG, $\\beta=%s$" % m1, alpha, y_fgb1)

            learner = ForwardGradMomentum(D, alpha, m2)
            y_fgb2 = train_learner(learner, steps)
            ns_a.update("FG, $\\beta=%s$" % m2, alpha, y_fgb2)

            learner = ForwardGradAdam(D, alpha)
            y_adam = train_learner(learner, steps)
            ns_a.update("FG+Adam", alpha, y_adam)

            table.update(
                alpha=alpha,
                repeat=i,
                y_bp=y_bp,
                y_fg=y_fg,
                y_fgb1=y_fgb1,
                y_fgb2=y_fgb2,
                y_adam=y_adam,
            )

    cp = plotting.ColourPicker.contrast()
    adam_lines = ns_a.plot(cp=cp)

    y0 = 0.5
    lr_list_theory = util.log_range(lr_min, lr_max, 100).tolist()
    theory_lines = [
        Bp().plot(y0, steps, lr_list_theory),
        Fg(D, S).plot(y0, steps, lr_list_theory),
        FgMomentum(D, S, m1).plot(y0, steps, lr_list_theory),
        FgMomentum(D, S, m2).plot(y0, steps, lr_list_theory),
    ]
    theory_legend_line = plotting.Line(c="k", ls=":", label="Theory")

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *ns_m.plot(
                cp=plotting.ColourPicker.cool(len(beta_list)),
                alpha_fill=0,
                alpha_scat=0,
            ),
            plotting.VLine(alpha_best, c="k", ls="--", z=50),
            plotting.HLine(rho_best, c="k", ls="--", z=50),
            xlabel="$\\alpha$",
            ylabel="$\\rho$",
            log_x=True,
        ),
        plotting.ColourBar(
            beta_list.min(),
            beta_list.max(),
            log=True,
            label="$\\beta$",
            cmap="cool",
        ),
        plotting.Subplot(
            *ns_m.transpose().plot(
                cp=plotting.ColourPicker.cool(len(alpha_list)),
                alpha_fill=0,
                alpha_scat=0,
            ),
            plotting.HLine(rho_best, c="k", ls="--", z=50),
            xlabel="$\\beta$",
            ylabel="$\\rho$",
            log_x=True,
        ),
        plotting.ColourBar(
            alpha_list.min(),
            alpha_list.max(),
            log=True,
            label="$\\alpha$",
            cmap="cool",
        ),
        plotting.Subplot(
            *adam_lines,
            *theory_lines,
            plotting.Legend.from_plottables(
                *adam_lines,
                theory_legend_line,
            ),
            log_x=True,
            log_y=True,
            xlabel="$\\alpha$",
            ylabel="$y_T$",
            ylim=[1e-35, 1],
        ),
        nr=1,
        figsize=[11, 2.7],
        h_pad=0.05,
        wr=[1, 0.1, 1, 0.1, 1.2],
    )
    mp.save(
        "momentum",
        dir_name="paper/figures/img",
    )

def momentum_convergence_rate(
    alpha:  float,
    beta:   float,
    D:      int,
) -> float:
    A_list = [
        [
            1 - 2*alpha*beta + ((alpha*beta)**2)*(D+2),
            -2*alpha*(1 - beta)*(1 - alpha*beta),
            (alpha*(1 - beta))**2,
        ],
        [
            beta*(1 - alpha*beta*(D+2)),
            (1 - beta)*(1 - 2*alpha*beta),
            -alpha*((1 - beta)**2),
        ],
        [
            (beta**2)*(D+2),
            2*beta*(1 - beta),
            (1 - beta)**2,
        ],
    ]

    A = torch.tensor(A_list)
    e = torch.linalg.eigvals(A)
    rho = torch.abs(e).max().item()

    return rho

class TheoryPlotter:
    def convergence_rate(self, lr: float) -> float:
        raise NotImplementedError()

    def get_yt(self, y0: float, steps: int, lr: float) -> float:
        log_yt = steps*math.log(self.convergence_rate(lr)) + math.log(y0)
        yt = math.exp(log_yt)
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

class Bp(TheoryPlotter):
    def convergence_rate(self, lr: float) -> float:
        return (1 - lr) ** 2

class Fg(TheoryPlotter):
    def __init__(self, d: int, s: int):
        self.d = d
        self.s = s

    def convergence_rate(self, lr: float) -> float:
        return ((1 - lr)**2) + (lr**2)*(self.d + 1)/self.s

class FgMomentum(TheoryPlotter):
    def __init__(self, d: int, s: int, m: float):
        self.d = d
        self.s = s
        self.m = m

    def convergence_rate(self, lr: float) -> float:
        return momentum_convergence_rate(lr, self.m, self.d)

def train_learner(
    learner:    Learner,
    steps:      int,
) -> float:
    f = Quadratic()

    for _ in range(steps):
        learner.step(f)

    return learner.forward(f).item()

if __name__ == "__main__":
    with util.Timer("main"):
        main()
