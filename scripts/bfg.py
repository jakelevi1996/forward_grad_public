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
    alpha_list = util.log_range(1e-3, 3e-1, 20)
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

    return plotting.Subplot(
        *nds.plot(),
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

if __name__ == "__main__":
    with util.Timer("main"):
        main()
