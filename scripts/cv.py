import torch
from jutility import plotting, util, cli
from forward_grad_public.objective import Quadratic
from forward_grad_public.learner import (
    Learner,
    ForwardGrad,
    ForwardGradMinDd,
    ForwardGradDiagonalFunctional,
    ForwardGradCvBias,
    ForwardGradCvDiagBias,
)

def main():
    D = 100

    alpha_list = util.log_range(0.001, 0.02, 100)
    beta_list  = util.log_range(1e-5, 1, 100)
    ylim = [0.98, 1.01]

    alpha_best = 1 / (D + 2)
    rho_best = (D + 1) / (D + 2)

    ns_bs = plotting.NoisySweep()

    for alpha in alpha_list:
        for beta in beta_list:
            rho = cv_convergence_rate(alpha, beta, D)
            ns_bs.update(beta, alpha, rho)

    rho_best_sweep = min(ns_bs)
    print(rho_best, alpha_best)
    print(rho_best_sweep, ns_bs.inverse(rho_best_sweep))
    util.hline()

    ns_stl = plotting.NoisySweep(log_y=True)
    table = util.Table(
        util.TimeColumn(),
        util.CountColumn(),
        util.Column("alpha", ".5f"),
        util.Column("repeat"),
        util.Column("name", width=-30),
        util.Column("y", ".5f"),
    )

    torch.manual_seed(0)
    alpha_list_stl = util.log_range(5e-4, 2e-2, 20)
    repeats = 5
    steps = 5000

    beta_diag_bias  = 0.00774
    beta_diag       = 0.005
    beta_bias       = 0.00464
    beta_dense      = 7.74e-05

    name_fg         = "FG"
    name_diag_bias  = "FG+diag+bias($\\beta$=%.2e)" % beta_diag_bias
    name_diag       = "FG+diag($\\beta$=%.2e)"      % beta_diag
    name_bias       = "FG+bias($\\beta$=%.2e)"      % beta_bias
    name_dense      = "FG+dense($\\beta$=%.2e)"     % beta_dense

    for alpha in alpha_list_stl.tolist():
        for i in range(repeats):
            learner = ForwardGrad(D, alpha)
            y = train_learner(learner, steps)
            ns_stl.update(name_fg, alpha, y)
            table.update(alpha=alpha, repeat=i, name=name_fg, y=y)

            learner = ForwardGradCvDiagBias(D, alpha, beta_diag_bias)
            y = train_learner(learner, steps)
            ns_stl.update(name_diag_bias, alpha, y)
            table.update(alpha=alpha, repeat=i, name=name_diag_bias, y=y)

            learner = ForwardGradDiagonalFunctional(D, alpha, beta_diag)
            y = train_learner(learner, steps)
            ns_stl.update(name_diag, alpha, y)
            table.update(alpha=alpha, repeat=i, name=name_diag, y=y)

            learner = ForwardGradCvBias(D, alpha, beta_bias)
            y = train_learner(learner, steps)
            ns_stl.update(name_bias, alpha, y)
            table.update(alpha=alpha, repeat=i, name=name_bias, y=y)

            learner = ForwardGradMinDd(D, alpha, beta_dense, bias=False)
            y = train_learner(learner, steps)
            ns_stl.update(name_dense, alpha, y)
            table.update(alpha=alpha, repeat=i, name=name_dense, y=y)

    cp = plotting.ColourPicker(len(ns_stl), cmap_name="gist_rainbow")
    stl_lines = ns_stl.plot(cp=cp)

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *ns_bs.plot(
                cp=plotting.ColourPicker(len(beta_list), cyclic=False),
                alpha_fill=0,
                alpha_scat=0,
            ),
            plotting.VLine(alpha_best, c="k", ls="--", z=50),
            plotting.HLine(rho_best, c="k", ls="--", z=50),
            xlabel="$\\alpha$",
            ylabel="$\\rho$",
            log_x=True,
            ylim=ylim,
        ),
        plotting.ColourBar(
            beta_list.min(),
            beta_list.max(),
            log=True,
            label="$\\beta$",
            cmap="cool",
        ),
        plotting.Subplot(
            *ns_bs.transpose().plot(
                cp=plotting.ColourPicker(len(alpha_list), cyclic=False),
                alpha_fill=0,
                alpha_scat=0,
            ),
            plotting.HLine(rho_best, c="k", ls="--", z=50),
            xlabel="$\\beta$",
            ylabel="$\\rho$",
            log_x=True,
            ylim=ylim,
        ),
        plotting.ColourBar(
            alpha_list.min(),
            alpha_list.max(),
            log=True,
            label="$\\alpha$",
            cmap="cool",
        ),
        plotting.Subplot(
            *stl_lines,
            plotting.Legend.from_plottables(*stl_lines, framealpha=0.7, z=50),
            log_y=True,
            xlabel="$\\alpha$",
            ylabel="$y_T$",
            ylim=[None, 1],
        ),
        nr=1,
        figsize=[11, 2.7],
        h_pad=0.03,
        wr=[1, 0.1, 1, 0.1, 1.2],
    )
    mp.save(
        "cv",
        dir_name="paper/figures/img",
    )

def cv_convergence_rate(
    alpha:  float,
    beta:   float,
    D:      int,
) -> float:
    A_list = [
        [
            1 - 2*alpha + (alpha**2)*(D+2),
            -2*(alpha**2)*(D+1),
            (alpha**2)*(D+1),
        ],
        [
            beta*(1 - alpha*(D+2)),
            (1 - alpha)*(1 - beta) + alpha*beta*(D+1),
            0,
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

def train_learner(
    learner:    Learner,
    steps:      int,
) -> float:
    f = Quadratic()
    learner.x.fill_(1.0)

    for _ in range(steps):
        learner.step(f)

    return learner.forward(f).item()

if __name__ == "__main__":
    with util.Timer("main"):
        main()
