import torch
from jutility import plotting, util, cli

def main():
    make_convergence_rate_plot()
    make_optimal_lr_plot()

def make_convergence_rate_plot():
    mp = plotting.MultiPlot(
        make_subplot(1, [0, 1]),
        make_subplot(100, [0.9, 1]),
        plotting.ColourBar(1e-3, 1, cmap="cool", log=True, label="$\\alpha$"),
        legend=plotting.FigureLegend(
            plotting.Line(c="k", ls="--",   label="Best FG"),
            plotting.Line(c="k", ls=":",    label="Best SG"),
        ),
        nr=1,
        wr=[1, 1, 0.1],
        figsize=[8, 3],
        h_pad=0.03,
    )
    mp.save(
        "fg_sg",
        dir_name="paper/figures/img",
    )

def make_subplot(
    D:      int,
    ylim:   tuple[float, float],
):
    alpha_list = util.log_range(1e-3, 1, 100)
    beta_list  = util.log_range(1e-3, 1, 100)

    nds = plotting.NoisySweep()

    for alpha in alpha_list:
        for beta in beta_list:
            rho = fg_sg_convergence_rate(alpha, beta, D)
            nds.update(alpha, beta, rho)

    fg_best = (D + 1) / (D + 2)

    sg_best = [
        min(
            sg_convergence_rate(alpha, beta)
            for alpha in alpha_list
        )
        for beta in beta_list
    ]

    return plotting.Subplot(
        *nds.plot(
            cp=plotting.ColourPicker.cool(len(alpha_list)),
            alpha_fill=0,
            alpha_scat=0,
        ),
        plotting.HLine(fg_best, c="k", ls="--", z=50),
        plotting.Line(beta_list, sg_best, c="k", ls=":", z=50),
        xlabel="$\\beta$",
        ylabel="$\\rho$",
        log_x=True,
        ylim=ylim,
        title="$D = %i$" % D,
    )

def make_optimal_lr_plot():
    alpha_list  = util.log_range(1e-3, 1, 100)
    beta_list   = util.log_range(1e-3, 1, 100)
    D_list      = [1, 3, 10, 30, 100, 300, 1000]

    cp = plotting.ColourPicker.cool(len(D_list))

    best_alpha = [
        plotting.Line(
            beta_list,
            [
                min(
                    alpha_list,
                    key=(
                        lambda alpha: fg_sg_convergence_rate(alpha, beta, D)
                    ),
                )
                for beta in beta_list
            ],
            c=c,
            label=str(D),
        )
        for D, c in zip(D_list, cp)
    ]
    best_rho = [
        plotting.Line(
            beta_list,
            [
                min(
                    fg_sg_convergence_rate(alpha, beta, D)
                    for alpha in alpha_list
                )
                for beta in beta_list
            ],
            c=c,
            label=str(D),
        )
        for D, c in zip(D_list, cp)
    ]

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *best_alpha,
            log_x=True,
            log_y=True,
            xlabel="$\\beta$",
            ylabel="$\\alpha^*$",
        ),
        plotting.Subplot(
            *best_rho,
            log_x=True,
            xlabel="$\\beta$",
            ylabel="$\\rho^*$",
        ),
        legend=plotting.FigureLegend(
            *best_alpha,
            loc="outside center right",
            num_rows=None,
            title="$D$",
        ),
        figsize=[8, 3],
    )
    mp.save(
        "fg_sg_optimal_lr",
        dir_name="paper/figures/img",
    )

def fg_sg_convergence_rate(
    alpha:  float,
    beta:   float,
    D:      int,
) -> float:
    A_list = [
        [
            (
                ((1 - alpha*beta)**2)
                - 2*alpha*(1 - beta)*(1 - alpha*beta)
                + (alpha**2)*((1 - beta)**2)*(D + 2)
            ),
            -2*(alpha**2)*((1 - beta)**2)*(D + 1),
            (alpha**2)*((1 - beta)**2)*(D + 1),
        ],
        [
            beta*(1 - alpha),
            (1 - alpha)*(1 - beta),
            0,
        ],
        [
            beta**2,
            2*beta*(1 - beta),
            (1 - beta)**2,
        ],
    ]

    A = torch.tensor(A_list)
    e = torch.linalg.eigvals(A)
    c = torch.abs(e).max().item()

    return c

def sg_convergence_rate(
    alpha:  float,
    beta:   float,
) -> float:
    A_list = [
        [
            (1 - alpha*beta)**2,
            -2*alpha*(1 - beta)*(1 - alpha*beta),
            (alpha**2)*((1 - beta)**2),
        ],
        [
            beta*(1 - alpha*beta),
            (1 - beta)*(1 - 2*alpha*beta),
            -alpha*((1 - beta)**2),
        ],
        [
            beta**2,
            2*beta*(1 - beta),
            (1 - beta)**2,
        ],
    ]

    A = torch.tensor(A_list)
    e = torch.linalg.eigvals(A)
    c = torch.abs(e).max().item()

    return c

if __name__ == "__main__":
    with util.Timer("main"):
        main()
