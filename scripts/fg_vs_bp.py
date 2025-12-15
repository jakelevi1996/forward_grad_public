import torch
from jutility import plotting, util, cli
from forward_grad_public.objective import Quadratic
from forward_grad_public.learner import (
    Learner,
    BackProp,
    ForwardGrad,
)

def main():

    d0 = torch.tensor([1, 2, 5, 10, 20, 50])
    cp = plotting.ColourPicker(2)
    mp = plotting.MultiPlot(
        make_subplot(0.1,   100,    d0,     100,    cp),
        make_subplot(0.01,  1000,   10*d0,  5,      cp),
        make_subplot(0.001, 10000,  100*d0, 5,      cp),
        legend=plotting.FigureLegend(
            plotting.Line(c=cp(0), label="BP"),
            plotting.Line(c=cp(1), label="FG"),
            plotting.Line(c="k", label="Theory",    ls=":"),
            plotting.Line(c="k", label="$y_0$",     ls="--"),
            num_rows=None,
            loc="outside center right",
        ),
        nr=1,
        figsize=[10, 2.6],
        h_pad=0,
    )
    mp.save(
        "fg_vs_bp",
        dir_name="paper/figures/img",
    )

def make_subplot(
    alpha:  float,
    steps:  int,
    dims:   torch.Tensor,
    seeds:  int,
    cp:     plotting.ColourPicker,
):
    ns = plotting.NoisySweep()
    table = util.Table(
        util.TimeColumn(),
        util.CountColumn(),
        util.Column("dim"),
        util.Column("seed"),
        util.Column("y_bp", ".5f"),
        util.Column("y_fg", ".5f"),
    )

    for d in dims.tolist():
        for s in range(seeds):
            learner = BackProp(d, alpha)
            y_bp = train_learner(learner, steps, s)
            ns.update("BP", d, y_bp)

            learner = ForwardGrad(d, alpha)
            y_fg = train_learner(learner, steps, s)
            ns.update("FG", d, y_fg)

            table.update(dim=d, seed=s, y_bp=y_bp, y_fg=y_fg)

    xlim = [0.8 * min(dims), 1.2 * max(dims)]
    d = torch.tensor(util.log_range(*xlim, 200))
    r_bp = (1 - alpha) ** 2
    r_fg = r_bp + (alpha ** 2) * (1 + d)
    y0 = 0.5
    y_bp = y0 * (r_bp ** steps)
    y_fg = y0 * (r_fg ** steps)
    return plotting.Subplot(
        *ns.plot(cp=cp, key_order=["BP", "FG"], alpha_fill=0),
        plotting.HLine(y0,      c="k", ls="--", z=40),
        plotting.HLine(y_bp,    c="k", ls=":",  z=40),
        plotting.Line(d, y_fg,  c="k", ls=":",  z=40),
        log_x=True,
        log_y=True,
        xlim=xlim,
        xlabel="$\\Delta$",
        ylabel="$y_T$",
        title="$\\alpha=%s, T=%i$" % (alpha, steps),
    )

def train_learner(
    learner:    Learner,
    steps:      int,
    seed:       int,
) -> float:
    torch.manual_seed(seed)

    f = Quadratic()

    for _ in range(steps):
        learner.step(f)

    return learner.forward(f).item()

if __name__ == "__main__":
    with util.Timer("main"):
        main()
