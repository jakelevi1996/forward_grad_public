import math
import torch
import torch.utils.data
import torchvision
from jutility import plotting, util, cli

def main(
    args:           cli.ParsedArgs,
    seed:           int,
    repeats:        int,
    epochs:         int,
    batch_size:     int,
    widths:         list[int],
    lrs:            list[float],
    pops:           list[int],
    max_loss:       float,
    max_batches:    int,
    loss_lims:      tuple[float, float] | None,
    acc_lims:       tuple[float, float] | None,
):
    torch.manual_seed(seed)

    dataset = Mnist()

    table = util.Table(
        util.CountColumn(),
        util.TimeColumn(),
        util.Column("lr",       ".5f"),
        util.Column("width",    "i", 5),
        util.Column("repeat",   "i", 6),
        util.Column("model",    "s", 11),
        util.Column("batch",    "i", 5),
        util.Column("epoch",    "i", 5),
        util.Column("loss",     ".5f"),
        util.Column("test_acc", ".5f"),
        print_interval=util.TimeInterval(1),
    )

    model_names = [
        "BP",
        *[
            "FG$(S=%i)$" % p
            for p in pops
        ],
        *[
            "ES$(S=%i)$" % p
            for p in pops
        ],
    ]
    cp = plotting.ColourPicker.contrast()

    loss_subplots = []
    acc_subplots = []

    for lr in lrs:
        ns_loss = plotting.NoisySweep()
        ns_acc = plotting.NoisySweep()

        for w in widths:
            for r in range(repeats):
                kwargs = {
                    "input_dim":    dataset.get_input_dim(),
                    "output_dim":   dataset.get_output_dim(),
                    "hidden_dim":   w,
                    "lr":           lr,
                }
                models = [
                    Bp(**kwargs),
                    *[
                        Fg(**kwargs, pop_size=p)
                        for p in pops
                    ],
                    *[
                        Es(**kwargs, pop_size=p)
                        for p in pops
                    ],
                ]

                for m in models:
                    table.update(
                        level=1,
                        lr=lr,
                        width=w,
                        repeat=r,
                        model=m,
                    )

                    results = train(
                        model=m,
                        dataset=dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        max_loss=max_loss,
                        max_batches=max_batches,
                        table=table,
                    )
                    if results is not None:
                        train_loss, test_acc = results
                        ns_loss.update(str(m), w, train_loss)
                        ns_acc.update(str(m), w, test_acc)
                        table.update(level=1, model=m, loss=train_loss)

                    h = table.format_header()
                    print("-" * (len(h)//2))
                    print(h)

        kw_loss = dict()
        kw_acc = dict()
        if lr == lrs[0]:
            kw_loss["ylabel"] = "Train loss ($\\downarrow$)"
            kw_acc["ylabel"] = "Test acc ($\\uparrow$)"

        sp_loss = plotting.Subplot(
            *ns_loss.plot(cp, model_names),
            ylim=loss_lims,
            title="$\\alpha=%s$" % lr,
            log_x=True,
            **kw_loss,
        )
        sp_acc = plotting.Subplot(
            *ns_acc.plot(cp, model_names),
            ylim=acc_lims,
            xlabel="Width",
            log_x=True,
            xticks=widths,
            xticklabels=widths,
            **kw_acc,
        )
        loss_subplots.append(sp_loss)
        acc_subplots.append(sp_acc)

    util.save_pickle(
        [loss_subplots, acc_subplots, ns_loss],
        "sweep_mlp",
        dir_name="paper/figures/img",
    )
    plotting.set_latex_params()
    mp = plotting.MultiPlot(
        *loss_subplots,
        *acc_subplots,
        legend=plotting.FigureLegend(
            *cp.get_legend_sweeps(*model_names),
            num_rows=None,
            loc="outside center right",
        ),
        nr=2,
        figsize=[10, 5.2],
        h_pad=0,
        sharex=True,
        sharey=True,
    )
    mp.save(
        "sweep_mlp",
        dir_name="paper/figures/img",
        pdf=True,
    )

class Mlp(torch.nn.Module):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        hidden_dim: int,
        lr:         float,
        pop_size:   (int | None)=None,
        sigma:      float=1e-5,
    ):
        super().__init__()

        w1_scale = 1 / math.sqrt(input_dim)
        w2_scale = 1 / math.sqrt(hidden_dim)
        w1_shape = [input_dim, hidden_dim]
        w2_shape = [hidden_dim, output_dim]

        self.w1_ih = torch.nn.Parameter(torch.normal(0, w1_scale, w1_shape))
        self.w2_ho = torch.nn.Parameter(torch.normal(0, w2_scale, w2_shape))

        self.b1_h = torch.nn.Parameter(torch.zeros([hidden_dim]))
        self.b2_o = torch.nn.Parameter(torch.zeros([output_dim]))

        self.opt = torch.optim.SGD(self.parameters(), lr)

        self.eps_w1_shape = [pop_size, *w1_shape]
        self.eps_w2_shape = [pop_size, *w2_shape]
        self.eps_b1_shape = [pop_size, 1, hidden_dim]
        self.eps_b2_shape = [pop_size, 1, output_dim]
        self.dloss_shape  = [pop_size, 1, 1]
        self.output_dim   = output_dim
        self.pop_size     = pop_size
        self.sigma        = sigma

    def forward(self, x_ni: torch.Tensor) -> torch.Tensor:
        x_nh = x_ni @ self.w1_ih + self.b1_h
        x_nh = torch.relu(x_nh)
        y_no = x_nh @ self.w2_ho + self.b2_o
        return y_no

    def step(self, x_ni: torch.Tensor, t_n: torch.Tensor) -> float:
        raise NotImplementedError()

class Bp(Mlp):
    def step(self, x_ni: torch.Tensor, t_n: torch.Tensor) -> float:
        y_no = self.forward(x_ni)
        loss = torch.nn.functional.cross_entropy(y_no, t_n)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def __str__(self) -> str:
        return "BP"

class Fg(Mlp):
    def step(self, x_ni: torch.Tensor, t_n: torch.Tensor) -> float:
        eps_w1_sih = torch.normal(0, 1, self.eps_w1_shape)
        eps_w2_sho = torch.normal(0, 1, self.eps_w2_shape)
        eps_b1_s1h = torch.normal(0, 1, self.eps_b1_shape)
        eps_b2_s1o = torch.normal(0, 1, self.eps_b2_shape)
        t_no = torch.nn.functional.one_hot(t_n, self.output_dim).float()

        a_nh   = x_ni @ self.w1_ih + self.b1_h
        da_snh = x_ni @ eps_w1_sih + eps_b1_s1h

        z_nh   = torch.relu(a_nh)
        dz_snh = torch.where(a_nh > 0, 1.0, 0.0) * da_snh

        y_no   = z_nh @ self.w2_ho + self.b2_o
        dy_sno = z_nh @ eps_w2_sho + eps_b2_s1o + dz_snh @ self.w2_ho

        loss    = -((y_no * t_no).sum(-1) - y_no.logsumexp(-1)).mean()
        dloss_s = -((t_no - y_no.softmax(-1)) * dy_sno).sum(-1).mean(-1)

        dloss_s11 = dloss_s.reshape(self.dloss_shape)

        self.w1_ih.grad = (dloss_s11 * eps_w1_sih).mean(0)
        self.w2_ho.grad = (dloss_s11 * eps_w2_sho).mean(0)
        self.b1_h.grad  = (dloss_s11 * eps_b1_s1h).mean([0, 1])
        self.b2_o.grad  = (dloss_s11 * eps_b2_s1o).mean([0, 1])

        self.opt.step()

        return loss.item()

    def __str__(self) -> str:
        return "FG$(S=%i)$" % self.pop_size

class Es(Mlp):
    def step(self, x_ni: torch.Tensor, t_n: torch.Tensor) -> float:
        eps_w1_sih = torch.normal(0, self.sigma, self.eps_w1_shape)
        eps_w2_sho = torch.normal(0, self.sigma, self.eps_w2_shape)
        eps_b1_s1h = torch.normal(0, self.sigma, self.eps_b1_shape)
        eps_b2_s1o = torch.normal(0, self.sigma, self.eps_b2_shape)
        t_no = torch.nn.functional.one_hot(t_n, self.output_dim).float()

        xp_snh = x_ni @ (self.w1_ih + eps_w1_sih) + (self.b1_h + eps_b1_s1h)
        xm_snh = x_ni @ (self.w1_ih - eps_w1_sih) + (self.b1_h - eps_b1_s1h)
        xp_snh = torch.relu(xp_snh)
        xm_snh = torch.relu(xm_snh)
        yp_sno = xp_snh @ (self.w2_ho + eps_w2_sho) + (self.b2_o + eps_b2_s1o)
        ym_sno = xm_snh @ (self.w2_ho - eps_w2_sho) + (self.b2_o - eps_b2_s1o)

        lossp_s = -((yp_sno * t_no).sum(-1) - yp_sno.logsumexp(-1)).mean(-1)
        lossm_s = -((ym_sno * t_no).sum(-1) - ym_sno.logsumexp(-1)).mean(-1)

        eps_scale_s     = (lossp_s - lossm_s) / (2 * self.sigma * self.sigma)
        eps_scale_s11   = eps_scale_s.unsqueeze(-1).unsqueeze(-1)

        self.w1_ih.grad = (eps_scale_s11 * eps_w1_sih).mean(0)
        self.w2_ho.grad = (eps_scale_s11 * eps_w2_sho).mean(0)
        self.b1_h.grad  = (eps_scale_s11 * eps_b1_s1h).mean([0, 1])
        self.b2_o.grad  = (eps_scale_s11 * eps_b2_s1o).mean([0, 1])

        self.opt.step()

        return (lossp_s + lossm_s).mean().item() / 2

    def __str__(self) -> str:
        return "ES$(S=%i)$" % self.pop_size

class Mnist:
    def __init__(self):
        self.split_dict = {
            "train": torchvision.datasets.MNIST(
                root="data",
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
            "test": torchvision.datasets.MNIST(
                root="data",
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
        }

    @classmethod
    def get_input_dim(cls) -> int:
        return 28*28

    @classmethod
    def get_output_dim(cls) -> int:
        return 10

    def get_split(self, split: str) -> torch.utils.data.Dataset:
        return self.split_dict[split]

    def get_data_loader(
        self,
        split:      str,
        batch_size: int,
        shuffle:    bool=True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return torch.utils.data.DataLoader(
            dataset=self.get_split(split),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_full_batch(
        self,
        split:  str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_loader = self.get_data_loader(
            split=split,
            batch_size=len(self.get_split(split)),
            shuffle=False,
        )
        x, t = next(iter(data_loader))
        x, t = self.format_batch(x, t)
        return x, t

    def format_batch(
        self,
        x:      torch.Tensor,
        t:      torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(-3, -1)
        return x, t

def multiclass_acc(y: torch.Tensor, t: torch.Tensor) -> float:
    y_hard = y.argmax(-1)
    acc = torch.where(t == y_hard, 1.0, 0.0).mean().item()
    return acc

def train(
    model:          Mlp,
    dataset:        Mnist,
    batch_size:     int,
    epochs:         int,
    max_loss:       float,
    max_batches:    int,
    table:          util.Table,
) -> tuple[float, float] | None:

    train_loader = dataset.get_data_loader("train", batch_size)
    x_train, t_train = dataset.get_full_batch("train")
    x_test,  t_test  = dataset.get_full_batch("test")

    for e in range(epochs):
        for b, (x, t) in enumerate(train_loader):
            x, t = dataset.format_batch(x, t)
            loss = model.step(x, t)
            table.update(
                epoch=e,
                batch=b,
                loss=loss,
            )
            if (max_batches is not None) and (b+1 >= max_batches):
                break
            if loss > max_loss:
                table.print_last()
                return None

        table.print_last()
        y = model.forward(x_test)
        acc = multiclass_acc(y, t_test)
        table.update(
            level=1,
            epoch=e,
            test_acc=acc
        )

    y = model.forward(x_train)
    train_loss = torch.nn.functional.cross_entropy(y, t_train).item()
    test_acc = table.get_item("test_acc", -1)

    return train_loss, test_acc

if __name__ == "__main__":
    parser = cli.Parser(
        cli.Arg("seed",         type=int, default=0),
        cli.Arg("repeats",      type=int, default=3),
        cli.Arg("epochs",       type=int, default=3),
        cli.Arg("batch_size",   type=int, default=100),
        cli.JsonArg(
            "widths",
            default=[20, 50, 100, 200, 500, 1000],
            help="EG --widths '[20, 50, 100]'",
        ),
        cli.JsonArg(
            "lrs",
            default=[0.01, 0.05, 0.1],
            help="EG --lrs '[0.01, 0.1]'",
        ),
        cli.JsonArg(
            "pops",
            default=[10, 100],
            help="EG --pops '[1, 10]'",
        ),
        cli.Arg("max_loss",     type=float, default=100.0),
        cli.Arg("max_batches",  type=int,   default=None),
        cli.Arg("loss_lims", type=float, default=[0.0, 3.0], nargs=2),
        cli.Arg("acc_lims",  type=float, default=[0.0, 1.0], nargs=2),
    )
    args = parser.parse_args()

    with util.Timer("main"):
        main(args, **args.get_kwargs())
