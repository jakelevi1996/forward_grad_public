# forward_grad_public

Public repository for paper "On The Scalability Of Forward Gradients, Evolutionary Strategies, And Control Variates"

## Installation

This package can be installed locally in "editable mode" with the following commands:

```
python -m pip install -U pip
python -m pip install -U jutility==0.0.29
git clone https://github.com/jakelevi1996/forward_grad_public.git
cd forward_grad_public
python -m pip install -e .
```

This package has `torch` as a dependency, and has been tested with version `2.1.2+cpu`. Installation instructions for `torch` vary depending on available hardware, see [official installation instructions](https://pytorch.org/get-started/locally/).

## Create figures

All figures can be created with the following commands:

```
python scripts/fg_vs_bp.py

python scripts/momentum.py

python scripts/cv.py

python scripts/fg_sg.py

python scripts/bfg.py
```
