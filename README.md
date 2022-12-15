# [MOGPJax](https://github.com/JaxGaussianProcesses/MOGPJax)

[![PyPI version](https://badge.fury.io/py/MOGPJax.svg)](https://badge.fury.io/py/MOGPJax)

`MOGPJax` aims to provide a low-level interface to multi-output Gaussian process (GP) models in [`Jax`](https://github.com/google/jax), structured to give researchers maximum flexibility in extending the code to suit their own needs.

Currently the library is under major development.

# Installation

## Stable version

The latest stable version of `MOGPJax` can be installed via [`pip`](https://pip.pypa.io/en/stable/):

```bash
pip install mogpjax
```

> **Note**
>
> We recommend you check your installation version:
> ```python
> python -c 'import mogpjax; print(mogpjax.__version__)'
> ```



## Development version
> **Warning**
>
> This version is possibly unstable and may contain bugs. 

Clone a copy of the repository to your local machine and run the setup configuration in development mode.
```bash
git clone https://github.com/JaxGaussianProcesses/MOGPJax.git
cd mogpjax
python -m setup develop
```

> **Note**
>
> We advise you create virtual environment before installing:
> ```
> conda create -n mogpjax_ex python=3.10.0
> conda activate mogpjax_ex
>  ```
>
> and recommend you check your installation passes the supplied unit tests:
>
> ```python
> python -m pytest tests/
> ```
