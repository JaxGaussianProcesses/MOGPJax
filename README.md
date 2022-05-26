<h1 align='center'>MOGPJax</h1>
<h2 align='center'>Multi-output Gaussian processes in Jax.</h2>

[![codecov](https://codecov.io/gh/daniel-dodd/mogpjax/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/daniel-dodd/mogpjax)
[![CodeFactor](https://www.codefactor.io/repository/github/danieldodd/mogpjax/badge)](https://www.codefactor.io/repository/github/danieldodd/mogpjax)
[![Documentation Status](https://readthedocs.org/projects/mogpjax/badge/?version=latest)](https://gpjax.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/MOGPJax.svg)](https://badge.fury.io/py/MOGPJax)
[![Downloads](https://pepy.tech/badge/mogpjax)](https://pepy.tech/project/mogpjax)


[**Install guide**](#installation)
| [**Documentation**](https://mogpjax.readthedocs.io/en/latest/)

Through extending the GPJax package, MOGPJax aims to provide a low-level interface to multi-output Gaussian process (GP) models in [Jax](https://github.com/google/jax), structured to give researchers maximum flexibility in extending the code to suit their own needs.

## Installation

### Stable version

To install the latest stable version of MOGPJax run

```bash
pip install mogpjax
```

### Development version

To install the latest, possibly unstable, version, the following steps should be followed. It is by no means compulsory, but we do advise that you do all of the below inside a virtual environment.

```bash
git clone https://github.com/daniel-dodd/MOGPJax.git
cd MOGPJax
python setup.py develop
```

We then recommend you check your installation using the supplied unit tests.

```python
python -m pytest tests/
```
