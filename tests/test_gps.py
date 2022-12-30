import typing as tp

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxkern.kernels as jk
import pytest
from jaxutils import Dataset

import mogpjax as mgpx


@pytest.mark.parametrize("n_data", [10, 100])
@pytest.mark.parametrize("data_dim", [10, 20, 50])
@pytest.mark.parametrize("latent_dim", [1, 2, 5])
@pytest.mark.parametrize("kernel", [jk.Matern12, jk.Matern32, jk.Matern52, jk.RBF])
@pytest.mark.parametrize("jittable", [True, False])
def test_gplvm(n_data, data_dim, latent_dim, kernel, jittable):
    key = jr.PRNGKey(123)

    y = jr.normal(key, shape=(n_data, data_dim))
    D = Dataset(y=y)

    latent_proces = gpx.Prior(kernel=kernel()) * gpx.Gaussian(num_datapoints=n_data)
    gplvm = mgpx.GPLVM(latent_process=latent_proces, latent_dim=latent_dim)

    params, *_ = gpx.initialise(gplvm, key=key).unpack()
    assert params["latent"].shape == (n_data, latent_dim)

    mll = gplvm.marginal_log_likelihood(D, negative=True)

    assert isinstance(mll, tp.Callable)

    if jittable:
        mll = jax.jit(mll)

    assert isinstance(mll(params), jnp.DeviceArray)
    assert mll(params).shape == ()

    mll_grad = jax.grad(mll)
    assert isinstance(mll_grad(params), tp.Dict)
