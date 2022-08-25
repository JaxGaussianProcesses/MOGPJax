import typing as tp

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import mogpjax as mgpx


@pytest.mark.parametrize("n_data", [10, 100])
@pytest.mark.parametrize("data_dim", [10, 20, 50])
@pytest.mark.parametrize("latent_dim", [1, 2, 5])
@pytest.mark.parametrize("kernel", [gpx.Matern12, gpx.Matern32, gpx.Matern52, gpx.RBF])
@pytest.mark.parametrize("jittable", [True, False])
def test_gplvm(n_data, data_dim, latent_dim, kernel, jittable):
    key = jr.PRNGKey(123)
    observations = jr.normal(key, shape=(n_data, data_dim))
    observed_data = mgpx.UnsupervisedDataset(y=observations)

    latent_proces = gpx.Prior(kernel=kernel()) * gpx.Gaussian(num_datapoints=n_data)
    gplvm = mgpx.GPLVM(latent_process=latent_proces, latent_dim=latent_dim)

    params, trainables, constrainers, unconstrainers = gpx.initialise(
        gplvm, key=key
    ).unpack()
    assert params["latent"].shape == (n_data, latent_dim)

    mll = gplvm.marginal_log_likelihood(
        observed_data, transformations=constrainers, negative=True
    )
    assert isinstance(mll, tp.Callable)
    if jittable:
        mll = jax.jit(mll)

    assert isinstance(mll(params), jnp.DeviceArray)
    assert mll(params).shape == ()

    mll_grad = jax.grad(mll)
    assert isinstance(mll_grad(params), tp.Dict)
