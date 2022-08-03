import typing as tp

import distrax as dx
import gpjax as gpx
import jax.numpy as jnp
from chex import dataclass
from gpjax.config import add_parameter, get_defaults
from gpjax.utils import concat_dictionaries
from jax import vmap
from jaxtyping import f64

from .data import UnsupervisedDataset

DEFAULT_JITTER = get_defaults()["jitter"]


@dataclass
class GPLVM:
    latent_process: gpx.gps.ConjugatePosterior
    latent_dim: int
    jitter: tp.Optional[float] = DEFAULT_JITTER

    @property
    def params(self) -> dict:
        default_key = get_defaults()["key"]
        n_data = self.latent_process.likelihood.num_datapoints
        latent_prior = dx.MultivariateNormalDiag(
            loc=jnp.zeros(self.latent_dim), scale_diag=jnp.ones(self.latent_dim)
        )
        X = latent_prior.sample(seed=default_key, sample_shape=(n_data,))
        model_params = concat_dictionaries(
            self.latent_process.prior.params,
            {"likelihood": self.latent_process.likelihood.params},
        )
        add_parameter("latent", dx.Lambda(lambda x: x))
        return concat_dictionaries(model_params, {"latent": X})

    def marginal_log_likelihood(
        self,
        train_data: UnsupervisedDataset,
        transformations: tp.Dict,
        priors: dict = None,
        negative: bool = False,
    ) -> tp.Callable[[dict], f64["1"]]:
        """Compute the marginal log likelihood of the model.
        Args:
            train_data (UnsupervisedDataset): The training dataset.
            transformations (dict): The transformations to apply to the training dataset.
            priors (dict): The priors to apply to the training dataset.
            negative (bool): Whether to return the negative log likelihood.
        Returns:
            Callable[[dict], f64["1"]]: A function that takes a dictionary of parameters and returns the marginal log likelihood.
        """

        def _single_mll(y, params):
            D = gpx.Dataset(X=params["latent"], y=y)
            mll = self.latent_process.marginal_log_likelihood(
                D, transformations, priors
            )
            return mll(params)

        def mll(params):
            constant = jnp.array(-1.0) if negative else jnp.array(1.0)
            return constant * jnp.sum(
                vmap(_single_mll, in_axes=(0, None))(
                    jnp.transpose(train_data.y), params
                )
            )

        return mll
