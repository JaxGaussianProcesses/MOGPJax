# Copyright 2022 The MOGPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, Callable
from jaxtyping import Float, Array

import jax.numpy as jnp
from jax import vmap
from jax.random import KeyArray
import distrax as dx

import gpjax as gpx
from gpjax.utils import concat_dictionaries
from jaxutils import PyTree, Dataset

class AbstractGP(PyTree):
    pass

class Prior(AbstractGP):
    pass

class AbstractPosterior(AbstractGP):
    pass

class ConjugatePosterior(AbstractPosterior):
    pass

class NonConjugatePosterior(AbstractPosterior):
    pass


class GPLVM(PyTree):
    """A Gaussian Process Latent Variable Model (GPLVM)."""

    def __init__(
        self,
        latent_process: gpx.gps.ConjugatePosterior,
        latent_dim: int,
        ) -> None:

        self.latent_process = latent_process
        self.latent_dim = latent_dim
    
    

    def _initialise_params(self, key: KeyArray) -> Dict:
        """Initialise the GP's parameter set.

        Args:
            key (KeyArray): The random key to use for initialisation.

        Returns:
            Dict: The initial parameter set.
        """

        latent_process = self.latent_process
        n_data = latent_process.likelihood.num_datapoints

        latent_prior = dx.MultivariateNormalDiag(
            loc=jnp.zeros(self.latent_dim), 
            scale_diag=jnp.ones(self.latent_dim)
        )

        model_params = self.latent_process._initialise_params(key)

        X = latent_prior.sample(seed=key, sample_shape=(n_data,))

        return concat_dictionaries(model_params, {"latent": X})

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        priors: Dict = None,
        negative: bool = False,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Compute the marginal log likelihood of the model.
        Args:
            train_data (Dataset): The training dataset.
            priors (dict): The priors to apply to the training dataset.
            negative (bool): Whether to return the negative log likelihood.
        Returns:
            Callable[[Dict], Float[Array, "1"]]: A function that takes a dictionary of parameters and returns the marginal log likelihood.
        """

        if not train_data.is_unsupervised():
            raise ValueError("Data must be unsupervised.")

        def _single_mll(y: Float[Array, "N Q"] , params: Dict):
            X = params["latent"]
            D = Dataset(X, y)
            
            mll = self.latent_process.marginal_log_likelihood(
                D, priors
            )
            return mll(params)

        constant = jnp.array(-1.0) if negative else jnp.array(1.0)
        y = train_data.y

        def mll(params):
            return constant * jnp.sum(
                vmap(_single_mll, in_axes=(0, None))(
                    jnp.transpose(y), params
                )
            )

        return mll

__all__ = [
    "GPLVM",
]
