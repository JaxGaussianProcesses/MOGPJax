# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gaussian Process Latent Variable Models
#
# The Gaussian process latent variable model (GPLVM) <strong data-cite="lawrence2003gaussian"></strong> employs GPs to learn a low-dimensional latent space representation of a high-dimensional, unsupervised dataset. Within this notebook, we use 3-phase oil flow data whose use is demonstrated in [GPFlow's GPLVM notebook](https://gpflow.github.io/GPflow/2.5.2/notebooks/basics/GPLVM.html).

import io

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import jaxkern.kernels as jk
import matplotlib.pyplot as plt

# %%
import numpy as np
import optax as ox
import requests
from jax import jit
from jaxutils import Dataset
from sklearn.decomposition import PCA

import mogpjax as mgpx

key = jr.PRNGKey(123)

response = requests.get(
    "https://github.com/GPflow/GPflow/blob/develop/doc/sphinx/notebooks/advanced/data/three_phase_oil_flow.npz?raw=true"
)
response.raise_for_status()
data = np.load(io.BytesIO(response.content))
labels = jnp.asarray(data["labels"])
observations = jnp.asarray(data["Y"])

# %% [markdown]
# ## Data
#
# The [3-Phase Oil Flow Data](https://inverseprobability.com/3PhaseData) contains 100, 12-dimensional observations with a three-levelled categorical label. Each observation corresponds to measurements made of an oil, water and gas pipeline and the corresponding label describes whether the measurement was made in a 1) horizontally stratified, 2) nested annular, or 3) homogenous mixture flow, configuration.
#
# ## Model specification
#
# GPLVMs use a set of $Q$ Gaussian process $(f_1, f_2, \ldots, f_Q)$ to project from the latent space $\mathbf{X}\in\mathbb{R}^{N\times Q}$ to the observed dataset $\mathbf{Y}\in\mathbb{R}^{N\times D}$ where $Q\ll D$. The hierarchical model can then be written as
# $$\begin{align}
# p(\mathbf{X}) & = \prod_{n=1}^N \mathcal{N}(\mathbf{x}_{n}\mid\mathbf{0}, \mathbf{I}_Q) \\
# p(\mathbf{f}\mid \mathbf{X}, \mathbf{\theta}) & = \prod_{d=1}^{D} \mathcal{N}(\mathbf{f}_{d}\mid \mathbf{0}, \mathbf{K}_{\mathbf{ff}}) \\
# p(\mathbf{Y}\mid\mathbf{f}, \mathbf{X}) & = \prod_{n=1}^{N}\prod_{d=1}^{D}\mathcal{N}(y_{n, d}\mid f_d(\mathbf{x}_n), \sigma^2)
# \end{align}
# $$
# where $\mathbf{f}_d = f_d(\mathbf{X})$. In the GPLVM implemented with MOGPJax, we perform MAP estimation to learn the latent coordinates to enable analytical marginalisation of the latent GP. In the future, support for the Bayesian GPLVM <strong data-cite="titsias2010bayesian"></strong> is something we'd like to support within MOGPJax whereby the latent coordiantes are jointly marginalised from the model.
#
# To interface with the GPLVM presented in MOGPJax, we require used to explicitly define the latent process that is responsible for projecting from the latent space using the GP objects given in [GPJax](https://github.com/thomaspinder/GPJax) <strong data-cite="pinder2022gpjax"></strong>. The latent process and the desired dimensionality of the latent space are then consumed by the GPLVM object.

# %%
latent_dim = 2
kernel = jk.RBF(active_dims=[0, 1])
latent_process = gpx.Prior(kernel=kernel) * gpx.Gaussian(
    num_datapoints=observations.shape[0]
)

gplvm = mgpx.GPLVM(latent_process=latent_process, latent_dim=latent_dim)

# %% [markdown]
# ### Parameters
#
# We'll then initialise the parameters for our model and unconstrain their value in the regular GPJax manner. To aid inference in our model, we'll intialise the latent coordinates using principal component analysis.

# %%
parameter_state = gpx.initialise(gplvm, key=key)

obs_pca = jnp.asarray(PCA(n_components=2).fit_transform(observations))
parameter_state.params["latent"] = obs_pca

# %% [markdown]
# ## Optimisation
#
# We can now maximise the marginal log-likelihood of our GPLVM with respect to the kernel parameters, observation noise term, and the latent coordinate. We'll JIT compile this function to accelerate optimisation.

# %%
observed = Dataset(y=observations)
objective = jit(gplvm.marginal_log_likelihood(observed, negative=True))

opt = ox.adam(0.05)
learned_params, history = gpx.abstractions.fit(
    objective, parameter_state, optax_optim=opt, n_iters=1000
).unpack()

# %% [markdown]
# ## Latent space visualisation
#
# With optimisation complete, we can now visualise our latent space. To do this, we'll simply plot the 2D coordinate that has been learned for each observation and colour it by oil's tranportation type. We should note that this flow-type variable has been used only for visualisation and was not part of the matrix that we have constructed a latent representation of.

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.scatter(learned_params["latent"][:, 0], learned_params["latent"][:, 1], c=labels)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set(xlabel="Latent Coordinate 1", ylabel="Latent Coordinate 2", title="Latent Space")

# %% [markdown]
# ## System configuration

# %%
# % reload_ext watermark
# % watermark -n -u -v -iv -w -a 'Thomas Pinder'
