import jax.numpy as jnp
from chex import dataclass
from jaxtyping import f64


@dataclass
class _MultioutputData:
    X: jnp.DeviceArray
    y = jnp.DeviceArray

    @property
    def n(self) -> int:
        """The number of observations in the dataset."""
        return self.y.shape[0]


@dataclass
class UnsupervisedDataset:
    y: jnp.DeviceArray
    X = None

    def _add_input(self, X):
        self.X = X
        return self

    @property
    def n(self) -> int:
        """The number of observations in the dataset."""
        return self.y.shape[0]


@dataclass
class HeterotopicDataset:
    X: jnp.DeviceArray
    y: jnp.DeviceArray


@dataclass
class IsotopicDataset:
    X: jnp.DeviceArray
    y: jnp.DeviceArray
