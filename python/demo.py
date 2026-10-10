"""Run with: uv run python python/demo.py."""
import jax.numpy as jnp
from pooling import attention_pool

values = jnp.array([[1., 2.], [3., 4.], [2., -1.]])
scorer = jnp.array([0.2, -0.1])
padded = jnp.concatenate([values, jnp.full((2, 2), jnp.nan)])
permutation = jnp.array([2, 0, 1])
print("original: ", attention_pool(values, scorer, 3))
print("padded:   ", attention_pool(padded, scorer, 3))
print("permuted: ", attention_pool(values[permutation], scorer, 3))
