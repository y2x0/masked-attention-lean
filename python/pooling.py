"""Masked attention pooling over rows with shared, instance-local scores."""
import jax.numpy as jnp


def masked_pool(values, logits, mask):
    """Pool [n, d] values using [n] logits and a Boolean validity mask."""
    safe_logits = jnp.where(mask, logits, 0.0)
    safe_values = jnp.where(mask[:, None], values, 0.0)
    weights = jnp.where(mask, jnp.exp(safe_logits), 0.0)
    total = jnp.sum(weights)
    denominator = jnp.where(total > 0.0, total, 1.0)
    return jnp.sum(weights[:, None] * safe_values, axis=0) / denominator


def stable_masked_pool(values, logits, mask):
    """Numerically stabilized pooling; return zero for an all-masked bag."""
    # The invalid rows must not set the stabilizing maximum.
    maximum = jnp.max(jnp.where(mask, logits, -jnp.inf), initial=-jnp.inf)
    maximum = jnp.where(jnp.any(mask), maximum, 0.0)
    return masked_pool(values, logits - maximum, mask)


def attention_pool(values, scorer, length):
    """Apply a shared linear scorer and a prefix mask to an [n, d] bag."""
    mask = jnp.arange(values.shape[0]) < length
    safe_values = jnp.where(mask[:, None], values, 0.0)
    return stable_masked_pool(values, safe_values @ scorer, mask)
