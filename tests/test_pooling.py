import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pooling import masked_pool, stable_masked_pool, attention_pool


@pytest.mark.parametrize("fn", [masked_pool, stable_masked_pool])
def test_padding_and_permutation(fn):
    rng = np.random.default_rng(14)
    for n, d in [(1, 1), (3, 2), (17, 8)]:
        x = jnp.array(rng.normal(size=(n, d)), dtype=jnp.float32)
        scores = jnp.array(rng.normal(size=n), dtype=jnp.float32)
        mask = jnp.array(rng.random(n) > 0.4).at[0].set(True)
        expected = fn(x, scores, mask)
        real = np.asarray(mask)
        weights = np.exp(np.asarray(scores)[real].astype(np.float64))
        reference = (weights[:, None] * np.asarray(x)[real]).sum(0) / weights.sum()
        np.testing.assert_allclose(expected, reference, rtol=2e-5, atol=2e-6)
        for p in [0, 1, 11]:
            padded_x = jnp.concatenate([x, jnp.full((p, d), jnp.nan)])
            padded_scores = jnp.concatenate([scores, jnp.full(p, jnp.inf)])
            padded_mask = jnp.concatenate([mask, jnp.zeros(p, dtype=bool)])
            np.testing.assert_allclose(jax.jit(fn)(padded_x, padded_scores, padded_mask),
                                       expected, rtol=2e-5, atol=2e-6)
        permutation = rng.permutation(n)
        np.testing.assert_allclose(fn(x[permutation], scores[permutation], mask[permutation]),
                                   expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("fn", [masked_pool, stable_masked_pool])
def test_all_masked(fn):
    result = fn(jnp.full((4, 2), jnp.nan), jnp.full(4, jnp.inf), jnp.zeros(4, dtype=bool))
    np.testing.assert_array_equal(result, jnp.zeros(2))


def test_stability_and_length_wrapper():
    x = jnp.array([[1., 2.], [3., 4.], [jnp.nan, jnp.nan]])
    scorer = jnp.array([0.2, -0.1])
    expected = attention_pool(x[:2], scorer, 2)
    np.testing.assert_allclose(attention_pool(x, scorer, 2), expected)
    mask = jnp.array([True, True, False])
    logits = jnp.array([1000., 1001., jnp.inf])
    result = stable_masked_pool(x, logits, mask)
    weights = np.exp(np.array([-1., 0.]))
    np.testing.assert_allclose(result, (weights[:, None] * np.asarray(x[:2])).sum(0) / weights.sum(),
                               rtol=2e-5)


def test_empty_core():
    np.testing.assert_array_equal(masked_pool(jnp.zeros((0, 2)), jnp.zeros(0),
                                             jnp.zeros(0, dtype=bool)), jnp.zeros(2))
