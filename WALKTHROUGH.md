# Reading the proofs

## The idea with two instances

Take a bag with two instances, values `a` and `b`, logits `u` and `v`. For one
output feature the pooled value is

```
(exp(u) * a + exp(v) * b) / (exp(u) + exp(v))
```

Add a padded row. Its mask is false, so it adds 0 to the top and 0 to the
bottom, whatever its value and logit are. Swap the two instances and you swap
two terms in each sum, which doesn't change anything. The Lean proofs make the
same two arguments for any finite bag and every feature.

## Lean notation you'll run into

- `Fin d` is a feature index, one of `0, ..., d-1`.
- `ι → Fin d → ℝ` is a bag: each row index gives a real feature vector.
- `[Fintype ι]` says there are finitely many rows, so we can sum over them.
- `e : κ ≃ ι` is a bijection between two row index types.
- `values ∘ e` is `fun i => values (e i)`, i.e. the rows reordered by `e`.
- `∑ i, ...` sums over all rows.
- `funext j` proves two vectors equal by proving them equal at each feature `j`.

## Permutation (`pool_reindex`)

`e.sum_comp` says that summing `g (e k)` over `k` is the same as summing
`g i` over `i` when `e` is a bijection. The proof uses it twice, once for the
denominator and once for the numerator, and then both sides are literally the
same expression.

## Padding (`pool_padding`)

The padded bag is indexed by `Sum ι κ`, the original rows plus the new ones.
`Sum.elim values padding` picks the original data on the left and the padding
on the right, and the mask on the right is always false.
`Fintype.sum_sum_type` splits each sum into the original rows plus the new
rows. Every new term is zero, so `simp` finishes it.

Nothing is assumed about the padded values or logits, only that they're
masked. Masked rows that were already in the original bag are fine too.

## From Python to the theorems

1. JAX traces `masked_pool` at a fixed shape.
2. `jaxlean.certify_module` writes a readable Lean function and a typed copy of
   the Jaxpr, with a certificate that they agree on every real input.
3. `generated3_spec` and `generated5_spec` show each generated function equals
   the general `pool`. This is where the mask handling, the sums and the
   guarded denominator actually get checked.
4. The general theorems apply, and `certified_permutation` and
   `certified_padding` restate them about the Jaxpr itself.

The tracer and importer are trusted, and floating point is out of scope. The
identities are exact over the reals; the JAX tests use tolerances.

## Why the scorer has to be per-instance

If the logit is `score(x[i])` for some shared function `score`, the logit moves
with its row when you shuffle, which is exactly what `pool_reindex` needs.
`attention_permutation` proves this for any `score`, with no assumptions about
its form. A scorer that looks at position, or at other rows, isn't covered.

## Where to start

```sh
uv run python python/demo.py
~/.elan/bin/lake build
~/.elan/bin/lake env lean ProofAudit.lean
```

Try changing the padding or the permutation in the demo. Then read
`Basic.lean`, then `Connection.lean`. You don't need to read the generated
files to follow the argument.
