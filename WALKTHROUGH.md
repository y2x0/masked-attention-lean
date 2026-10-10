# Understanding the proof

Start with a bag of two real instances, values `a, b`, and logits `u, v`.
For one output feature the result is

```
(exp(u) * a + exp(v) * b) / (exp(u) + exp(v)).
```

Appending a padded instance adds zero to both sums, whatever its value or logit.
Swapping the two instances swaps the two terms in each sum; addition is commutative.
The Lean proofs repeat these arguments for every finite bag size and every feature.

## Reading the Lean notation

`Fin d` is a valid feature index: one of `0, ..., d-1`.
`ι → Fin d → ℝ` says that each row has a real-valued feature vector.
`[Fintype ι]` says that the set of rows is finite, so Lean can sum over it.
`e : κ ≃ ι` is a bijection: it reindexes every row exactly once.
`values ∘ e` means `fun i => values (e i)`.
`∑ i, ...` is a finite sum over all valid row indices.
`funext j` says: to prove two output vectors equal, prove their feature `j` equal.

`pool_reindex` uses `e.sum_comp` twice: once on the normalizing mass and once
on the feature numerator. After those substitutions, the two sides are identical.

For padding, `Sum ι κ` is the disjoint union of real and added row positions.
`Sum.elim values padding` chooses the right data for each side. The new side's
mask is false. `Fintype.sum_sum_type` splits each sum into original rows plus
new rows. Every term in the new sum is zero, so `simp` closes the equality.

The theorem handles arbitrary invalid rows already present in the original bag.
The added rows need not have zero features or zero logits. Only the mask matters.

## From Python to the theorem

1. JAX traces `masked_pool` for a chosen fixed input shape.
2. `jaxlean.certify_module` writes a readable Lean tensor function and an imported,
   typed mathematical Jaxpr, including helper calls for `where`.
3. Lean checks generated translation theorems connecting that function to the
   imported mathematical Jaxpr evaluator for every real input of that shape.
4. `generated3_spec` and `generated5_spec` identify those generated functions with
   the size-generic `pool` definition. This checks the mask branches, sums, and denominator.
5. The generic invariance theorems apply. The final `certified_*` theorems carry
   the result back across the translation certificates to the mathematical Jaxpr.

JAX's Python tracer/importer remain trusted, and machine rounding is outside the
model. The mathematical invariance is exact; the JAX tests allow floating-point tolerance.

## Why instance-local scoring matters

A shared scorer `score(x[i])` moves with each row. It can be linear, nonlinear,
or gated, provided it does not depend on the row's position or other rows' order.
`attention_permutation` proves the result for any such function; no particular
weights or activation are assumed.

A contextual encoder can also work if separately proved permutation equivariant
and insensitive to padding. These pooling proofs alone do not establish that
property for an entire encoder or training pipeline.

## What to run when learning

```sh
uv run python python/demo.py
~/.elan/bin/lake build
~/.elan/bin/lake env lean ProofAudit.lean
```

Change the demo's padding values or permutation and run it again. Then read
`Basic.lean` before `Connection.lean`. The generated files are implementation
detail; you do not need to start by reading their typed SSA environments.
