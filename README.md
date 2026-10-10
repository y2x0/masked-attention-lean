# Masked attention pooling in Lean

A MIL example built on [srush/jax-lean](https://github.com/srush/jax-lean).

Checked locally: all Lean proofs and translation certificates compile, all seven
theorem axiom audits pass, and six JAX tests pass. See `VALIDATION.md` and `validation.log`.

For values `x[i, j]`, instance-local logits `s[i]`, and a Boolean mask `m[i]`,

```
w[i] = exp(s[i]) if m[i] else 0
z = sum(w)
pool[j] = sum(w[i] * x[i, j] for valid i) / (z if z > 0 else 1)
```

Invalid rows have zero mass before normalization. Zeroing features alone is
insufficient: a padded row with logit zero would still add `exp(0) = 1` to `z`.
An all-masked or empty bag returns the zero vector.

## Run locally

```sh
git clone --recurse-submodules https://github.com/y2x0/masked-attention-lean.git
cd masked-attention-lean
./scripts/bootstrap.sh
uv run pytest -q
uv run python python/demo.py
uv run python python/generate.py --check
~/.elan/bin/lake build
```

Prerequisites: [uv](https://docs.astral.sh/uv/) and
[elan](https://github.com/leanprover/elan).

CPU JAX runs on macOS; no CUDA installation or cluster connection is needed.
Python dependencies stay in `.venv`. Lean's toolchain manager lives in `~/.elan`.
Lean and JAX versions are pinned to match the upstream library.
For a single command that repeats all checks, run `./scripts/check.sh`.
`scripts/bootstrap.sh` resolves pinned Lean dependencies and their compiled cache.

## Read the project

- `python/pooling.py`: JAX kernel, stabilized runtime variant, and length wrapper.
- `python/generate.py`: trace the kernel at `[3, 2]` and `[5, 2]`, then use
  `certify_module` to emit Lean functions and translation certificates.
- `MaskedAttention/Basic.lean`: proofs for arbitrary finite row sets and feature dimensions.
- `MaskedAttention/Generated3.lean`, `Generated5.lean`: reproducible transpiler output.
- `MaskedAttention/Connection.lean`: connect generated code to the generic specification
  and transfer invariance to the imported mathematical Jaxpr.
- `tests/test_pooling.py`: compare with an independent NumPy formula and test
  permutation, extra padding, all-masked bags, nonfinite padding, and large logits.

## What the proofs mean

`pool_reindex` proves that any bijection of rows leaves pooling unchanged when
values, logits, and masks move together. Both numerator and denominator are
finite sums; a bijection only renames their summation indices.

`pool_padding` proves that appending any finite number of masked rows leaves
pooling unchanged. The padded values and logits can be arbitrary real numbers.
The additional terms are exactly zero in both sums.

`pool_masked_congr` proves that changing values or logits at invalid positions
has no effect. `pool_all_masked` proves the zero-output convention.

`attention_permutation` covers any shared instance-local scorer, including
an MLP or gated attention score. Its logit moves with its instance. A positional
scorer or an order-dependent encoder needs a separate argument.

The generic proofs use arbitrary finite types; padding is a disjoint union
of the original row set and a new row set. The generated examples specialize
this to concrete tensor shapes. No bags or values are enumerated in the generic proof.

## Exact scope

Lean proves identities in ideal real arithmetic. The generated certificates
connect readable functions to the upstream library's imported Jaxpr evaluator.
The Python importer and JAX tracing remain trusted. This is not a proof of
IEEE-754 rounding, XLA execution, or JAX itself.

The certified kernel uses unshifted exponentials so its proof is short and
transparent. `stable_masked_pool` and `attention_pool` provide a maximum-shifted
runtime variant and are tested numerically, but their own Jaxprs are not certified
in this first version. Use finite real values/logits at valid positions; the
runtime variant also suppresses NaN/Inf at invalid positions. Reordering
floating-point sums can change the last few bits, so runtime comparisons use tolerances.

This proves structural pooling properties, not model accuracy or clinical validity.

Upstream reference: `srush/jax-lean` revision
`570aed69049f8e2931df9f55175313a340bcb456`, including
`examples/transformer/proofs/TransformerProofs.lean` and
`SUPPORTED_JAXPR.md` for the existing proof pattern and semantic boundary.

## Typeset guide

Read [the five-page PDF](output/pdf/masked-attention-pooling.pdf), or edit
[the LaTeX source](output/pdf/masked-attention-pooling.tex). Rebuild locally with:

```sh
latexmk -pdf -outdir=output/pdf output/pdf/masked-attention-pooling.tex
```
