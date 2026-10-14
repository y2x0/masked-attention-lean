# masked-attention-lean

Lean proofs that masked attention pooling, as used in multiple instance learning,
doesn't care how much padding you add or what order the instances are in. The
proofs are connected to the actual JAX kernel through
[jax-lean](https://github.com/srush/jax-lean).

Paper: [arXiv:2610.17132](https://arxiv.org/abs/2610.17132)

## The problem

When you batch MIL bags you pad them to a common length and pass a mask. The
easy mistake is to zero out the padded features and stop there. A padded row
with logit 0 still puts `exp(0) = 1` into the softmax denominator, so the output
changes with the amount of padding:

```
real instances: 2 and 6, equal logits   -> should pool to 4
plus p zero-feature padded rows         -> pools to 8 / (2 + p)
```

Masking the logits with `-inf` fixes that, but you get NaN for an empty bag or
for NaN padding. A large negative constant like `-1e9` only works because of
float underflow, and an empty bag returns the average of its padding.

The kernel here takes padded rows out of both sums and guards the denominator:

```
w[i] = exp(s[i]) if m[i] else 0
z    = sum(w)
out  = sum(w[i] * x[i] for valid i) / (z if z > 0 else 1)
```

An empty or all-masked bag returns zero.

## What's proved

All in `MaskedAttention/Basic.lean`, for any finite row type and any feature
dimension:

| theorem | says |
| --- | --- |
| `pool_reindex` | any bijection of rows leaves the output unchanged, as long as values, logits and mask move together |
| `pool_padding` | appending any number of masked rows, with arbitrary real values and logits, leaves the output unchanged |
| `pool_masked_congr` | values and logits at masked positions don't matter |
| `pool_all_masked` | an all-masked bag gives zero |
| `attention_permutation` | the permutation result holds for any shared per-instance scorer (linear, MLP, gated attention) |

The proofs are short. Permutation is just reindexing two finite sums, and
padding splits each sum over a disjoint union where the new terms are all zero.
`WALKTHROUGH.md` goes through them slowly if you haven't read Lean before.

## Connection to the JAX code

`python/generate.py` traces `masked_pool` at shapes `[3, 2]` and `[5, 2]` and
uses jax-lean to emit two things per shape: a readable Lean function and an
imported copy of the Jaxpr, along with a certificate that the two agree on
every real input. `MaskedAttention/Connection.lean` then shows each generated
function equals the generic `pool`, and transfers the invariance results back
to the imported Jaxpr:

- `certified_permutation`: permuting the rows of the `[3, 2]` program's input
  doesn't change its output.
- `certified_padding`: the `[5, 2]` program on a bag with two masked rows
  appended gives the same output as the `[3, 2]` program on the original bag.

## Running it

You need [uv](https://docs.astral.sh/uv/) and
[elan](https://github.com/leanprover/elan).

```sh
git clone --recurse-submodules https://github.com/y2x0/masked-attention-lean.git
cd masked-attention-lean
./scripts/bootstrap.sh     # submodule, python env, mathlib cache
./scripts/check.sh         # everything below in one go
```

or one step at a time:

```sh
uv run pytest -q
uv run python python/demo.py
uv run python python/generate.py --check
~/.elan/bin/lake build
~/.elan/bin/lake env lean ProofAudit.lean
```

JAX runs on CPU, so this works on a laptop. Versions are pinned to match
jax-lean: Lean 4.33.0, JAX 0.8.0, and jax-lean at
`570aed69049f8e2931df9f55175313a340bcb456`.

The last run is in `validation.log`: 6 tests pass, the build succeeds, and all
seven audited theorems use only `propext`, `Classical.choice` and `Quot.sound`
(no `sorry`).

## Layout

```
python/pooling.py            the kernel, a max-shifted version, and a length-mask wrapper
python/generate.py           traces the kernel and writes the Generated*.lean files
MaskedAttention/Basic.lean   definitions and the general proofs
MaskedAttention/Connection.lean
                             links the generated code to the general proofs
MaskedAttention/Generated*.lean
                             generated, don't edit by hand
tests/test_pooling.py        numerical checks against a NumPy reference
ProofAudit.lean              prints the axioms each main theorem uses
```

## Limits

- The proofs are over the reals, not floating point. They say nothing about
  rounding, XLA, or JAX itself, and the tests compare with tolerances for that
  reason.
- JAX tracing and jax-lean's Python importer are trusted. The readable
  generated functions aren't: they're checked against the imported Jaxpr.
- Only the `[3, 2]` and `[5, 2]` traces are certified. The general theorems
  cover every size, but a Jaxpr has fixed shapes, so another shape means
  rerunning `generate.py`.
- `stable_masked_pool` (subtract the max valid logit first) is what you'd
  actually run. It's tested but its Jaxpr isn't certified yet, because the
  masked max starts from `-inf`, which jax-lean's real-number model doesn't
  cover.
- The scorer result needs the score to depend on one instance at a time.
  Anything that mixes instances before pooling, like a transformer over the
  bag or positional encodings, needs its own proof.

None of this says anything about model accuracy. It's about the pooling layer
behaving the way you'd assume it does.

## Citing

```bibtex
@misc{sadek2026maskedattention,
  title         = {Masked Attention Pooling Is Invariant to Padding and Permutation: A Formal Proof in Lean},
  author        = {Sadek, Yusuf Khaled and Yilmaz, Enes},
  year          = {2026},
  eprint        = {2610.17132},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
