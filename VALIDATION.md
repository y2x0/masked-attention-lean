# Validation record

Date: 2026-10-09, America/New_York.
Machine: macOS, Apple silicon; JAX CPU device.

- JAX 0.8.0 and jaxlib 0.8.0 installed inside `.venv`.
- `uv run pytest -q`: 6 passed.
- `uv run python python/generate.py --check`: both generated files match.
- Demo: original, padded, and permuted output all `[2.057208, 1.3531581]`.
- `./scripts/check.sh`: exit 0, including successful `lake build` and all 7 theorem axiom audits.
- All audited theorems depend only on `propext`, `Classical.choice`, and `Quot.sound`; no `sorryAx` or custom axioms.
- Lean 4.33.0, native Apple silicon, official release SHA-256 verified.
- Complete final check output: `validation.log`.

The JAX tests exercise eager/JIT execution, an independent NumPy reference,
multiple bag sizes, permutations, 0/1/11 added rows, NaN/Inf in invalid rows,
all-masked and empty bags, length masking, and stable evaluation of logits near 1000.
These numerical tests are evidence of execution only. They do not replace the
Lean proofs or prove equality of floating-point computations.

At the validation date above, the repository was local and no upload had occurred.
On 2026-10-10, publication was authorized to `y2x0/masked-attention-lean` as a public repository. Upstream modifications and pull requests remain outside the authorized scope.
