"""Reproduce fixed-shape Lean functions and their translation certificates."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxlean import certify_module
from pooling import masked_pool

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for n in (3, 5):
    program = jax.make_jaxpr(masked_pool)(
        jax.ShapeDtypeStruct((n, 2), jnp.float32),
        jax.ShapeDtypeStruct((n,), jnp.float32),
        jax.ShapeDtypeStruct((n,), jnp.bool_),
    )
    source = certify_module(program, name="pool", namespace=f"MaskedAttention.Generated{n}")
    source = source.replace("import JaxLean.Core.RealOps\n",
                            "import JaxLean.Core.RealOps\n\nset_option linter.style.nameCheck false\n", 1)
    path = root / "MaskedAttention" / f"Generated{n}.lean"
    if args.check:
        if not path.exists() or path.read_text() != source:
            raise SystemExit(f"stale generated file: {path}")
    else:
        path.write_text(source)
    print(path.name)
