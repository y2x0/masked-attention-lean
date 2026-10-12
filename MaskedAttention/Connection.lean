import MaskedAttention.Basic
import MaskedAttention.Generated3
import MaskedAttention.Generated5
import Mathlib.Logic.Equiv.Fin.Basic

open JaxLean
open scoped BigOperators

namespace MaskedAttention

/-- The generated tensor function implements the size-generic specification. -/
theorem generated3_spec (x : Tensor ℝ [3, 2]) (s : Tensor ℝ [3]) (m : Tensor Bool [3]) :
    Generated3.pool x s m = fun j =>
      pool (fun i j => x (i, j, ())) (fun i => s (i, ())) (fun i => m (i, ())) j.1 := by
  funext j
  simp +contextual [Generated3.pool, Generated3.fn__where, Generated3.fn__where_2,
    Generated3.fn__where_3, pool, mass, Tensor.map, Tensor.map₂,
    Tensor.reindex, Tensor.scalar, Tensor.sumFirst, RealOps.exp]

theorem generated5_spec (x : Tensor ℝ [5, 2]) (s : Tensor ℝ [5]) (m : Tensor Bool [5]) :
    Generated5.pool x s m = fun j =>
      pool (fun i j => x (i, j, ())) (fun i => s (i, ())) (fun i => m (i, ())) j.1 := by
  funext j
  simp +contextual [Generated5.pool, Generated5.fn__where, Generated5.fn__where_2,
    Generated5.fn__where_3, pool, mass, Tensor.map, Tensor.map₂,
    Tensor.reindex, Tensor.scalar, Tensor.sumFirst, RealOps.exp]

/-- Permutation invariance of the function actually emitted from JAX. -/
theorem generated_permutation (e : Fin 3 ≃ Fin 3)
    (x : Tensor ℝ [3, 2]) (s : Tensor ℝ [3]) (m : Tensor Bool [3]) :
    Generated3.pool (fun i => x (e i.1, i.2)) (fun i => s (e i.1, ()))
      (fun i => m (e i.1, ())) = Generated3.pool x s m := by
  rw [generated3_spec, generated3_spec]
  funext j
  exact congrFun (pool_reindex e (fun i j => x (i, j, ()))
    (fun i => s (i, ())) (fun i => m (i, ()))) j.1

/-- Append two rows to a three-row tensor, retaining ordinary prefix order. -/
def appendRows {R : Type} {shape : List ℕ} (x : Tensor R (3 :: shape))
    (p : Tensor R (2 :: shape)) : Tensor R (5 :: shape) :=
  fun i => Sum.elim (fun r => x (r, i.2)) (fun r => p (r, i.2))
    ((finSumFinEquiv : Fin 3 ⊕ Fin 2 ≃ Fin 5).symm i.1)

/-- Padding invariance across two separately traced JAX tensor shapes. -/
theorem generated_padding (x : Tensor ℝ [3, 2]) (s : Tensor ℝ [3])
    (m : Tensor Bool [3]) (p : Tensor ℝ [2, 2]) (ps : Tensor ℝ [2]) :
    Generated5.pool (appendRows x p) (appendRows s ps)
      (appendRows m (fun _ => false)) = Generated3.pool x s m := by
  rw [generated5_spec, generated3_spec]
  funext j
  let values := Sum.elim (fun i j => x (i, j, ())) (fun i j => p (i, j, ()))
  let logits := Sum.elim (fun i => s (i, ())) (fun i => ps (i, ()))
  let mask := Sum.elim (fun i => m (i, ())) (fun _ : Fin 2 => false)
  let e : Fin 5 ≃ Fin 3 ⊕ Fin 2 := (finSumFinEquiv : Fin 3 ⊕ Fin 2 ≃ Fin 5).symm
  change pool (values ∘ e) (logits ∘ e) (mask ∘ e) j.1 = _
  rw [pool_reindex e values logits mask]
  exact congrFun (pool_padding (fun i j => x (i, j, ())) (fun i => s (i, ()))
    (fun i => m (i, ())) (fun i j => p (i, j, ())) (fun i => ps (i, ()))) j.1

/-- Transfer permutation invariance to the imported Jaxpr's real semantics. -/
theorem certified_permutation (e : Fin 3 ≃ Fin 3)
    (x : Tensor ℝ [3, 2]) (s : Tensor ℝ [3]) (m : Tensor Bool [3]) :
    Jaxpr.Program.eval (.cons (fun i : Index [3, 2] => x (e i.1, i.2))
      (.cons (fun i : Index [3] => s (e i.1, ())) (.cons (fun i : Index [3] => m (e i.1, ())) .nil)))
      Generated3.pool_ir =
    Jaxpr.Program.eval (.cons x (.cons s (.cons m .nil))) Generated3.pool_ir := by
  rw [Generated3.pool_translation_correct, Generated3.pool_translation_correct]
  exact generated_permutation e x s m

/-- Transfer padding invariance to both separately imported Jaxprs. -/
theorem certified_padding (x : Tensor ℝ [3, 2]) (s : Tensor ℝ [3])
    (m : Tensor Bool [3]) (p : Tensor ℝ [2, 2]) (ps : Tensor ℝ [2]) :
    Jaxpr.Program.eval (.cons (appendRows x p) (.cons (appendRows s ps)
      (.cons (appendRows m (fun _ => false)) .nil))) Generated5.pool_ir =
    Jaxpr.Program.eval (.cons x (.cons s (.cons m .nil))) Generated3.pool_ir := by
  rw [Generated5.pool_translation_correct, Generated3.pool_translation_correct]
  exact generated_padding x s m p ps

end MaskedAttention
