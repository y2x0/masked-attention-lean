import JaxLean.Core.Tensor
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

open scoped BigOperators

namespace MaskedAttention

/-- An invalid row contributes exactly zero attention mass. -/
noncomputable def mass {ι : Type*} (logits : ι → ℝ) (mask : ι → Bool) (i : ι) : ℝ :=
  if mask i then Real.exp (logits i) else 0

/-- Weighted pooling, with a zero output for a bag with no valid rows. -/
noncomputable def pool {ι : Type*} [Fintype ι] {d : ℕ}
    (values : ι → Fin d → ℝ) (logits : ι → ℝ) (mask : ι → Bool) : Fin d → ℝ :=
  let total := ∑ i, mass logits mask i
  fun j => (∑ i, mass logits mask i * (if mask i then values i j else 0)) /
    (if total > 0 then total else 1)

/-- Renaming rows preserves pooling when values, scores and masks move together. -/
theorem pool_reindex {ι κ : Type*} [Fintype ι] [Fintype κ] {d : ℕ}
    (e : κ ≃ ι) (values : ι → Fin d → ℝ) (logits : ι → ℝ) (mask : ι → Bool) :
    pool (values ∘ e) (logits ∘ e) (mask ∘ e) = pool values logits mask := by
  funext j
  unfold pool mass
  simp only [Function.comp_apply]
  rw [e.sum_comp (fun i => if mask i then Real.exp (logits i) else 0)]
  rw [e.sum_comp (fun i => (if mask i then Real.exp (logits i) else 0) *
    (if mask i then values i j else 0))]

/-- Append any number of arbitrary rows marked invalid. -/
theorem pool_padding {ι κ : Type*} [Fintype ι] [Fintype κ] {d : ℕ}
    (values : ι → Fin d → ℝ) (logits : ι → ℝ) (mask : ι → Bool)
    (padding : κ → Fin d → ℝ) (paddingLogits : κ → ℝ) :
    pool (Sum.elim values padding) (Sum.elim logits paddingLogits)
      (Sum.elim mask (fun _ => false)) = pool values logits mask := by
  funext j
  simp [pool, mass, Fintype.sum_sum_type]
  rfl

/-- Values and scores at invalid positions do not affect the result. -/
theorem pool_masked_congr {ι : Type*} [Fintype ι] {d : ℕ}
    (values values' : ι → Fin d → ℝ) (logits logits' : ι → ℝ) (mask : ι → Bool)
    (hv : ∀ i, mask i = true → values i = values' i)
    (hl : ∀ i, mask i = true → logits i = logits' i) :
    pool values logits mask = pool values' logits' mask := by
  have hm : mass logits mask = mass logits' mask := by
    funext i
    cases h : mask i with
    | false => simp [mass, h]
    | true => simp [mass, h, hl i h]
  have hn : ∀ j, (∑ i, mass logits mask i * (if mask i then values i j else 0)) =
      ∑ i, mass logits' mask i * (if mask i then values' i j else 0) := by
    intro j
    apply Finset.sum_congr rfl
    intro i _
    cases h : mask i with
    | false => simp [mass, h]
    | true => simp [mass, h, hv i h, hl i h]
  funext j
  unfold pool
  rw [hn j, hm]

/-- An empty or all-masked bag has zero output. -/
theorem pool_all_masked {ι : Type*} [Fintype ι] {d : ℕ}
    (values : ι → Fin d → ℝ) (logits : ι → ℝ) :
    pool values logits (fun _ => false) = fun _ => 0 := by
  funext j
  simp [pool, mass]

/-- A shared instance-local scoring function moves with its instances. -/
theorem attention_permutation {ι : Type*} [Fintype ι] {d : ℕ}
    (e : ι ≃ ι) (values : ι → Fin d → ℝ) (score : (Fin d → ℝ) → ℝ)
    (mask : ι → Bool) :
    pool (values ∘ e) (fun i => score (values (e i))) (mask ∘ e) =
      pool values (fun i => score (values i)) mask :=
  pool_reindex e values (fun i => score (values i)) mask

end MaskedAttention
