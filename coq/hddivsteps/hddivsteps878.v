Require Import ZArith.
Require Import hddivsteps_base.

Definition example:=(N.iter 878 (processDivstep 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab) state0).
Extraction "hddivsteps878.ml" example.

Lemma example878 : ZMap.Empty (N.iter 878 (processDivstep 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab) state0).
Proof.
apply ZMap.is_empty_2.
vm_cast_no_check (refl_equal true).
Time Qed.

