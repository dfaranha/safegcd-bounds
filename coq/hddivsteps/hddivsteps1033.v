Require Import ZArith.
Require Import hddivsteps_base.

Definition example:=(N.iter 1033 (processDivstep 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff) state0).
Extraction "hddivsteps1033.ml" example.

Lemma example1033 : ZMap.Empty (N.iter 1033 (processDivstep 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff) state0).
Proof.
apply ZMap.is_empty_2.
vm_cast_no_check (refl_equal true).
Time Qed.

