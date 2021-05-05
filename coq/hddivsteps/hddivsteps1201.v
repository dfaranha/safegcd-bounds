Require Import ZArith.
Require Import hddivsteps_base.

Definition example:= (N.iter 1201 (processDivstep 0x1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff) state0).
Extraction "hddivsteps1201.ml" example.

Lemma example1201 : ZMap.Empty (N.iter 1201 (processDivstep 0x1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff) state0).
Proof.
apply ZMap.is_empty_2.
vm_cast_no_check (refl_equal true).
Time Qed.

