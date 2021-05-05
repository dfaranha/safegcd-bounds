Require Import ZArith.
Require Import hddivsteps_base.

Definition example:= (N.iter 1325 (processDivstep 0x553d402ae2d5e4bc392bfdd23348b6a26137d053aa03007122696c9fb7a0951f7514f41a6c7018715e1d4944218ea2d852064509ac3491bf86cbb9ba813c985df1e5cef00ad97efb) state0).
Extraction "hddivsteps1325.ml" example.

Lemma example1325 : ZMap.Empty (N.iter 1325 (processDivstep 0x553d402ae2d5e4bc392bfdd23348b6a26137d053aa03007122696c9fb7a0951f7514f41a6c7018715e1d4944218ea2d852064509ac3491bf86cbb9ba813c985df1e5cef00ad97efb) state0).
Proof.
apply ZMap.is_empty_2.
vm_cast_no_check (refl_equal true).
Time Qed.

