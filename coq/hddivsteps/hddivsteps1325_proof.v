Require Import ZArith.
Require Import hddivsteps1325.
Require Import hddivsteps_def.
Require Import hddivsteps_theory.

Theorem hddivsteps1325_gcd : forall f g,
  Z.Odd f ->
  (f <= 0x553d402ae2d5e4bc392bfdd23348b6a26137d053aa03007122696c9fb7a0951f7514f41a6c7018715e1d4944218ea2d852064509ac3491bf86cbb9ba813c985df1e5cef00ad97efb)%Z ->
  (0 <= g <= f)%Z -> 
  Znumtheory.Zis_gcd f g
    (hddivsteps.f (N.iter 1325 hddivsteps.step (hddivsteps.init f g))).
Proof.
intros f g Hf HM Hg.
eapply processDivstep_gcd; try assumption; try apply HM.
apply example1325.
Qed.

Check hddivsteps1325_gcd.
Print Assumptions hddivsteps1325_gcd.

Theorem hddivsteps1325_inverse : forall f g,
  Z.Odd f ->
  (f <= 0x553d402ae2d5e4bc392bfdd23348b6a26137d053aa03007122696c9fb7a0951f7514f41a6c7018715e1d4944218ea2d852064509ac3491bf86cbb9ba813c985df1e5cef00ad97efb)%Z ->
  (0 <= g <= f)%Z -> 
  Znumtheory.rel_prime f g ->
  let st := N.iter 1325 hddivsteps.step (hddivsteps.init f g) in
  Z.abs (hddivsteps.f st) = 1%Z /\
   eqm f ((hddivsteps.d st * hddivsteps.f st) * g) 1.
Proof.
intros f g Hf HM Hg Hprime.
eapply processDivstep_inverse; try assumption; try apply HM.
apply example1325.
Qed.

Check hddivsteps1325_inverse.
Print Assumptions hddivsteps1325_inverse.

Theorem hddivsteps1325_prime_inverse : forall f g,
  Z.Odd f ->
  Znumtheory.prime f ->
  (g < f <= 0x553d402ae2d5e4bc392bfdd23348b6a26137d053aa03007122696c9fb7a0951f7514f41a6c7018715e1d4944218ea2d852064509ac3491bf86cbb9ba813c985df1e5cef00ad97efb)%Z ->
  let st := N.iter 1325 hddivsteps.step (hddivsteps.init f g) in
  (g = 0 -> hddivsteps.d st = 0)%Z /\
  (0 < g -> Z.abs (hddivsteps.f st) = 1 /\
            eqm f ((hddivsteps.d st * hddivsteps.f st) * g) 1)%Z.
Proof.
intros f g Hf Hprime HM Hg.
eapply processDivstep_prime_inverse; try assumption; try apply HM.
apply example1325.
Qed.

Check hddivsteps1325_prime_inverse.
Print Assumptions hddivsteps1325_prime_inverse.
