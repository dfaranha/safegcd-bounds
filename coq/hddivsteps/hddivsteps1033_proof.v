Require Import ZArith.
Require Import hddivsteps1033.
Require Import hddivsteps_def.
Require Import hddivsteps_theory.

Theorem hddivsteps1033_gcd : forall f g,
  Z.Odd f ->
  (f <= 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff)%Z ->
  (0 <= g <= f)%Z -> 
  Znumtheory.Zis_gcd f g
    (hddivsteps.f (N.iter 1033 hddivsteps.step (hddivsteps.init f g))).
Proof.
intros f g Hf HM Hg.
eapply processDivstep_gcd; try assumption; try apply HM.
apply example1033.
Qed.

Check hddivsteps1033_gcd.
Print Assumptions hddivsteps1033_gcd.

Theorem hddivsteps1033_inverse : forall f g,
  Z.Odd f ->
  (f <= 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff)%Z ->
  (0 <= g <= f)%Z -> 
  Znumtheory.rel_prime f g ->
  let st := N.iter 1033 hddivsteps.step (hddivsteps.init f g) in
  Z.abs (hddivsteps.f st) = 1%Z /\
   eqm f ((hddivsteps.d st * hddivsteps.f st) * g) 1.
Proof.
intros f g Hf HM Hg Hprime.
eapply processDivstep_inverse; try assumption; try apply HM.
apply example1033.
Qed.

Check hddivsteps1033_inverse.
Print Assumptions hddivsteps1033_inverse.

Theorem hddivsteps1033_prime_inverse : forall f g,
  Z.Odd f ->
  Znumtheory.prime f ->
  (g < f <= 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffff)%Z ->
  let st := N.iter 1033 hddivsteps.step (hddivsteps.init f g) in
  (g = 0 -> hddivsteps.d st = 0)%Z /\
  (0 < g -> Z.abs (hddivsteps.f st) = 1 /\
            eqm f ((hddivsteps.d st * hddivsteps.f st) * g) 1)%Z.
Proof.
intros f g Hf Hprime HM Hg.
eapply processDivstep_prime_inverse; try assumption; try apply HM.
apply example1033.
Qed.

Check hddivsteps1033_prime_inverse.
Print Assumptions hddivsteps1033_prime_inverse.
