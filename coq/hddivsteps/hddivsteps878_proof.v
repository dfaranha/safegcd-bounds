Require Import ZArith.
Require Import hddivsteps878.
Require Import hddivsteps_def.
Require Import hddivsteps_theory.

Theorem hddivsteps878_gcd : forall f g,
  Z.Odd f ->
  (f <= 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab)%Z ->
  (0 <= g <= f)%Z -> 
  Znumtheory.Zis_gcd f g
    (hddivsteps.f (N.iter 878 hddivsteps.step (hddivsteps.init f g))).
Proof.
intros f g Hf HM Hg.
eapply processDivstep_gcd; try assumption; try apply HM.
apply example878.
Qed.

Check hddivsteps878_gcd.
Print Assumptions hddivsteps878_gcd.

Theorem hddivsteps878 : forall f g,
  Z.Odd f ->
  (f <= 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab)%Z ->
  (0 <= g <= f)%Z -> 
  Znumtheory.rel_prime f g ->
  let st := N.iter 878 hddivsteps.step (hddivsteps.init f g) in
  Z.abs (hddivsteps.f st) = 1%Z /\
   eqm f ((hddivsteps.d st * hddivsteps.f st) * g) 1.
Proof.
intros f g Hf HM Hg Hprime.
eapply processDivstep_inverse; try assumption; try apply HM.
apply example878.
Qed.

Check hddivsteps878_inverse.
Print Assumptions hddivsteps878_inverse.

Theorem hddivsteps878_prime_inverse : forall f g,
  Z.Odd f ->
  Znumtheory.prime f ->
  (g < f <= 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab)%Z ->
  let st := N.iter 878 hddivsteps.step (hddivsteps.init f g) in
  (g = 0 -> hddivsteps.d st = 0)%Z /\
  (0 < g -> Z.abs (hddivsteps.f st) = 1 /\
            eqm f ((hddivsteps.d st * hddivsteps.f st) * g) 1)%Z.
Proof.
intros f g Hf Hprime HM Hg.
eapply processDivstep_prime_inverse; try assumption; try apply HM.
apply example878.
Qed.

Check hddivsteps878_prime_inverse.
Print Assumptions hddivsteps878_prime_inverse.
