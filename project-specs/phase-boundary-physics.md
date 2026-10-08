# Physics checks for moiré WSe2 phase boundaries

This note is for a later agent that writes extractors or decides whether a proposed temperature–filling boundary is physically viable. It records how phase transitions and crossovers work in this experiment, and the constraints a boundary has to satisfy. It is not a substitute for Xia et al., Nature 650, 585 (2026).

Use it together with:

- [Phase_diagram_construction_E-103_v3.docx](Phase_diagram_construction_E-103_v3.docx), the manual workflow at E = −103 mV/nm.
- [nature_figure3_ground_truth.csv](nature_figure3_ground_truth.csv), the published Figure 3 symbols for 87, 96, 99, and 103 mV/nm.
- [goal-scope-and-progress.md](goal-scope-and-progress.md), what the code currently does.

The published points are a check after scoring. They are not features inside a score.

## The system

The sample is twisted bilayer WSe2 in the moderate-correlation regime of the triangular-lattice Hubbard model, twist angle near 4.6°. A vertical displacement field E tunes the bandwidth and drives a Mott transition. Hole filling ν is the number of holes per moiré cell. The diagrams in this repo are slices at fixed E: temperature T vertical, filling ν horizontal.

Repository field id `F` means the paper’s E = −F mV/nm. The file `Rxx_matrix_E-103mV_nm.csv` is the E = −103 mV/nm cut.

The Mott critical field on the published device is about E_c ≈ −92 mV/nm.

- Insulator side of the Mott transition, E more negative than E_c: 96, 96.2, 99, 103, 151, and 176 mV/nm.
- Metal side, E less negative than E_c: 74 and 87 mV/nm.

Only 87, 96, 99, and 103 appear in Nature Figure 3. The placement of 74, 96.2, 151, and 176 relative to E_c is a plausibility guide for this device. E_c itself moves with twist angle and strain.

## Transition versus crossover

A phase is a region of (T, ν) with one kind of order or one kind of charge transport. Two different things get drawn on the same diagram.

A transition is a qualitative change tied to an order parameter or a collapse of resistance:

- Antiferromagnetic order onsets at the Néel temperature T_N. Magnetic susceptibility shows a cusp. In transport, the same scale is read from a local resistivity minimum or from the onset of a jump in Hall density.
- Superconductivity onsets at T_c. In two dimensions this is a Berezinskii–Kosterlitz–Thouless transition. The paper’s highest T_c is taken from that superconducting characterization. A linecut only shows the related fact that resistance falls toward zero.

A crossover is a smooth change in the form of ρ(T). There is no requirement of a diverging length or a broken symmetry. The paper defines both crossover scales by a 10% rule:

- Coherence temperature T_coh: temperature where ρ leaves ρ = ρ_0 + A T² by 10%. Below T_coh the metal is a Fermi liquid.
- Crossover temperature T′: temperature where ρ leaves a T-linear fit by 10% on the way down into a sublinear metal.

T_coh and T′ are scales on a boundary, and they are wider and more ambiguous than T_N or T_c. A fit that moves either of them by a few tenths of a kelvin can still be the same crossover. A boundary that treats them as infinitely sharp, or that searches for a smaller fit window until the exponent matches a preferred story, is not the paper’s definition.

The same filling is often more than one phase in different temperature windows. A low-temperature T² metal can be a Fermi liquid. The same quadratic shape at high temperature does not by itself earn that name.

## What each phase looks like in transport

Longitudinal resistivity ρ_xx at B = 0. The paper’s workbook column is `rho_xx (ohm)`. The manual workflow treats a drop below about 20 Ω as resistance having reached zero.

| Region | Where it sits | What ρ(T) does |
| --- | --- | --- |
| AF insulator | ν = 1, insulator side of E_c | ρ falls on cooling, then rises sharply below T_N, often by orders of magnitude. The local minimum is T_N. At E = −103 mV/nm that minimum is about 3.5 K. |
| Doped AF metal | small electron or hole doping off ν = 1 | Metallic overall, with a sharp peak or rise near T_N. Below that peak the metal has a small Fermi surface of density about \|1 − ν\|. T_N falls as doping increases and hits zero at a finite doping. |
| Superconductor | a dome on each side of ν = 1 on the insulator side of E_c; one dome on the metal side | ρ → 0 at low T. Highest T_c on this device is about 0.4 K, next to the insulator. The electron-side dome (ν < 1) is the stronger one at E = −103 mV/nm. |
| Strange metal | near the doping where the Mott gap collapses, covering the superconducting dome and a window just outside it | ρ linear in T over a wide range, down to the lowest T once a small field (0.3 T) has killed superconductivity. Slope of order 100 Ω/K, residual resistivity below about 100 Ω. |
| Sublinear metal | electron side, below T′ | ρ linear above T′, sublinear below T′. |
| Fermi liquid | large doping, outside the strange metal | ρ = ρ_0 + A T² at low T, with A > 0. Departure from that form is T_coh. A grows large as T_coh collapses toward the strange metal. |

Sign of the slope, used by the manual workflow on a metallic extraction window:

- dρ/dT > 0 over most of the window: metal.
- dρ/dT < 0 over a substantial low-temperature stretch: insulator, or the insulating side of T_N.
- A local minimum, with metallic behavior again at the lowest T: doped AF metal.
- ρ falling through the zero-resistance threshold: superconductor.

At E = −103 mV/nm the published layout, from the paper’s linecuts, is approximately:

- ν = 1.00: AF insulator, T_N from the resistivity minimum.
- ν = 0.98 and 1.02: doped AF metals; T_N lower than at ν = 1.
- T_N vanishes near ν = 0.96 and ν = 1.04.
- 0.88 ≤ ν ≤ 0.96: T-linear above T′, sublinear below T′. T′ is dome-shaped and vanishes near ν = 0.88. ν = 0.94 is the worked example of this crossover.
- ν = 0.88: strange metal, linear over the whole measured range. A gap of about 0.02 in filling separates this from the Fermi liquid.
- ν below about 0.88, and the hole-side example ν = 1.13: Fermi liquid. T_coh falls to zero as ν approaches the strange metal from outside.
- Hole side, ν > 1: the same sequence of AF metal, strange metal, and Fermi liquid, without an identified T′ or sublinear metal.

Those fillings are the E = −103 mV/nm anchors. They move as E changes. They are not universal constants to hard-code for every field.

## Shape of a viable boundary

Draw each scale as its own curve in the (T, ν) plane. A viable diagram on the insulator side of E_c has this topology.

- T_N is one dome centered on ν = 1. Its maximum is at ν = 1, or indistinguishable from ν = 1 given the filling step of the matrix. It falls on both the electron side and the hole side and reaches zero at finite doping. The two sides are one physical dome, split by the insulator at ν = 1 only in the sense that the insulator is the top of the dome.
- T_coh is two separate branches. The electron branch lives at ν below the strange metal and falls to zero as ν increases toward that metal. The hole branch lives at ν above the hole-side strange metal and falls to zero as ν decreases toward it. The two branches do not connect across ν = 1. Interpolating T_coh through the AF insulator is unphysical.
- T′ is an electron-side dome between the Fermi liquid and the doped AF metal. It vanishes where the Fermi liquid ends. The published hole side has no T′. Inventing a hole-side T′ requires a linecut that is linear at high T and persistently sublinear below the 10% departure. A single noisy point 10% off a line is not that linecut.
- T_c is a dome under the strange metal, one on each side of ν = 1, with the electron side higher at E = −103 mV/nm. T_c sits at the lowest temperature of the scales on that filling. On this device it is a fraction of a kelvin, while T_N at ν = 1 is a few kelvin, until E approaches E_c and T_N collapses.

Inside one branch, T changes smoothly with ν. The manual workflow forces that smoothness by shrinking the fit window from one anchor linecut to the next, so T_coh and T_N descend instead of jumping. Adjacent fillings in these matrices are about 0.002 to 0.01 apart. A jump of order 1 K between neighbors, or a boundary that oscillates by much more than the local temperature spacing, fails the continuity test even if every point is a locally acceptable fit.

Empty is allowed. At E = −87 mV/nm the workbook has T_coh and has neither T_N nor T′. Forcing those curves to exist on the metal side of E_c fails the Mott-transition test.

## How the diagram moves with E

Published trend, insulator side, moving from E = −103 toward E = −96 (toward E_c):

- The maximum T_N drops.
- The doped-AF filling width shrinks.
- The superconducting domes widen and the optimal T_c rises.
- Fermi liquids remain at large doping on both sides, and A still rises where T_coh collapses.
- T′ remains an electron-side scale in the published panels.

Published trend from the susceptibility and from T_c / T_N: at E = −110 mV/nm, T_c / T_N is about 4%; at E = −96 mV/nm it is about 18%; the ratio diverges as T_N vanishes at E_c. So 151 and 176 mV/nm, further onto the insulator side than 103, are expected to show a taller T_N dome and a weaker superconductor than 103. That expectation is not a published diagram.

Metal side, E = −87 mV/nm, past E_c:

- The AF insulator at ν = 1 is gone, so there is no T_N dome.
- One superconducting dome remains.
- T-linear resistivity spans a wide window, about 0.86 ≤ ν ≤ 1.01, including across ν = 1.
- The published symbols are T_coh only.

74 mV/nm is further onto that metal side than 87. The same qualitative picture is the expectation. 96.2 mV/nm sits next to 96 and should look like 96.

## Checks an agent can run

Run these after a boundary is proposed. Failures and warnings are separate. A warning can be a real feature of a noisy linecut; a failure means the boundary contradicts the phase diagram.

Failures:

- A T_coh branch crosses ν = 1 or joins the electron and hole Fermi liquids into one curve.
- T_N increases as \|ν − 1\| increases, or its maximum sits far from ν = 1 on the insulator side of E_c.
- T_N is reported where the smoothed linecut has no local minimum, no sharp peak, and no low-temperature insulating rise.
- T_N or a Fermi-liquid T_coh is reported inside the ν = 1 insulator, where ρ is rising by a large factor on cooling.
- T_coh is reported where the retained low-temperature fit is not ρ_0 + A T² with A > 0, or where the 10% departure is a single point rather than a persistent departure.
- T′ is reported where the high-temperature side is not linear, or where the low-temperature side is not persistently sublinear (exponent remaining below 1 inside its uncertainty).
- A hole-side T′ is reported for a published insulator-side field without that persistent sublinear departure. The paper looked for it at E = −103 mV/nm and did not find it.
- T_c is reported where ρ stays well above the zero-resistance threshold over the whole linecut.
- T_c lies above T_N at ν = 1 on the deep insulator side. T_c may become comparable to T_N only as E approaches E_c and T_N falls toward zero.
- On the metal side of E_c, a T_N dome or an AF insulator at ν = 1 is reported.
- The score uses a published (ν, T) from the ground-truth table as an input. Comparison happens after the candidate exists.
- A phase name is assigned from the exponent alone, with no temperature window. High-temperature T² is not automatically a Fermi liquid. Low-temperature T² in a metal outside the strange-metal filling is.

Warnings:

- Within one branch, \|ΔT\| between neighboring fillings is large compared with the temperature spacing and with the 0.2 K tolerance already used against the published anchors.
- The best local score and the candidate nearest a published anchor disagree by well above that tolerance. The ground-truth comparison on the current extractor shows this often: the nearest candidate is much closer than the top score.
- The fit window stays wide while T_coh or T_N is supposed to be descending toward zero.
- A linear, sublinear, or superlinear label flips between adjacent fillings with no T_coh or T′ there to mark the change.
- A low-temperature minimum is only a few noise widths tall, or it exists on one linecut and on neither neighbor. The smoothing step exists because low-T noise manufactures minima.

Against [nature_figure3_ground_truth.csv](nature_figure3_ground_truth.csv), for fields 87, 96, 99, and 103:

- Match each published point to the nearest measured filling.
- Report the absolute error of the selected temperature, and the error of the nearest candidate if several are kept.
- A selected point within about 0.2 K is consistent with the tolerance already used in `crossover_analysis`. A larger error is a miss, not an automatic failure of the physics, because the published symbol is one expert’s 10% reading of a smooth curve.
- A miss in which no candidate lies near the published point is more serious than a miss in which the right temperature was proposed and then discarded.
- Field 87 must be allowed to have zero T_N and zero T′. Scoring a penalty for those absences is wrong.

## What R_xx cannot decide

Hall density and magnetic susceptibility are part of the paper’s identification and are not in this repository.

- The doped AF metal is confirmed by n_H / n_M approaching 1 − ν, and by a Hall jump of about one moiré density where the Mott gap collapses. At E = −87 mV/nm the jump near the van Hove singularity is about two moiré densities. An R_xx-only T_N is the resistivity minimum, which the paper uses as the estimate, and it can be shifted by noise.
- The highest T_c anchor is fixed by the superconducting characterization (critical field, BKT temperature), not by the single best ρ(T) curve. The resistance threshold is a proxy that finds where the dome is, and it can place T_c a bit off the published anchor.
- Calling the ν = 1 state an antiferromagnet, rather than a featureless insulator, uses the susceptibility cusp. Transport shows the insulating upturn.

An agent may still draw a viable transport diagram from R_xx. It should leave Hall reconstruction and the magnetic order parameter unmarked, and it should say so.

## How to turn this into code

Keep the two stages in the SURP report. First extract scales. Then name phases. The naming stage is where this note applies.

Practical rules that follow from the physics:

- Score each linecut locally, then keep or reject candidates with the neighbor and continuity rules above. A linecut-only score cannot see that T_N must fall away from ν = 1 or that the two T_coh branches stay apart.
- Store electron and hole branches as separate segments, as the ground-truth table already does (`segment` is `electron` for ν < 1 and `hole` for ν > 1).
- Define T_coh and T′ by a persistent 10% departure, with a positive-A quadratic or a linear baseline respectively. Keep the fit window that the baseline actually describes. Do not shrink it merely to manufacture a chosen exponent, except when enforcing a smooth descent between anchors that have already been identified.
- Define T_N from a persistent local minimum or peak on the insulator side of E_c, centered at ν = 1, and allow the scale to be absent on the metal side.
- Define the transport T_c from a collapse through the zero-resistance threshold, and keep it below the other scales on that filling.
- Name Fermi liquid, strange metal, sublinear metal, doped AF metal, AF insulator, and superconductor only after the scales exist, using the temperature window each name belongs to.
- When several parameter settings are possible, prefer the setting whose boundaries pass the failure list and whose selected temperatures sit near the ground-truth points on the four published fields. Then ask a person to look at 74, 96.2, 151, and 176, which have no Figure 3 reference.
