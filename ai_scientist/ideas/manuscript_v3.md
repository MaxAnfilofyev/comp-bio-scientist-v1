# A Topological Tipping Point Explains the Selective Vulnerability of Substantia Nigra Neurons

**Max Anfilofyev**
[max.anfilofyev@gmail.com](mailto:max.anfilofyev@gmail.com)

## Abstract

**Introduction:** The selective degeneration of substantia nigra pars compacta (SNc) dopaminergic neurons, while adjacent ventral tegmental area (VTA) neurons survive, is a defining paradox of Parkinson’s disease. Current molecular theories—citing mitochondrial dysfunction, calcium stress, or $\alpha$-synuclein aggregation—describe damage mechanisms but fail to explain why these ubiquitous stressors selectively doom SNc neurons while sparing VTA counterparts.

**Methods:** We developed a minimal two-variable dynamical model of neuronal energetics, incorporating mitochondrial functional capacity ($M$) and energetic reserve ($E$) under two physiological loads: axonal arborization ($A$) and calcium pacemaking ($C$). We performed bifurcation analysis, stochastic noise simulations, and hysteresis tracking to map the stability landscapes of VTA and SNc populations.

**Results:** We identify a load-induced saddle-node bifurcation that divides the physiological phase space into two distinct regimes: a robust monostable regime and a fragile bistable regime containing a "collapsed" energetic attractor. Using anatomical calibration, we show that VTA neurons ($A \approx 0.4$) reside safely in the monostable zone, while SNc neurons ($A \approx 1.0$) occupy the bistable window. In this regime, physiological noise is sufficient to drive neurons across a separatrix into irreversible collapse. Furthermore, we demonstrate that once collapsed, neurons are trapped in a hysteresis loop where even substantial axonal pruning cannot restore the high-energy state.

**Discussion:** These findings suggest that SNc vulnerability is a topological consequence of extreme structural and metabolic loads. By defining the geometry of this "energetic trap," the model provides a unified explanation for the selectivity, prodromal latency, and therapeutic irreversibility of Parkinsonian degeneration.

-----

## Author Summary

Why do the dopamine-producing neurons in the substantia nigra die in Parkinson's disease, while their neighbors in the ventral tegmental area survive? We propose that the answer lies not just in "bad genes" or molecular damage, but in geometry. SNc neurons are physically massive, maintaining axonal arbors ten times larger than their neighbors. Using a mathematical model of cellular energy, we discovered that this extreme structural load pushes SNc neurons onto a "metabolic cliff edge." While healthy neurons sit in a deep, safe valley, SNc neurons operate in a precarious state where even minor biological noise can push them over the edge. Once they fall, they are trapped at the bottom of the cliff, explaining why the disease is irreversible. This model suggests that successful treatments must pull neurons back from the edge *before* they fall, reframing Parkinson's as a problem of structural energy management.

## 1. Introduction

Parkinson’s disease is characterized by the relentless and selective degeneration of dopaminergic neurons in the substantia nigra pars compacta (SNc), whereas dopaminergic neurons in the nearby ventral tegmental area (VTA) are remarkably resistant. This anatomical specificity poses a longstanding riddle: Why do two neuronal populations that share the same neurotransmitter, similar genetic profiles, and identical exposure to environmental toxins exhibit such divergent fates?

Current explanations largely focus on molecular checklists. SNc neurons are known to harbor specific vulnerabilities, including elevated mitochondrial oxidant stress, high levels of cytosolic dopamine oxidation, and a reliance on L-type Ca²⁺ channels for pacemaking. Furthermore, genetic risk factors like LRRK2 mutations and pathogenic processes like $\alpha$-synuclein fibril formation contribute heavily to the overall burden. Yet, these molecular stressors are drivers of damage, not explanations for the *topology* of the collapse. They do not inherently explain why the disease is characterized by a decades-long prodromal phase followed by a rapid, irreversible decline, nor why VTA neurons—which also face significant metabolic demands—remain robust.

To resolve this paradox, we adopt a theoretical framework based on **minimal dynamical systems**. Complex biological models with dozens of parameters often obscure the fundamental mechanism of failure. **A two-variable system (energy $E$ and mitochondrial mass $M$) is mathematically the minimal dimensionality capable of expressing a saddle-node bifurcation with hysteresis while maintaining biological interpretability; one-variable models are constrained to monotonic decay.** We propose that selective vulnerability is not a failure of specific molecules, but a geometric inevitability of **systemic load**. SNc neurons carry an extraordinary structural burden: their axonal arbors are orders of magnitude larger than those of VTA neurons, comprising hundreds of thousands of synapses. This massive infrastructure imposes a continuous metabolic tax that, when combined with the energetic cost of Ca²⁺ clearance, pushes the neuron toward the limits of its bioenergetic capacity.

Here, we demonstrate that these combined loads deform the neuron's energetic landscape. We show that increasing structural load drives the system through a **saddle-node bifurcation**, creating a **bistable** topology where a healthy high-energy state coexists with a collapsed low-energy state. By mapping anatomical data onto this landscape, we find that VTA neurons reside in a monostable "safe zone," while SNc neurons are trapped in a bistable regime. In this precarious state, intrinsic physiological noise is sufficient to trigger stochastic collapse. Furthermore, the topology of the system enforces a "hysteresis trap" that renders degeneration mathematically irreversible via simple pruning. This framework unifies the diverse molecular correlates of Parkinson's into a single mechanistic theory: SNc neurons die because their extreme anatomy places them on the wrong side of a metabolic tipping point.

---

## 2. Results

### 2.1. Structural Load Induces a Metabolic Tipping Point
To determine how physical structure constrains energetic stability, we analyzed the model’s equilibrium behavior across a continuous range of axonal loads ($A$). To distinguish structural inevitability from parameter tuning, we explicitly compared the full cooperative model (EC3) against two "null" architectures: a linear mitochondrial model (EC1) and a feedback-only model without cooperative amplification (EC2) (see S2 Text).

**Analytically, the energy nullcline of the linear model (EC1) is strictly monotonic ($dE/dM > 0$). This condition mathematically precludes multiple fixed points regardless of parameter values.** Similarly, feedback-only models (EC2) lack the cubic inflection point required for multiple intersections. This confirms that the **nonlinear cooperative feedback** term ($k_2 E^2$)—representing the sigmoidal kinetics of Ca²⁺-stimulated dehydrogenases and the $\Delta\Psi_m$ threshold for ATP synthesis—is essential for generating the S-shaped energy nullcline required for a tipping point.

In the EC3 model, low loads characteristic of VTA neurons ($A \approx 0.3–0.5$) yield a single stable equilibrium corresponding to a healthy, high-energy state (Fig 3). However, as axonal load increases, the increased ATP demand deforms the energy nullcline. We identified a critical threshold at $A \approx 0.86$ where the nullcline inflection forces a **saddle-node bifurcation**. Beyond this point, a new pair of equilibria emerges: a stable low-energy node and an unstable saddle point.

It is important to distinguish this "collapsed" state from immediate necrosis. The low-energy attractor ($E \approx 0.1$) represents a state of **metabolic quiescence or depolarization block**. While functionally silent, the cell remains viable in the short term. However, at this energy level, the neuron cannot maintain the ionic gradients (via Na⁺/K⁺ ATPase) or proteostatic clearance mechanisms required for long-term survival, rendering it a "metabolic zombie" that eventually degenerates (5, 24).

### 2.2. SNc and VTA Neurons Occupy Distinct Topological Regimes
We next mapped the physiological load of specific dopaminergic populations onto this bifurcation structure. Crucially, our anatomical calibration was performed **independently** of the dynamical analysis: we defined the load parameter $A$ using an affine mapping of synaptic counts derived from literature (see Methods). When we subsequently overlaid these ranges onto the bifurcation diagram, we found that VTA neurons cluster safely in the monostable regime, while the median SNc neuron falls directly inside the bistable window (Fig 3, orange band).

In this main bifurcation analysis, we held Ca²⁺-handling load constant ($C=1$). However, SNc neurons express higher densities of Cav1.3 channels and exhibit more robust autonomous pacemaking than VTA neurons. **We treat $A$ and $C$ as multiplicative co-loads because the total metabolic cost is physically the product of the number of functional sites ($A$) and the energetic cost per site ($C$).** As shown in our full 2D stability analysis (S8 Fig), increasing $C$ shifts the bifurcation fold leftward by the same magnitude as increasing $A$. Consequently, the empirically higher $C$ in SNc neurons ($C \approx 1.2–1.4$) pushes them further into the "danger zone." Based on reported anatomical variances, our synthetic population analysis estimates that approximately **78% of SNc neurons fall within the bistable band**, compared to 0% of VTA neurons.

### 2.3. Noise-Driven Collapse and the Prodromal Phase
The existence of a bistable window explains how neurons can survive for decades before collapsing. In a deterministic system, a neuron in the high-energy well would remain there indefinitely. However, biological systems are inherently noisy. We simulated physiological noise by introducing stochastic fluctuations to the energy variable $E$.

For VTA-like neurons (monostable), large noise fluctuations were rapidly corrected (Fig 5A). Conversely, for SNc-like neurons (bistable), the "basin of attraction" for the healthy state is shallow. **As the load ($A$) increases toward the right edge of the window, the energetic barrier ($\Delta E$) separating the healthy state from the saddle point shrinks exponentially.** We observed that physiological levels of noise ($\sigma=0.05$, consistent with mitochondrial potential variance) were sufficient to stochastically kick SNc trajectories over this barrier (Fig 5B). This dependence on basin depth offers a geometric explanation for the variable age of onset in Parkinson's: aging or pathology acts to shrink the basin, thereby exponentially increasing the probability of a noise-induced crossing.

### 2.4. The Trap of Irreversibility (Hysteresis)
A critical feature of the saddle-node topology is hysteresis: the path to collapse is different from the path to recovery. We simulated a "rescue" scenario where an SNc neuron undergoes noise-induced collapse, followed by a therapeutic reduction in axonal load (pruning) (Fig S14).

Remarkably, moderate pruning (reducing $A$ from 1.0 to 0.9) failed to rescue the neuron. **Recovery is mathematically impossible because the low-energy attractor persists until $A$ drops below the left fold ($A \approx 0.86$).** We confirmed that the mitochondrial nullcline geometry prevents any "backdoor" escape via $M$-dynamics. Crossing the left fold implies an arbor reduction of >50–70%, a magnitude that exceeds the limits of endogenous plasticity without triggering axonal die-back [16]. This hysteresis loop renders the collapse effectively irreversible in vivo.

---

## 3. Discussion

### 3.1. Mechanism: Topology Overrides Molecular Inventory
Our model offers a unified interpretation of why ubiquitous stressors cause selective degeneration. In our framework, specific molecular defects act as parameter shifts. For example, **LRRK2 mutations** (increasing kinase activity) and **$\alpha$-synuclein accumulation** (impairing Complex I and mitophagy) can be modeled as reductions in $M$ and $k_M$. These shifts move the saddle-node bifurcation leftward, expanding the danger zone to engulf neurons with previously "safe" loads.

A VTA neuron, sitting far from the bifurcation in a deep monostable well ($A \approx 0.4$), has a massive safety margin. An SNc neuron, perched inside the bistable window, lacks this margin. Importantly, **other large projection neurons (e.g., corticospinal tracts) possess large arbors ($A$) but lack the high-cost calcium pacemaking ($C$), placing them in a safe regime.** This intersection of structural and physiological loads explains the unique vulnerability of the SNc. We emphasize that load-induced bistability is a **sufficient**, though not necessarily **exclusive**, condition for selectivity.

### 3.2. Mapping Dynamics to Clinical Staging
The model provides a dynamical mapping to clinical progression:
1.  **Prodromal Phase (Metastability):** The long latency of PD corresponds to the time neurons spend in the high-energy well of the bistable regime. They are functional but dynamically fragile.
2.  **Onset (Stochastic Collapse):** The random nature of noise-driven basin crossing explains the heterogeneity of onset age.
3.  **Progression (Fold Shifting):** As pathology accumulates (e.g., $\alpha$-synuclein spread), the bifurcation fold shifts leftward, recruiting more of the population into the bistable/collapsed regime.

### 3.3. Predictive Biomarkers
The model offers testable predictions. Specifically:
1.  **Critical Slowing Down:** iPSC-derived dopaminergic neurons under metabolic stress should exhibit a measurable increase in the time constant ($\tau$) of mitochondrial recovery following transient stimulation.
2.  **Graft Failure:** The hysteresis trap explains the poor survival of dopaminergic grafts, which must re-establish massive arbors under metabolic stress, potentially entering the bistable regime before maturation.

### 3.4. Human Relevance and Scaling
While calibrated to rodent data, the topological trap is likely exacerbated in humans. The scaling ratio between SNc and VTA arborization is conserved or even amplified in primates, with **human SNc axons reaching total lengths of up to 4 meters**. This massive expansion of $A$ implies that human neurons operate even deeper within the bistable regime, potentially explaining the species-specific prevalence of Parkinson’s disease.

### 3.5. Limitations
Our approach has limitations. First, we simplify complex biochemistry into two aggregate variables, $E$ and $M$. Second, the nonlinear gain parameter $k_2$ is phenomenological, though grounded in cooperative mitochondrial kinetics. Third, we model the neuron as a single compartment. Biologically, the collapse likely initiates in the **distal axon**, where the energetic safety margin is thinnest. This minimal framework captures the geometric mechanism of failure but is not intended to model the full molecular complexity of PD pathology.

---

## 4. Methods

### 4.1. Mathematical Model and Timescales
We constructed a minimal energetic framework consisting of two coupled ordinary differential equations on the state space $(E, M) \in [0,1]^2$. **A two-variable system is mathematically the minimal dimensionality capable of expressing a saddle-node bifurcation with hysteresis; one-variable models are constrained to monotonic decay.**

$$
\frac{dE}{dt} = k_1 M (1-E) + k_2 E^2 (1-E) - (L_0 + L_1 A C) E
$$
$$
\frac{dM}{dt} = k_M (1-M) - \beta A C M (1-E)
$$

**Total metabolic demand is modeled as an affine function:** a basal housekeeping load ($L_0$) plus a variable load ($L_1 \cdot A \cdot C$) proportional to the product of functional sites ($A$) and cost-per-site ($C$). The term $k_2 E^2 (1-E)$ represents cooperative nonlinear feedback. The dynamics of $E$ (ATP/ADP ratio) evolve on seconds to minutes, while $M$ (mitochondrial mass) evolves over hours to days.

### 4.2. Parameterization and Calibration
* **Axonal Load ($A$):** We utilized an affine mapping $A = 0.3 + 0.7 \times (N_{syn} / 400,000)$. With $N_{SNc} \approx 400,000$ synapses [1], the median SNc neuron maps to $A=1.0$, and the median VTA neuron ($N \approx 40,000$) maps to $A \approx 0.37$.
* **Calcium Load ($C$):** We normalized $C=1$ to a baseline pacemaking phenotype. Given the reliance of SNc neurons on L-type Cav1.3 channels versus HCN-driven mechanisms in the VTA, estimates suggest the specific metabolic cost per spike is 20–40% higher in the SNc ($C \approx 1.2–1.4$).

### 4.3. Bifurcation and Stability Analysis
The saddle-node bifurcation locus is defined by the simultaneous solution of $F(E,M;A)=0$ and $\det(J)=0$. Numerical continuation confirmed the left fold at $A_{left} \approx 0.86$. We analytically verified the **positive invariance** of the unit square $[0,1]^2$; the vector field points inward at all boundaries.

### 4.4. Numerical Implementation and Sensitivity
Stochastic dynamics were simulated by adding Gaussian white noise to the $E$ variable ($\sigma dW_t$). The noise amplitude $\sigma = 0.05$ approximates the 5–10% coefficient of variation observed in mitochondrial membrane potential fluctuations. **This fast-slow separation justifies applying stochastic fluctuations primarily to the fast variable $E$.** To rule out numerical artifacts near the stiff bifurcation fold, we validated critical results using implicit BDF integrators with strict tolerances ($10^{-6}$).

---

## Supporting Information Captions

**S2 Text. Null Model Analysis.** Explicit derivation showing that linear models ($\dot{E} \propto M$) have a strictly monotonic nullcline ($dE/dM > 0$) and cannot exhibit bistability.

**S3 Fig. Nullcline geometry comparison.** (A) Monostable nullcline intersection in the linear model (EC1). (B) The S-shaped nullcline created by the $k_2 E^2 (1-E)$ term in the EC3 model.

**S8 Fig. Stability regimes in the (A, C) parameter space.** The diagonal purple band denotes the bistable "danger zone." Increasing Ca²⁺ load ($C$) shifts the danger zone to lower axonal loads ($A$), confirming multiplicative scaling.

**S12 Fig. Population distribution relative to the bistable window.** Synthetic distributions of VTA and SNc neurons based on log-normal fits to anatomical variance. The SNc distribution significantly overlaps with the bistable band.

**S14 Fig. The hysteresis trap.** Time course showing failure to recover from collapse despite moderate pruning.

**S16 Fig. Separatrix geometry.** Phase plane at $A=1.0$ showing the basins of attraction for the healthy and collapsed states, separated by the stable manifold of the saddle.

**S17 Fig. Noise sensitivity sweep.** Probability of collapse as a function of noise amplitude $\sigma$, showing threshold-like behavior.

### 4.5. Reproducibility
All code used to generate the models, phase planes, and bifurcation diagrams is available at [\[GitHub Repository Link\]](https://github.com/MaxAnfilofyev/parkinsons-ec3-model). The repository includes the full Python environment specifications to ensure exact reproducibility of the bifurcation loci and stochastic outcomes.

## **References**

1. Matsuda, Wakoto, Takahiro Furuta, Kouichi C. Nakamura, et al. “Single Nigrostriatal Dopaminergic Neurons Form Widely Spread and Highly Dense Axonal Arborizations in the Neostriatum.” The Journal of Neuroscience 29, no. 2 (2009): 444–53. https://doi.org/10.1523/JNEUROSCI.4029-08.2009.
2. Bolam, John P., Jonathan J. Hanley, Peter A. Booth, and Michael D. Bevan. “Synaptic Organisation of the Basal Ganglia.” Journal of Anatomy 196, no. 4 (2000): 527–42. https://doi.org/10.1046/j.1469-7580.2000.19640527.x.
3. Gauthier, J., M. Parent, M. Lévesque, and A. Parent. “The Axonal Arborization of Single Nigrostriatal Neurons in Rats.” Brain Research 834, nos. 1–2 (1999): 228–32. https://doi.org/10.1016/S0006-8993(99)01573-5.
4. Pan, Wei-Xin, Tong Mao, and Joshua T. Dudman. “Inputs to the Dorsal Striatum of the Mouse Reflect the Parallel Circuit Architecture of the Forebrain.” Frontiers in Neuroanatomy 4 (2010): Article 147. https://doi.org/10.3389/fnana.2010.00147.
5. Surmeier, D. James, J. N. Guzman, J. Sanchez-Padilla, and J. A. Goldberg. “What Causes the Death of Dopaminergic Neurons in Parkinson’s Disease?” Progress in Brain Research 183 (2010): 59–77. https://doi.org/10.1016/S0079-6123(10)83004-3.
6. Guzman, J. N., J. Sánchez-Padilla, C. S. Chan, and D. J. Surmeier. “Robust Pacemaking in Substantia Nigra Dopaminergic Neurons.” The Journal of Neuroscience 29, no. 35 (2009): 11011–19. https://doi.org/10.1523/JNEUROSCI.2519-09.2009.
7. Chan, C. S., J. N. Guzman, E. Ilijic, et al. “‘Rejuvenation’ Protects Neurons in Mouse Models of Parkinson’s Disease.” Nature 447, no. 7148 (2007): 1081–86. https://doi.org/10.1038/nature05865.
8. Zampese, E., D. L. Wokosin, P. Gonzalez-Rodriguez, et al. “Ca²⁺ Channels Couple Spiking to Mitochondrial Metabolism in Substantia Nigra Dopaminergic Neurons.” Science Advances 8, no. 37 (2022): eabp8701. https://doi.org/10.1126/sciadv.abp8701.
9. Surmeier, D. J. “Calcium, Ageing, and Neuronal Vulnerability in Parkinson’s Disease.” The Lancet Neurology 6, no. 10 (2007): 933–38. https://doi.org/10.1016/S1474-4422(07)70246-6.
10. Exner, N., A. K. Lutz, C. Haass, and K. F. Winklhofer. “Mitochondrial Dysfunction in Parkinson’s Disease: Molecular Mechanisms and Pathophysiological Consequences.” The EMBO Journal 31, no. 14 (2012): 3038–62. https://doi.org/10.1038/emboj.2012.170.
11. Grünewald, A., K. A. Rygiel, P. D. Hepplewhite, et al. “Mitochondrial DNA Depletion in Respiratory Chain-Deficient Parkinson Disease Neurons.” Annals of Neurology 79, no. 3 (2016): 366–78. https://doi.org/10.1002/ana.24571.
12. Burbulla, L. F., P. Song, J. R. Mazzulli, et al. “Dopamine Oxidation Mediates Mitochondrial and Lysosomal Dysfunction in Parkinson’s Disease.” Science 357, no. 6357 (2017): 1255–61. https://doi.org/10.1126/science.aam9080.
13. Schapira, A. H. V. “Mitochondrial Complex I Deficiency in Parkinson’s Disease.” Journal of Neurochemistry 55, no. 4 (1990): 1471–75. https://doi.org/10.1111/j.1471-4159.1990.tb02325.x.
14. Bose, Anup, and Michael F. Beal. “Mitochondrial Dysfunction in Parkinson’s Disease.” Journal of Neurochemistry 139 (Suppl. 1, 2016): 216–31. https://doi.org/10.1111/jnc.13731.
15. Pacelli, Consiglia, Nicolas Giguère, Marie-José Bourque, et al. “Elevated Mitochondrial Bioenergetics and Axonal Arborization Size Are Key Contributors to the Vulnerability of Dopamine Neurons.” Current Biology 25, no. 18 (2015): 2349–60. https://doi.org/10.1016/j.cub.2015.07.050.
16. Giguère, Nicolas, Samuel Burke Nanni, and Louis-Éric Trudeau. “On Cell Loss and Selective Vulnerability of Neuronal Populations in Parkinson’s Disease.” Frontiers in Neurology 9 (2018): Article 455. https://doi.org/10.3389/fneur.2018.00455.
17. Surmeier, D. J., J. N. Guzman, and J. Sanchez-Padilla. “Calcium, Cellular Aging, and Selective Neuronal Vulnerability in Parkinson's Disease.” Cell Calcium 47, no. 2 (2010): 175–82. https://doi.org/10.1016/j.ceca.2009.12.003.
18. Wong, Y. C., and D. Krainc. “α-Synuclein Toxicity in Neurodegeneration: Mechanism and Therapeutic Strategies.” Nature Medicine 23, no. 2 (2017): 1–13. https://doi.org/10.1038/nm.4269.
19. Conway, K. A., J. D. Harper, and P. T. Lansbury. “Accelerated In Vitro Fibril Formation by a Mutant α-Synuclein Linked to Early-Onset Parkinson’s Disease.” Nature Medicine 4, no. 11 (1998): 1318–20. https://doi.org/10.1038/3311.
20. Cookson, M. R. “The Role of Leucine-Rich Repeat Kinase 2 (LRRK2) in Parkinson’s Disease.” Nature Reviews Neuroscience 11, no. 12 (2010): 791–801. https://doi.org/10.1038/nrn2930.
21. Strogatz, Steven H. Nonlinear Dynamics and Chaos. 2nd ed. Boca Raton: CRC Press, 2018. https://doi.org/10.1201/9780429492563.
22. Scheffer, Marten, Stephen R. Carpenter, Timothy M. Lenton, et al. “Anticipating Critical Transitions.” Science 338, no. 6105 (2012): 344–48. https://doi.org/10.1126/science.1225244.
23. Kuehn, Christian. “A Mathematical Framework for Critical Transitions: Bifurcations, Fast-Slow Systems and Stochastic Dynamics.” Journal of Nonlinear Science 23, no. 3 (2013): 457–510. https://doi.org/10.1007/s00332-012-9158-x.
24. Harris, John J., Régis Jolivet, and David Attwell. “Synaptic Energy Use and Supply.” Neuron 75, no. 5 (2012): 762–77. https://doi.org/10.1016/j.neuron.2012.08.019.
25. Attwell, David, and S. B. Laughlin. “An Energy Budget for Signalling in the Grey Matter of the Brain.” Journal of Cerebral Blood Flow & Metabolism 21, no. 10 (2001): 1133–45. https://doi.org/10.1097/00004647-200110000-00001.
26. Virtanen, Pauli, Ralf Gommers, Travis E. Oliphant, et al. “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.” Nature Methods 17, no. 3 (2020): 261–72. https://doi.org/10.1038/s41592-019-0686-2.
27. Hunter, J. D. “Matplotlib: A 2D Graphics Environment.” Computing in Science & Engineering 9, no. 3 (2007): 90–95. https://doi.org/10.1109/MCSE.2007.55.
28. Harris, Charles R., K. J. Millman, S. J. van der Walt, et al. “Array Programming with NumPy.” Nature 585, no. 7825 (2020): 357–62. https://doi.org/10.1038/s41586-020-2649-2.