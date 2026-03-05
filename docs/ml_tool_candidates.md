# ML Tool Candidates for TDC Molecular Property Prediction

Ranked list of ML-based molecular property tools that provide information **not available from RDKit** and are **not themselves TDC tasks** (avoiding answer leakage).

**Current toolset (v4):** RDKit descriptors (14 basic + 26 task-specific), AccFG functional groups, pKa (MolGpKa), LogD (Henderson-Hasselbalch), structural alerts (PAINS/BRENK/ChEMBL), Haydn tools (similarity, MCS, pharmacophore, ionization, scaffolds, ring systems).

## Ranking Criteria

- **Novelty**: Does this provide information completely absent from the current toolset?
- **TDC Relevance**: How many TDC tasks benefit from this signal?
- **Mechanistic Value**: Does it help the LLM *reason* about why a molecule has a property, rather than just predicting the answer?
- **Integration Effort**: How easy is it to pip install and wrap?

---

## Rank 1: Synthesizability Score (RAscore)

**What:** ML-learned retrosynthesis feasibility score (0-1). Trained on 200K ChEMBL compounds labeled by whether AiZynthfinder could find a synthesis route.

**Why it matters:** SA score from RDKit is a crude heuristic. RAscore captures actual synthetic feasibility — macrocycles, complex heterocycles, and strained systems are scored correctly. Helps the LLM reason about whether suggested molecular modifications are practical.

**TDC tasks helped:** All tasks indirectly (drug-likeness reasoning, filtering implausible molecules).

**Current coverage:** Zero. RDKit has SA score but it's heuristic, not ML.

**Package:** `pip install rascore` or install from [github.com/reymond-group/RAscore](https://github.com/reymond-group/RAscore). MIT license. XGBoost + NN variants. Input: SMILES -> Output: float [0, 1].

---

## Rank 2: Metabolic Site Prediction

**What:** Predicts *which atoms* on a molecule are metabolized by specific CYP isoforms (CYP3A4, CYP2D6, CYP2C9, etc.). Returns per-atom metabolic vulnerability scores and the predicted sites of metabolism (SOMs).

**Why it matters:** This is fundamentally different from predicting clearance (a TDC task). It gives the LLM mechanistic *why* information: "the methyl group at position 4 is a CYP3A4 soft spot" vs "this molecule has high clearance." Useful for reasoning about toxicity (reactive metabolites), half-life, and drug-drug interactions.

**TDC tasks helped:** Clearance (hepatocyte + microsomal), half-life, CYP inhibition/substrate, DILI, LD50.

**Current coverage:** Zero. No metabolic site information in any tool version.

**Options:**
- **FAME 3** (Fast Assessment of Metabolic sites using Extended fingerprints): Random forest, predicts SOMs for CYP1A2/2A6/2B6/2C8/2C9/2C19/2D6/2E1/3A4 + UGT/FMO/NAT/MAO. Published model, requires Java/Python bridge. [DOI: 10.1021/acs.jcim.9b00376](https://doi.org/10.1021/acs.jcim.9b00376)
- **XenoSite** (deep learning): CNN on molecular graphs for CYP metabolism, epoxidation, reactivity. Web API at [swami.wustl.edu/xenosite](https://swami.wustl.edu/xenosite). Python package available but less maintained.
- **SyGMa** (`pip install sygma`): Rule-based metabolite prediction (phase I + II). Not ML but complementary — predicts what metabolites form, not just where.
- **GLORYx**: Predicts metabolites with CYP/UGT site-of-metabolism. [GitHub](https://github.com/christinadebruynkops/GLORYx). Java-based.

**Recommended approach:** SyGMa for metabolite enumeration (pip-installable, lightweight) + a custom wrapper that identifies the most labile positions using RDKit reactivity SMARTS as a fallback if FAME 3 is too heavy.

---

## Rank 3: Electronic Properties (HOMO/LUMO, Electrophilicity)

**What:** Frontier molecular orbital energies (HOMO, LUMO), electrophilicity index, chemical hardness/softness, dipole moment. These are quantum-chemical properties that govern reactivity.

**Why it matters:** Electrophilicity correlates with reactive metabolite formation (→ DILI, Ames mutagenicity). HOMO-LUMO gap indicates chemical stability. Dipole moment affects membrane permeability and protein binding. None of these are in RDKit.

**TDC tasks helped:** AMES (mutagenicity via electrophilic reactivity), DILI (reactive metabolites), hERG (electronic interactions with channel), toxicity endpoints broadly.

**Current coverage:** Zero. RDKit has Gasteiger partial charges but no orbital energies.

**Options:**
- **GFN2-xTB** (`pip install xtb-python` or conda): Semi-empirical tight-binding DFT. Fast (~1s per molecule). Returns HOMO, LUMO, dipole, Mulliken charges, Wiberg bond orders, Fukui indices. Well-validated. Open source (LGPL-3.0).
- **ANI-2x** (`pip install torchani`): Neural network potential trained on DFT data. Very fast for conformer energies and strain, but doesn't directly output orbital energies.
- **MACE-OFF** (`pip install mace-torch`): Equivariant GNN potential. State-of-the-art for conformer energies. Same limitation as ANI-2x for orbital properties.
- **Auto3D + xtb pipeline**: Generate 3D conformer with Auto3D, then compute electronic properties with xtb.

**Recommended:** GFN2-xTB — it's the only option that gives orbital energies (HOMO/LUMO/Fukui) directly, is fast enough for online use, and is pip-installable.

---

## Rank 4: Conformational Strain Energy

**What:** The energy penalty a molecule pays to adopt its "bioactive" conformation vs its lowest-energy conformer. High strain = poor binding efficiency, unexpected reactivity.

**Why it matters:** A molecule might have perfect 2D descriptors but be too rigid or too strained to bind well. Strain energy is completely invisible to 2D descriptors.

**TDC tasks helped:** Bioavailability, lipophilicity (conformer-dependent partitioning), BBB (flexible molecules cross differently).

**Current coverage:** Partial (v3 has 3D EPSA from freesasa, but no energy calculations).

**Options:**
- **ANI-2x** (`pip install torchani`): Fast conformer energy evaluation. Generate N conformers with RDKit ETKDG, minimize with ANI-2x, report energy spread.
- **GFN2-xTB**: Slightly slower but more robust for heterocycles and charged species.
- **RDKit MMFF/UFF + ANI-2x**: Use RDKit force field for quick conformer generation, then ANI-2x for accurate single-point energies.

**Recommended:** ANI-2x for energy, RDKit ETKDG for conformer generation.

---

## Rank 5: Abraham Solute Descriptors (A, B, S, E, V, L)

**What:** Six physicochemical descriptors that characterize a molecule's hydrogen bond acidity (A), basicity (B), dipolarity/polarizability (S), excess molar refraction (E), McGowan volume (V), and hexadecane/air partition (L).

**Why it matters:** These are the fundamental parameters in Linear Solvation Energy Relationships (LSERs), used to predict partitioning across *any* two phases (water/octanol, water/membrane, blood/brain). Much richer than just logP or HBD/HBA counts.

**TDC tasks helped:** BBB (blood-brain partitioning is literally an LSER application), solubility, lipophilicity, permeability-related tasks.

**Current coverage:** Partial overlap — logP and HBD/HBA counts are crude proxies. But Abraham descriptors decompose these into orthogonal contributions.

**Options:**
- **SoluteML**: ML prediction of Abraham descriptors from SMILES. Research code.
- **UFZ-LSER database + ML fill-in**: Database of experimental values with ML gap-filling.
- **COSMO-RS**: Gold standard but requires commercial license (COSMOtherm).

**Harder to integrate than ranks 1-4; less mature pip packages.**

---

## Rank 6: Tautomer Stability Ranking

**What:** Given a molecule, enumerate tautomers and rank by thermodynamic stability (which tautomer is dominant at physiological pH).

**Why it matters:** Different tautomers have different logP, pKa, and pharmacophore features. RDKit can enumerate tautomers but can't rank them. The dominant tautomer may have very different properties than the input SMILES.

**TDC tasks helped:** Any task where the input SMILES might not represent the dominant tautomer (common for heterocycles with NH/OH groups).

**Current coverage:** RDKit `TautomerEnumerator` exists but has no energy ranking.

**Options:**
- **GFN2-xTB**: Enumerate with RDKit, rank with xtb single-point energies. Straightforward pipeline.
- **Tautobase + ML**: Experimental tautomer equilibria database with ML models.

**Recommended:** Combine with Rank 3 (xtb) — if you're already running xtb for electronic properties, tautomer ranking is essentially free.

---

## Rank 7: Reactive Metabolite / Covalent Binding Alerts

**What:** Beyond structural alerts (PAINS/BRENK already in v4), predict whether a molecule forms reactive metabolites (epoxides, quinones, acyl glucuronides) that covalently bind to proteins.

**Why it matters:** Reactive metabolites are the primary mechanism for idiosyncratic DILI and some forms of mutagenicity. Structural alerts catch known patterns; ML models catch novel ones.

**TDC tasks helped:** DILI, AMES, LD50, carcinogenicity.

**Current coverage:** Partial — `score_structural_alerts` catches known PAINS/BRENK patterns. But doesn't predict metabolic activation.

**Options:**
- **XenoSite reactivity module**: Predicts sites of epoxidation and reactivity.
- **SyGMa** + electrophilicity filter: Enumerate metabolites, then check electrophilicity of each.

**Lower priority because structural alerts already cover the most common patterns.**

---

## Summary Table

| Rank | Tool | Key Property | TDC Tasks Helped | Integration Effort | Package |
|------|------|-------------|-----------------|-------------------|---------|
| 1 | RAscore | Synthesizability | All (reasoning) | Easy | `pip install rascore` |
| 2 | FAME 3 / SyGMa | Metabolic sites | Clearance, half-life, DILI, CYP | Medium | SyGMa: `pip install sygma` |
| 3 | GFN2-xTB | HOMO/LUMO, electrophilicity | AMES, DILI, hERG, toxicity | Medium | `conda install xtb-python` |
| 4 | ANI-2x | Conformational strain | Bioavailability, BBB | Medium | `pip install torchani` |
| 5 | SoluteML | Abraham descriptors (A,B,S,E,V,L) | BBB, solubility, lipophilicity | Hard | Research code |
| 6 | GFN2-xTB | Tautomer ranking | All (correctness) | Free w/ Rank 3 | Same as Rank 3 |
| 7 | XenoSite/SyGMa | Reactive metabolites | DILI, AMES, LD50 | Medium | SyGMa: `pip install sygma` |

## Recommended v5 Toolset

**v5 = v4 + RAscore + metabolic_sites + electronic_properties**

These three provide the highest-value non-cheating information currently absent from the toolset.
