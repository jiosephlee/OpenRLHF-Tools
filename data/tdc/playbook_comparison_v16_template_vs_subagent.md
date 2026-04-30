# v16 Playbook Comparison

This note compares three representative tasks across:

1. the original Haydn playbook
2. the templated `v16_no_neighbor/playbook` rewrite
3. the subagent-authored `v16_no_neighbor/playbook_subagent` rewrite

The goal is to make the differences visible without dumping the full files inline.

## HIA_Hou

Files:
- Original: `/vast/home/j/jojolee/OpenRLHF-Tools/Intern-S1-recipe/playbooks/Haydn_origin/HIA_Hou.jinja2`
- Templated: `/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results/prompt_artifacts/v16_no_neighbor/playbook/HIA_Hou.md`
- Subagent: `/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results/prompt_artifacts/v16_no_neighbor/playbook_subagent/HIA_Hou.md`

What changed:
- Original: explicit legacy tool chain, hard `0–100` output, rule table with numeric caps.
- Templated: mechanical conversion to `get_features` and semantic labels while keeping most original wording.
- Subagent: same structure, but more natural prose, more explicit interpretation language, and cleaner consolidation of workflow.

Workflow excerpt:

Original:
```md
## 1) Standard workflow (use tools in this order)
1. **Standardize**
   - `standardize_smiles(...)`
2. **Compute core descriptors**
   - `compute_descriptors(...)`
3. **Ionization class at pH 7.4**
   - `classify_ionization(...)`
4. **Functional groups + substructure flags**
   - `get_functional_groups(...)`
   - `match_substructure(...)`
```

Templated:
```md
## 1) Standard workflow (use tools in this order)
Call `get_features` once and read the returned evidence as a single integrated profile rather than as separate legacy tool calls.
Start with `molecular_profile`, `ionization_and_solubility`, then use the remaining groups to confirm or soften the first impression.
```

Subagent:
```md
## 1) Standard workflow (use tools in this order)
Call `get_features` once and read the returned evidence as one integrated profile rather than as separate legacy tool calls.

Start with `molecular_profile`, then `ionization_and_solubility`, then use `structure_and_topology` and `alert_screening` to confirm or soften the first impression.
```

Scoring excerpt:

Original:
```md
- **Cap probability at 15**
- **Cap probability at 10**
- tPSA ≤ 60: **90**
- 60 < tPSA ≤ 90: **75**
```

Templated:
```md
- **treat the ceiling as low confidence (~15)**
- **treat the ceiling as very low confidence (~10)**
- tPSA ≤ 60: strong confidence (~90)
- 60 < tPSA ≤ 90: moderately strong confidence (~75)
```

Subagent:
```md
- Cap the judgment at **very low confidence (~0-15)**
- Cap the judgment at **extremely low confidence (~0-10)**
- tPSA <= 60: **very likely (~90)**
- 60 < tPSA <= 90: **likely (~75)**
```

Read:
- The templated version is closer to the original and looks like a systematic rewrite.
- The subagent version reads more like a human-authored guide and smooths the semantic confidence language.

## ClinTox

Files:
- Original: `/vast/home/j/jojolee/OpenRLHF-Tools/Intern-S1-recipe/playbooks/Haydn_origin/ClinTox.jinja2`
- Templated: `/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results/prompt_artifacts/v16_no_neighbor/playbook/ClinTox.md`
- Subagent: `/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results/prompt_artifacts/v16_no_neighbor/playbook_subagent/ClinTox.md`

What changed:
- Original: detailed preprocessing and tool-by-tool toxicity workflow with raw SMARTS examples.
- Templated: preserves nearly all structure and PMIDs, swaps in consolidated groups, and collapses raw SMARTS into motif families in some places.
- Subagent: same content family, but cleaner phrasing, better grouping of evidence, and less mechanical references to feature retrieval.

Preprocessing excerpt:

Original:
```md
## 1) Preprocessing
1. **Standardize** the input to remove salts/counterions and canonicalize tautomers
2. Always keep the **original** SMILES too
```

Templated:
```md
## 1) Preprocessing
Call `get_features` once and read the returned evidence as a single integrated profile rather than as separate legacy tool calls.
Start with `alert_screening`, `molecular_profile`, then use the remaining groups to confirm or soften the first impression.
```

Subagent:
```md
## 1) Preprocessing
Call `get_features` once and read the returned evidence as a single integrated profile.
Use the four consolidated feature groups together:
- `molecular_profile`
- `ionization_and_solubility`
- `structure_and_topology`
- `alert_screening`
```

Alert section excerpt:

Original:
```md
### 3.2 Targeted SMARTS panel (high-value toxophores)
- Epoxide: `C1OC1`
- Aziridine: `C1NC1`
- Alkyl halide (activated): `[CX4][Cl,Br,I]`
```

Templated:
```md
### 3.2 Targeted SMARTS panel (high-value toxophores)
Separate the alert readout into a few easy-to-scan motif families:
- Epoxide
- Aziridine
- Alkyl halide (activated)
```

Subagent:
```md
### 3.2 Targeted motif families with high signal
Prefer motif-family descriptions over raw SMARTS dumps.

**Direct electrophiles / hard-reactive groups**
- Epoxide / oxirane family
- Aziridine family
- Activated alkyl halides
```

Read:
- This is where the subagent version is visibly better than the template version.
- The template removes raw SMARTS, but the subagent version also improves the grouping and readability.

## SARSCoV2_3CLPro_Diamond

Files:
- Original: `/vast/home/j/jojolee/OpenRLHF-Tools/Intern-S1-recipe/playbooks/Haydn_origin/SARSCOV2_3CLPro_Diamond.jinja2`
- Templated: `/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results/prompt_artifacts/v16_no_neighbor/playbook/SARSCoV2_3CLPro_Diamond.md`
- Subagent: `/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results/prompt_artifacts/v16_no_neighbor/playbook_subagent/SARSCoV2_3CLPro_Diamond.md`

What changed:
- Original: highly procedural agent workflow with explicit standardization, descriptor calls, and raw SMARTS warhead library.
- Templated: preserves the original scoring logic but still leaks some variable-style language like `matches[...]` and `desc[...]`.
- Subagent: rewrites the same logic into motif families and consolidated feature groups more naturally than the template.

Workflow excerpt:

Original:
```md
## 1. Agent Workflow
### Step 1: Normalize Input
**Tool:** `standardize_smiles`
### Step 2: Compute Properties & Features
**Tool:** `compute_descriptors`
**Tool:** `analyze_ring_systems`
**Tool:** `classify_ionization`
### Step 3: Targeted Substructure Screen
**Tool:** `match_substructure`
```

Templated:
```md
## 1. Agent Workflow
Call `get_features` once and read the returned evidence as a single integrated profile rather than as separate legacy tool calls.
Start with `structure_and_topology`, `molecular_profile`, then use the remaining groups to confirm or soften the first impression.
```

Subagent:
```md
## 1) Standard workflow (always do this)
Call `get_features` once and treat the returned evidence as one integrated profile with four consolidated groups:
- `molecular_profile`
- `ionization_and_solubility`
- `structure_and_topology`
- `alert_screening`
```

Decision-model excerpt:

Original:
```md
* **Nitrile** (`matches["nitrile"]`): **+28**
* **Aldehyde** (`matches["aldehyde"]`): **+35**
* **Alpha-Ketoamide** (`matches["alpha_ketoamide"]`): **+25**
```

Templated:
```md
* **Nitrile** (`matches["nitrile"]`): very strong upward adjustment (~+28)
* **Aldehyde** (`matches["aldehyde"]`): very strong upward adjustment (~+35)
* **Alpha-Ketoamide** (`matches["alpha_ketoamide"]`): very strong upward adjustment (~+25)
```

Subagent:
```md
* **Nitrile warhead**: strong upward adjustment (~+28)
* **Aldehyde warhead**: very strong upward adjustment (~+35)
* **Alpha-ketoamide / ketoamide scaffold**: strong upward adjustment (~+25)
```

Read:
- This task shows the main weakness of the templated approach: it often keeps legacy variable references even after converting the tool surface.
- The subagent rewrite is cleaner because it rewrites the medicinal-chemistry logic itself, not just the wording around it.

## Bottom line

Across these examples:
- Original Haydn: richest task-specific detail, but tightly coupled to the old tool surface and hard numeric output format.
- Templated v16 playbook: fast and faithful, but still somewhat mechanical and occasionally leaks old implementation details.
- Subagent v16 playbook: most readable and best aligned to the current tool surface, especially on motif-heavy sections and specialized tasks.

The tradeoff is consistency:
- the templated set is more uniform
- the subagent set is usually more natural, but task-to-task style will vary more
