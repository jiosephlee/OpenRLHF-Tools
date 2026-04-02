# SFT Trace Guide

Cold-start SFT traces for RL training — multi-turn tool-calling conversations showing how a model should reason through TDC binary classification tasks using cheminformatics tools.

## Trace Structure

Each trace is a JSON file with this structure:

```json
{
  "task": "AMES",
  "smiles": "...",
  "label": 0,
  "answer": "(A)",
  "messages": [ ... ]
}
```

### Message Flow

1. **User prompt** — task instruction with SMILES
2. **Assistant turn 1** — brief structural recognition + 4 parallel tool calls
3–6. **Tool results** — `get_molecule_profile`, `screen_structural_alerts`, `find_similar_molecules`, `analyze_functional_groups`
7. **Assistant turn 2** — analysis of turn 1 results + transition + N parallel tool calls (2–3 depending on task)
8–(7+N). **Tool results** — task-specific tools (see below)
(8+N). **Assistant turn 3** — final reasoning with explicit evidence weighing (`"channel": "final"`)
(9+N). **Assistant turn 4** — `"Answer: (X)"` (`"channel": "final"`)

Total messages: 11 (2 turn-2 tools) or 12 (3 turn-2 tools).

### GPT-OSS Harmony Format

The last two assistant messages (final reasoning + answer) must include `"channel": "final"`. This is required by the GPT-OSS / Harmony chat protocol.

## Tool Selection by Task

### Tool Tiers

Not every tool needs to be called in every trace. Use the following tiers to guide tool selection:

| Tier | Tools | When to call |
|------|-------|-------------|
| **Always** | `get_molecule_profile`, `analyze_functional_groups`, `find_similar_molecules`, `assess_adme_properties` | Every trace. These are foundational — you always need to know what the molecule is, what groups it has, what similar molecules do, and its ionization/LogD at physiological pH. |
| **Very frequent** | `screen_structural_alerts`, `analyze_ring_systems` | Most ADMET tasks. Alerts are essential for toxicity tasks (AMES, DILI) and informative for others. Ring systems matter when planarity (BBB), intercalation (AMES), or active site fit (CYP) is relevant. May skip for tasks where they add no signal (e.g., solubility). |
| **Sometimes** | `predict_metabolites` | When metabolism is part of the mechanism: DILI (bioactivation), CYP substrate (what the enzyme converts it into), AMES (pro-mutagen activation). Not needed for BBB or purely physicochemical tasks. |
| **Rare / situational** | `get_3d_properties`, `remove_salts`, `evaluate_arithmetic`, `get_scaffold` | `remove_salts` when SMILES contains counterions. `get_3d_properties` for shape-dependent tasks. These are not standard in traces. |

**Note:** `get_electronic_properties` is no longer in the default tool set — its data (Gasteiger charges, xTB properties) is included in `get_molecule_profile`.

### Current Trace Layout

Turn 1 calls the "always" tools plus `screen_structural_alerts` (for all current tasks):
1. `get_molecule_profile`
2. `screen_structural_alerts`
3. `find_similar_molecules(smiles, task, k=5)`
4. `analyze_functional_groups`

Turn 2 adds task-specific tools:

| Task | Turn 2 Tools | Rationale |
|------|-------------|-----------|
| AMES | `assess_adme_properties`, `analyze_ring_systems` | Ionization/bioavailability + intercalation check |
| DILI | `predict_metabolites`, `assess_adme_properties` | Bioactivation pathways + ionization/permeability |
| BBB_Martins | `assess_adme_properties`, `analyze_ring_systems` | Ionization at pH 7.4 (critical) + ring planarity |
| CYP3A4_Substrate | `assess_adme_properties`, `analyze_ring_systems`, `predict_metabolites` | Ionization + active site fit + CYP reaction confirmation |

The number of turn-2 tools is flexible (2–3). `assess_adme_properties` appears in both turns for some layouts because it is an "always" tool — when it's already in turn 1, it doesn't need to repeat in turn 2 (current traces call it in turn 2 after the first 4 tools).

## Reasoning Conventions

### Source Attribution
Every claim in the reasoning must be attributed to its source:
- `*From toxicophore screen:*` — structural alert results
- `*From neighbor analysis:*` — KNN similarity data
- `*From metabolism prediction:*` — GLORYx metabolite predictions
- `*From ADME:*` — pKa, logD, solubility data
- `*From electronic properties:*` — Gasteiger charges, xTB data
- `*From functional groups:*` — AccFG decomposition
- `*From chemical knowledge:*` — domain expertise not derivable from tools

### Evidence Weighing Structure
The final reasoning (message 10) should follow this structure:
1. Interpret each turn 2 tool result
2. **"Weighing the evidence:"** section with explicit **Evidence for (B):** and **Evidence against / Evidence for (A):** subsections
3. Each bullet under these sections has an italicized source attribution
4. **"Assessment:"** paragraph with the final judgment and confidence level

### Calibrated Uncertainty
- High-confidence cases: "High-confidence prediction. [converging evidence summary]."
- Hard cases: "This is a close call. [evidence conflict summary]. I lean toward X, but with low confidence."
- State when neighbor majority disagrees with your prediction and explain why you override it.

## Key Lessons Learned

### Structural alerts don't correlate with risk
Alert count is not predictive. Examples from our traces:
- **Dopamine**: 6 alert categories but DILI-negative (endogenous molecule with dedicated clearance)
- **Ibuprofen**: 1 alert category but DILI-positive (acyl glucuronide mechanism)
- **Aspirin**: 5 alert categories but DILI-negative (rapid ester hydrolysis prevents bioactivation)

### Metabolism tools vs structural alerts have complementary strengths
- **Metabolism tools** capture *quantitatively dominant* pathways (high-flux reactions)
- **Structural alerts** catch *qualitatively dangerous* minor pathways
- Example: Sulfanilamide — metabolism tool predicts Phase 2 detoxification (N-acetylation, score 0.956) but misses the minor CYP2C9 N-hydroxylation that causes DILI. Structural alerts correctly flag this.

### `get_molecule_profile` includes electronic properties
The profile tool includes Gasteiger charges, charge polarization, EState indices, and full GFN2-xTB quantum properties (HOMO, LUMO, gap, dipole, electrophilicity index). The standalone `get_electronic_properties` is no longer in the default tool set. Turn 1 reasoning should reference electronic data from the profile where relevant.

### `predict_metabolites` (formerly `predict_metabolism_sites`)
Predicts full metabolite structures with reaction types and priority scores (GLORYx + SyGMa fallback). Most relevant for DILI (bioactivation pathways) and CYP substrate tasks (what the enzyme converts the molecule into).

### `find_similar_molecules` includes basic properties per neighbor
Each neighbor now shows `Basic properties: MW=..., logP=..., TPSA=...` alongside functional groups. Reference these when they add insight (e.g., comparing neighbor lipophilicity patterns).

### `analyze_ring_systems` is condensed
Output is ~6 lines. Reasoning should match: "Single benzene ring, no PAH or fused systems" rather than quoting verbose fields.

### `assess_adme_properties` shows atom-specific pKa
Format: `Acidic sites (N): atom X pKa = Y, ...` and `Ionization at pH 7.4: class, charge N`. Reference `Dominant form` SMILES and `Ambiguous` flag where relevant.

## Molecule Selection Guidelines

For each task, select a diverse set covering:
- **Both labels** — approximately balanced A and B examples
- **Easy cases** (tools converge, high confidence) and **hard cases** (tools disagree, requires chemistry knowledge)
- **Different structural motifs** — avoid clustering around one scaffold
- **Pedagogically valuable cases** — molecules that teach the model something about when to trust/distrust specific tools

### Good hard cases teach specific lessons:
- When to override neighbor majority with mechanistic reasoning
- When structural alerts are false positives (e.g., catechol in dopamine)
- When structural alerts catch what metabolism tools miss (e.g., sulfanilamide N-hydroxylation)
- Positional isomers with different outcomes (e.g., 2,4- vs 2,6-dichloroaniline)

## Generating Tool Outputs

Tool outputs must come from actually running the tools, not from fabrication. Use:

```python
import sys
sys.path.insert(0, '/vast/home/j/jojolee/OpenRLHF-Tools')
from openrlhf.tools.therapeutic_tools import _FUNCTION_MAP

fn = _FUNCTION_MAP['tool_name']
result = fn(smiles)  # or fn(smiles, task='TASK', k=5) for find_similar_molecules

# Format as trace content string:
import json
content = json.dumps({
    "result": result,
    "function_name": "tool_name",
    "arguments": {"smiles": smiles}
})
```

Run with: `/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python`

## When Tools Are Redefined

If the underlying tool implementations change:
1. Re-run all tools for all trace molecules (batch script)
2. Replace tool result `content` strings in all traces
3. Diff old vs new outputs to identify what changed
4. Update reasoning text to match new output format and content
5. Verify all traces: valid JSON, tool results match, answers match labels

## File Naming Convention

`{TASK}_trace_{descriptor}.json`

Examples: `AMES_trace_nitroso.json`, `DILI_trace_phenacetin.json`
