"""
Smoke test for all 11 therapeutic tools.
Writes full outputs to tool_outputs.txt for easy review.

Run: conda run -n openrlhf python openrlhf/tools/test_all_tools.py
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Test molecules
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALT_SMILES = "CC(=O)O.[Na]"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "tool_outputs.txt")

RESULTS = {}
OUTPUT_LINES = []


def section(title):
    bar = "=" * 70
    OUTPUT_LINES.append(f"\n{bar}")
    OUTPUT_LINES.append(f"  {title}")
    OUTPUT_LINES.append(f"{bar}\n")


def run_test(name, fn, *args, **kwargs):
    section(name)
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        OUTPUT_LINES.append(str(result))
        OUTPUT_LINES.append(f"\n  [{elapsed:.2f}s] PASS")
        RESULTS[name] = ("PASS", elapsed)
        print(f"  [PASS] {name} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start
        OUTPUT_LINES.append(f"ERROR: {e}")
        OUTPUT_LINES.append(traceback.format_exc())
        RESULTS[name] = ("FAIL", elapsed)
        print(f"  [FAIL] {name} ({elapsed:.2f}s): {e}")


def main():
    from openrlhf.tools.therapeutic_tools import (
        get_molecule_profile,
        analyze_functional_groups,
        analyze_ring_systems,
        assess_adme_properties,
        get_3d_properties,
        screen_safety,
        find_similar_molecules,
        remove_salts,
        evaluate_arithmetic,
        get_electronic_properties,
        predict_metabolites,
        CONSOLIDATED_TOOLS,
        get_function_by_name,
    )

    # Header
    OUTPUT_LINES.append("Therapeutic Tools — Full Output Report")
    OUTPUT_LINES.append(f"Test molecule: Aspirin  {ASPIRIN}")
    OUTPUT_LINES.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Registry check
    print(f"Registry: {len(CONSOLIDATED_TOOLS)} tools")
    for t in CONSOLIDATED_TOOLS:
        fname = t["function"]["name"]
        assert get_function_by_name(fname) is not None, f"{fname} missing"

    # Run all tools
    run_test("1. get_molecule_profile", get_molecule_profile, ASPIRIN)
    run_test("2. analyze_functional_groups", analyze_functional_groups, ASPIRIN)
    run_test("3. analyze_ring_systems", analyze_ring_systems, ASPIRIN)
    run_test("4. assess_adme_properties", assess_adme_properties, ASPIRIN)
    run_test("5. get_3d_properties", get_3d_properties, ASPIRIN)
    run_test("6. screen_safety", screen_safety, ASPIRIN)
    run_test("7. find_similar_molecules", find_similar_molecules, ASPIRIN, "AMES", 3)
    run_test("8. remove_salts", remove_salts, SALT_SMILES)
    run_test("9. evaluate_arithmetic", evaluate_arithmetic, "2 + 3 * 4")
    run_test("10. get_electronic_properties", get_electronic_properties, ASPIRIN)
    run_test("11. predict_metabolites", predict_metabolites, ASPIRIN)

    # Summary
    section("SUMMARY")
    passed = sum(1 for s, _ in RESULTS.values() if s == "PASS")
    total = len(RESULTS)
    for name, (status, elapsed) in RESULTS.items():
        icon = "PASS" if status == "PASS" else "FAIL"
        OUTPUT_LINES.append(f"  [{icon}] {name}  ({elapsed:.2f}s)")
    OUTPUT_LINES.append(f"\n  {passed}/{total} passed")

    # Write file
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(OUTPUT_LINES) + "\n")
    print(f"\nFull output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
