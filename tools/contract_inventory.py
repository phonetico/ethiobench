"""Print every transformation a contract applies, in the order applied.

Every step, every equivalence set, the resolved fold character by character,
the filler inventory, and what is declared but not applied.

    python3 tools/contract_inventory.py amh
"""

import sys
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ethiobench.asr import CONTRACT_DIR, load_contract           # noqa: E402
from ethiobench.normalize.pipeline import fold_mapping           # noqa: E402

CONFIGS = CONTRACT_DIR


def wrap(s, width=74, indent=6):
    words = " ".join((s or "").split()).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(" " * indent + l for l in lines)


def main(language, configs=CONFIGS):
    # Prose comes from the file at `configs`, rules from the installed
    # contract. The release build passes its redacted copy, so the published
    # inventory quotes the published contract and not the internal one.
    raw = yaml.safe_load((pathlib.Path(configs) / f"{language}.yaml").read_text())
    contract = load_contract(language)
    folds = fold_mapping(contract)

    bar = "=" * 78
    print(bar)
    print(f"  {language}.yaml  version {raw['version']}")
    if raw.get("governed_by"):
        print(f"  governed by {raw['governed_by']}")
    print(bar)

    for layer in ("base", "fluent", "verbatim"):
        steps = (raw.get("pipeline") or {}).get(layer) or []
        print(f"\n{'-' * 78}\n{layer.upper()}  ({len(steps)} steps, applied in this order)\n{'-' * 78}")
        for i, s in enumerate(steps, 1):
            print(f"\n  {i}. {s.get('step')}")
            print(wrap(s.get("justification")))

    sets = (raw.get("wer_n") or {}).get("equivalence_sets") or []
    print(f"\n{'-' * 78}\nWER-N EQUIVALENCE LAYER  ({len(sets)} sets, {len(folds)} folds)\n{'-' * 78}")
    for s in sets:
        print(f"\n  {s['name']}")
        print(f"      members  {' '.join(s.get('members') or [])}")
        if s.get("orders"):
            print(f"      orders   {'  '.join(s['orders'])}")
        if s.get("unfolded"):
            print(f"      UNFOLDED {' '.join(s['unfolded'])}  (declared and deliberately not folded)")
        print(wrap(s.get("justification")))

    print(f"\n{'-' * 78}\nTHE RESOLVED FOLD TABLE  ({len(folds)} characters change)\n{'-' * 78}")
    items = sorted(folds.items())
    for i in range(0, len(items), 6):
        print("   " + "    ".join(f"{a} -> {b}" for a, b in items[i:i + 6]))

    fi = raw.get("filler_inventory") or {}
    print(f"\n{'-' * 78}\nFILLER INVENTORY  (version {fi.get('version')})\n{'-' * 78}")
    for cls, toks in (fi.get("classes") or {}).items():
        print(f"   {cls}: {' '.join(toks)}")
    print(wrap(f"Equivalence holds inside a class, never across. {fi.get('source')}"))

    lw = raw.get("loanwords") or {}
    if lw:
        print(f"\n{'-' * 78}\nLOANWORD LAYER\n{'-' * 78}")
        print(f"   step     {lw.get('step')}")
        print(f"   table    {lw.get('table')}")
        print(f"   profiles {lw.get('profiles')}")
        print(wrap(lw.get("justification")))

    pend = raw.get("pending") or []
    print(f"\n{'-' * 78}\nDECLARED BUT NOT APPLIED  ({len(pend)})\n{'-' * 78}")
    if not pend:
        print("   none")
    for s in pend:
        print(f"\n   {s.get('step')}   blocked_by {s.get('blocked_by')}")
        print(wrap(s.get("justification")))
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "amh",
         sys.argv[2] if len(sys.argv) > 2 else CONFIGS)
