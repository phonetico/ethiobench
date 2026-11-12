"""Command line entry point: `ethiobench-asr score ...` or `python -m ethiobench.asr score ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import load_contract
from . import score as scoring
from ..normalize import config

PACKAGE_DIR = Path(__file__).parents[1]   # the ethiobench package: every .py is stamped, shared and ASR


def _code_sha256() -> dict[str, str]:
    files = sorted(p for p in PACKAGE_DIR.rglob("*.py"))
    return {str(p.relative_to(PACKAGE_DIR)): scoring.sha256(p) for p in files}


def cmd_score(args: argparse.Namespace) -> int:
    contract = load_contract(args.language)
    header, hypotheses, summary = scoring.load_hypotheses(args.hypotheses)
    system_id = header["system"]["id"]

    # A runner output names its system and is checked against systems.yaml. A
    # plain hypothesis file names none, so there is nothing to check and the
    # report says so.
    declared = None
    if system_id is None:
        if args.systems:
            print(f"REFUSED: {args.hypotheses} has no header naming a system, so "
                  f"{args.systems} cannot be checked against it. Drop --systems.", file=sys.stderr)
            return 2
        system_id = args.system or args.hypotheses.stem
        # No declaration to read output_language_tags from, so the caller says.
        declared = {"output_language_tags": args.strip_language_tags, "_from": "command line"}
    elif args.strip_language_tags:
        print("REFUSED: --strip-language-tags is for hypotheses with no header; a runner "
              f"output's tags are declared in {args.systems}", file=sys.stderr)
        return 2
    else:
        if not args.systems:
            print(f"REFUSED: {args.hypotheses} names system {system_id!r}; pass --systems to "
                  f"check its declared revision", file=sys.stderr)
            return 2
        systems = yaml.safe_load(args.systems.read_text(encoding="utf-8"))["systems"]
        declared = next((s for s in systems if s["id"] == system_id), None)
        if declared is None:
            print(f"REFUSED: system {system_id!r} is not declared in {args.systems}", file=sys.stderr)
            return 2
        if declared.get("revision") != header["system"].get("revision"):
            print(f"REFUSED: {system_id} ran at revision {header['system'].get('revision')} but "
                  f"{args.systems} declares {declared.get('revision')}", file=sys.stderr)
            return 2

    with args.manifest.open(encoding="utf-8") as fh:
        manifest = [json.loads(line) for line in fh if line.strip()]
    # Published segment_ids are minted per release version, so a hypothesis
    # file made against one version can address other rows in another.
    made_for = header.get("reference_revision")
    rows_rev = sorted({r.get("reference_revision") for r in manifest if r.get("reference_revision")})
    if made_for is not None and rows_rev != [made_for]:
        print(f"REFUSED: {args.hypotheses} was made against set version {made_for}; "
              f"{args.manifest} is version {', '.join(rows_rev) or 'unlabelled'}. "
              f"Its segment_ids may address different rows.", file=sys.stderr)
        return 2
    spans = scoring.load_spans(args.spans) if args.spans else None
    try:
        report, per_segment = scoring.score(manifest, hypotheses, contract,
                                            strip_tags=bool((declared or {}).get("output_language_tags")),
                                            numerals=args.numerals, spans=spans)
    except scoring.ScoringError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    report = {
        "system": system_id,
        "stamp": {
            "contract": contract.stamp,
            "numerals": {
                "declared": next((s.direction for s in contract.layers["base"] if s.name == "numerals"), None),
                "override": args.numerals,
            },
            "manifest": str(args.manifest),
            "manifest_sha256": scoring.sha256(args.manifest),
            # The revision the rows declare, beside the hash that identifies
            # the file. A list, so a manifest mixing revisions says so.
            "reference_revision": sorted({r.get("reference_revision") for r in manifest
                                          if r.get("reference_revision")}) or None,
            "manifest_sha256_at_inference": header.get("manifest_sha256"),
            "spans": str(args.spans) if args.spans else None,
            "spans_sha256": scoring.sha256(args.spans) if args.spans else None,
            "hypotheses": str(args.hypotheses),
            "hypotheses_sha256": scoring.sha256(args.hypotheses),
            "run_complete": (summary or {}).get("complete"),
            "system_revision": header["system"].get("revision"),
            "systems_yaml_sha256": scoring.sha256(args.systems) if args.systems else None,
            "system_declaration_checked": declared is not None and "_from" not in declared,
            "language_tags_stripped": bool((declared or {}).get("output_language_tags")),
            "scorer_code_sha256": _code_sha256(),
        },
        **report,
    }
    if contract.pending:
        report["provisional"] = (
            "Scored with rules pending: " + ", ".join(f"{p.name} ({p.blocked_by})" for p in contract.pending)
            + ". Rescore when they are resolved; these numbers will move.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    with (args.out / "segments.jsonl").open("w", encoding="utf-8") as fh:
        for seg in per_segment:
            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
    m = report["metrics"]
    print(json.dumps({"system": system_id, "segments": report["segments"],
                      "fluent_wer": m["fluent"]["wer"], "fluent_wer_n": m["fluent"]["wer_n"],
                      "fluent_cer": m["fluent"]["cer"], "verbatim_wer": m["verbatim"]["wer"],
                      "provisional": bool(contract.pending)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ethiobench-asr")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score one system's stored hypotheses")
    s.add_argument("--language", required=True)
    s.add_argument("--manifest", required=True, type=Path)
    s.add_argument("--hypotheses", required=True, type=Path)
    s.add_argument("--systems", type=Path, default=None,
                   help="systems.yaml, required for a runner output and refused without one")
    s.add_argument("--system", default=None,
                   help="name for the system in the report, when the hypotheses carry no header")
    s.add_argument("--strip-language-tags", action="store_true",
                   help="remove language tags such as [AMH] (two to four capitals in brackets) "
                        "wherever they occur in a hypothesis; for systems that emit them, when "
                        "the hypotheses carry no header")
    s.add_argument("--out", required=True, type=Path,
                   help="directory; report.json and segments.jsonl are written inside it")
    s.add_argument("--numerals", choices=("to_digits", "to_words", "as_recorded", "off"), default=None,
                   help="override the contract's numeral direction; recorded in the report")
    s.add_argument("--spans", type=Path, default=None,
                   help="language spans JSONL (tools/sql/language_spans.sql); enables code-switching metrics")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except config.ContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
