# EthioBench

EthioBench is a family of evaluation benchmarks for Ethiopian languages, with the scorer for each in this repository. The current benchmarks cover Amharic (`amh`) and Tigrinya (`tir`).

**[EthioBench-ASR](https://huggingface.co/datasets/phoneticoai/ethiobench-asr)** evaluates speech recognition on 36.2 hours of human-verified speech in six recording conditions: read speech, image description, speak-topic, multi-speaker panel, broadcast monologue, and parliamentary and radio broadcast. Word-level language spans mark code-switching, including Amharic inside Tigrinya. The dataset also holds every baseline system's unedited output and its score report.

Amharic permits more than one correct spelling of some words, so a WER depends on how both sides are normalized. EthioBench publishes that normalization as a versioned contract per language, and every score names the contract version it was computed under. EthioBench does not run a leaderboard.

## This repository

- `ethiobench.normalize`: the text normalization every benchmark shares (fidel folds, numerals, abbreviations, loanword spellings, punctuation), declared per language in a versioned contract, with the language tables in `ethiobench/normalize/tables/`.
- `ethiobench.asr`: the EthioBench-ASR scorer. It applies the ASR contract to the reference and a system's transcript, then reports WER, CER and the code-switching metrics. Its contracts are in `ethiobench/asr/contracts/`.

## Install

    pip install -e .

Python 3.10 or later, as `pyproject.toml` declares; tested on 3.12. Runtime dependencies are PyYAML and RapidFuzz.

## Score an ASR system

Write your system's output as JSONL, one object per segment, keyed by the manifest's `segment_id`:

    {"segment_id": "amh-read-0001", "hypothesis": "..."}

Every segment in the manifest needs exactly one hypothesis; a partial file will not be scored as if it were whole. Then:

    ethiobench-asr score \
      --language amh \
      --manifest data/amh-read/metadata.jsonl \
      --hypotheses my-system.jsonl \
      --out results/amh-read

`python -m ethiobench.asr score ...` is the same command without the console script.

`--out` is a directory: `report.json` and `segments.jsonl` are written inside it. `--system NAME` sets the name the report uses, which otherwise is the hypothesis file's name.

If your system writes language tags in its output, such as `[AMH]`, pass `--strip-language-tags`, and it will remove every two-to-four-capital bracket tag wherever it occurs. Without it each tag is scored as an inserted word. The report will record whether tags were stripped.

The code-switching metrics are computed from the manifest's own `language_spans` wherever a set carries them, with no further input. `--numerals {to_digits,to_words,as_recorded,off}` overrides the contract's numeral direction and is recorded in the report.

The toolkit also reads the runner format the published system outputs use, with a header naming the system; for that format, pass `--systems configs/systems.yaml` and the declared revision is checked against the one that ran.

## The contract, and how to read it

Each language has one ASR contract: `ethiobench/asr/contracts/amh.yaml` and `tir.yaml`. Every step declares what it does and why, in the order applied. A table a contract names, such as `amh_loanwords.tsv`, is read from `ethiobench/normalize/tables/`.

`python tools/contract_inventory.py amh` prints every transformation a contract applies, the resolved character folds, the filler inventory and the rules declared but not applied. Its output for each language is in `docs/contract-amh.txt` and `docs/contract-tir.txt`.

A result depends on three things the contracts differ on:

- **Character folds.** Amharic declares 46. Several consonant series that merged in Amharic are still written with separate characters, therefore two spellings of one word are both correct. WER-N folds them in the metric. Tigrinya declares none because the same characters are distinct sounds there, thus WER-N equals WER.
- **Numerals.** Amharic references give every number in the words the speaker used and record the digit form beside them, which means a number is correct as those words or as that digit form. Tigrinya rewrites number words and digits to one value on both sides, which accepts any reading of a digit string.
- **Loanwords.** Amharic applies a table of accepted spellings under two profiles: `lenient` folds anything a pattern generates, `strict` only the spellings established in edited Ethiopian news text. Both are reported. Tigrinya has no table and applies none.

## Reporting a result

Every report records the contract version it was computed under. Quote it with the number, along with any `--numerals` override and the profile, and the set revision you scored, which every manifest row carries as `reference_revision` and the report records under `stamp.reference_revision`. Because segment ids are numbered within a version, score a hypothesis file only against the manifest of the version it was made for. The published system outputs name their version in the header, and the scorer refuses one scored against another version; a plain `{segment_id, hypothesis}` file carries no version. You will want to do that check yourself. Do note, that a WER without its contract version is not comparable with another one.

Rules a contract commits to but does not yet apply are listed under `pending`, and are recorded in every result:

    amh    normalize_amharic_spelling   no agreed table of accepted written forms
    tir    normalize_loanwords          no Tigrinya variant table

## The closed sets

Bracket tags in a reference. `[unclear]` and `[overlap]` carry the duration of the stretch they mark in seconds, as `SS.S`; the other three carry none:

    [unclear:SS.S] [overlap:SS.S] [laughter] [cough] [noise]

Filler inventory, per language and per class. Fillers are equivalent only within a class, so an English `um` is never an Amharic `እ`:

    amh   amh: እ እም አ      eng: um uh     (version 1.0-provisional)
    tir   tir: እ እም        eng: um uh     (version tir-observed-1.0)

The Amharic filler inventory is provisional. The Tigrinya inventory contains the fillers observed in the verified references.

## Contributing

A rule change is a contract change: it needs a version bump, a justification string saying what it does and why, and a test. Open an issue describing the spelling or reading at stake and the evidence for it before writing the rule. Scores already published under an earlier version stay valid for that version.

## Licence

CC-BY-SA-4.0, the same licence as the data. See `LICENSE`.
