# EthioBench

EthioBench is a family of evaluation benchmarks for Ethiopian languages, with the scorer for each in this repository. The current benchmarks cover Amharic (`amh`) and Tigrinya (`tir`).

**[EthioBench-ASR](https://huggingface.co/datasets/phoneticoai/ethiobench-asr)** evaluates speech recognition on 36.2 hours of human-verified speech in six recording conditions. They are read speech, image description, speak-topic, multi-speaker panel, broadcast monologue, and parliamentary and radio broadcast. Language spans mark code-switching in five of the sets, including Amharic inside Tigrinya. The dataset also holds every baseline system's unedited output and its score report.

Amharic permits more than one correct spelling of some words, so a WER depends on how both sides are normalized. EthioBench publishes that normalization as a contract per language. Every score report records the contract version, a hash of its rules and a hash of the scorer code. EthioBench does not run a leaderboard.

## This repository

- `ethiobench.normalize` holds the text normalization every benchmark shares. It covers the fidel folds, numerals, abbreviations, loanword spellings and punctuation. The language tables are in `ethiobench/normalize/tables/`.
- `ethiobench.asr` is the EthioBench-ASR scorer. It applies the ASR contract to the reference and to a system's transcript, then reports WER, CER and the code-switching metrics. Its contracts are in `ethiobench/asr/contracts/`.

## Install

    pip install -e .

Python 3.10 or later, as `pyproject.toml` declares. Tested on 3.12. The runtime dependencies are PyYAML and RapidFuzz.

## Score an ASR system

Write your system's output as JSONL, one object per segment, keyed by the manifest's `segment_id`:

    {"segment_id": "amh-read-0001", "hypothesis": "..."}

Every segment in the manifest needs exactly one hypothesis. A partial file is refused. Then run:

    ethiobench-asr score \
      --language amh \
      --manifest data/amh-read/metadata.jsonl \
      --hypotheses my-system.jsonl \
      --out results/amh-read

`python -m ethiobench.asr score ...` is the same command without the console script.

`--out` is a directory, and `report.json` and `segments.jsonl` are written inside it. For a plain JSONL file, `--system NAME` sets the name the report uses. Without it the name is the file's name without its extension.

If your system writes language tags such as `[AMH]`, pass `--strip-language-tags`. It removes every bracket tag of two to four capitals wherever it occurs. Without it, each tag is scored as an inserted word. The report records whether tags were stripped.

The code-switching metrics come from the manifest's own `language_spans`, wherever a set carries them. `--numerals {to_digits,to_words,as_recorded,off}` overrides the contract's numeral rule, and the report records the override.

The toolkit also reads the format of the published system outputs, whose header names the system. For that format, pass `--systems configs/systems.yaml`. The scorer then checks the declared revision against the one that ran. Its language tags are handled as `systems.yaml` declares, so `--strip-language-tags` and `--system` are refused for it.

## The contract, and how to read it

Each language has one ASR contract, `ethiobench/asr/contracts/amh.yaml` and `tir.yaml`. Every step says what it does and why, in the order applied. A table a contract names, such as `amh_loanwords.tsv`, is read from `ethiobench/normalize/tables/`.

`python tools/contract_inventory.py amh` prints every transformation a contract applies, the resolved character folds, the filler inventory and the rules declared but not applied. Its output for each language is in `docs/contract-amh.txt` and `docs/contract-tir.txt`.

A result depends on three things the contracts differ on.

- **Character folds.** Some Amharic letters are written differently for the same sound, so two spellings of one word are both correct. WER-N treats such letters as the same. The Amharic contract declares eight sets, which give 46 folds. Tigrinya declares none, because the same letters write distinct sounds there. Its WER-N equals its WER.
- **Numerals.** Amharic references write every number in words. A number a system writes in digits is correct when it matches the number's digit form. Some digit forms were recorded with the reference and checked against the audio. For the other numbers, the parser reads the digit form from the words. Tigrinya instead rewrites number words and digits to one value on both sides. In both languages the parser covers the readings its vocabulary composes, and a reading outside it is scored as written.
- **Loanwords.** Amharic applies a table of accepted spellings under two profiles. `lenient` folds every spelling a pattern generates. `strict` folds the canonical spelling and the spellings established in edited Ethiopian news text. Both fold whole words only. Both are reported. Tigrinya has no table and applies none.

## Reporting a result

Quote the contract version with every number, and any `--numerals` override and the loanword profile. Quote also the set revision you scored. Every manifest row carries it as `reference_revision`, and the report lists it under `stamp.reference_revision`. The report also records the contract's `rules_sha256` and a hash of every scorer file, under `stamp.contract` and `stamp.scorer_code_sha256`. Two results computed under the same contract version but different scorer code can differ, so compare those hashes too.

Segment ids are numbered within a set version. Score a hypothesis file only against the manifest of the version it was made for. The published system outputs name their version in the header, and the scorer refuses one scored against another version. A plain `{segment_id, hypothesis}` file carries no version, so that check is yours to make.

Rules a contract declares but does not yet apply are listed under `pending`, and every result records them:

    amh    normalize_amharic_spelling   no agreed table of accepted written forms
    tir    normalize_loanwords          no Tigrinya variant table

## The filler inventory

Per language and per class:

    amh   amh: እ እም አ      eng: um uh     (version 1.0-provisional)
    tir   tir: እ እም        eng: um uh     (version tir-observed-1.0)

Fluent mode removes every listed filler from both sides. In verbatim mode a filler matches only another filler of its class, so an English `um` never matches an Amharic `እ`. The Amharic inventory is provisional. The Tigrinya class lists እ and እም, the fillers written in the verified references. Standalone አ also occurs as a filler in Tigrinya panel and is not listed yet. The English class is the same as Amharic's.

## Contributing

Open an issue for a rule change. Describe the spelling or reading at stake and the evidence for it. A change ships with its justification in the contract and a test.

## Licence

CC-BY-SA-4.0, the same licence as the data. See `LICENSE`.
