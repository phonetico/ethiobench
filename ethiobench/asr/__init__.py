"""EthioBench-ASR: the speech recognition scorer, over the shared normalization in ethiobench.normalize."""

from __future__ import annotations

from pathlib import Path

from ..normalize import config

CONTRACT_DIR = Path(__file__).parent / "contracts"


def load_contract(language: str) -> config.NormalizationContract:
    """The ASR normalization contract for `language`, from CONTRACT_DIR."""
    return config.load(language, CONTRACT_DIR)


def available_languages() -> tuple[str, ...]:
    return config.available_languages(CONTRACT_DIR)
