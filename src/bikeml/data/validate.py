"""Contrôles qualité bloquants sur un lot de données.

Chaque contrôle produit un résultat (nom, succès, détail). Si un seul contrôle
échoue, `assert_valid` lève DataValidationError : le pipeline s'arrête avant
le prétraitement et l'entraînement, et le lot part en quarantaine.

Contrôles réalisés :
  - schéma          : toutes les colonnes attendues sont présentes ;
  - volume          : nombre minimal de lignes ;
  - complétude      : taux de valeurs manquantes ;
  - unicité         : pas de doublon sur (date, heure) ;
  - types           : les colonnes numériques ne contiennent que des nombres ;
  - bornes          : chaque variable reste dans son domaine ;
  - cohérence       : casual + registered == cnt ;
  - dates           : dates lisibles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from bikeml.config import Config, get_config


class DataValidationError(Exception):
    def __init__(self, report: "ValidationReport"):
        self.report = report
        failed = "; ".join(f"{c.name}: {c.detail}" for c in report.checks if not c.passed)
        super().__init__(f"Lot non conforme -> {failed}")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    n_rows: int
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "passed": self.passed,
                "checks": [asdict(c) for c in self.checks]}


def validate_batch(df: pd.DataFrame, cfg: Config | None = None) -> ValidationReport:
    cfg = cfg or get_config()
    rules = cfg.validation
    checks: list[CheckResult] = []

    missing_cols = [c for c in rules["required_columns"] if c not in df.columns]
    checks.append(CheckResult("schema", not missing_cols,
                              f"colonnes manquantes : {missing_cols}" if missing_cols else "ok"))

    checks.append(CheckResult("volume", len(df) >= rules["min_rows"],
                              f"{len(df)} lignes (min {rules['min_rows']})"))

    present = [c for c in rules["required_columns"] if c in df.columns]
    miss_ratio = float(df[present].isna().mean().max()) if len(df) and present else 0.0
    checks.append(CheckResult("completude", miss_ratio <= rules["max_missing_ratio"],
                              f"taux max de manquants : {miss_ratio:.2%}"))

    if {"dteday", "hr"} <= set(df.columns) and len(df):
        dup_ratio = float(df.duplicated(subset=["dteday", "hr"]).mean())
    else:
        dup_ratio = 0.0
    checks.append(CheckResult("unicite", dup_ratio <= rules["max_duplicate_ratio"],
                              f"taux de doublons (date, heure) : {dup_ratio:.2%}"))

    # Types : une valeur non numérique (ex. "chaud") n'est ni manquante ni hors
    # bornes ; sans ce contrôle, elle ne serait détectée qu'au plantage de
    # l'entraînement.
    bad_types = {}
    for col in rules["ranges"]:
        if col in df.columns:
            raw = df[col]
            n_bad = int((pd.to_numeric(raw, errors="coerce").isna() & raw.notna()).sum())
            if n_bad:
                bad_types[col] = n_bad
    checks.append(CheckResult("types", not bad_types,
                              f"valeurs non numériques : {bad_types}" if bad_types else "ok"))

    out_of_range = {}
    for col, (lo, hi) in rules["ranges"].items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            bad = int(((values < lo) | (values > hi)).sum())
            if bad:
                out_of_range[col] = bad
    checks.append(CheckResult("bornes", not out_of_range,
                              f"valeurs hors bornes : {out_of_range}" if out_of_range else "ok"))

    if {"casual", "registered", "cnt"} <= set(df.columns):
        incoherent = int((df["casual"] + df["registered"] != df["cnt"]).sum())
        checks.append(CheckResult("coherence", incoherent == 0,
                                  f"{incoherent} lignes où casual + registered != cnt"))

    if "dteday" in df.columns:
        bad_dates = int(pd.to_datetime(df["dteday"], errors="coerce").isna().sum())
        checks.append(CheckResult("dates", bad_dates == 0, f"{bad_dates} dates illisibles"))

    return ValidationReport(n_rows=len(df), checks=checks)


def assert_valid(df: pd.DataFrame, cfg: Config | None = None) -> ValidationReport:
    report = validate_batch(df, cfg)
    if not report.passed:
        raise DataValidationError(report)
    return report
