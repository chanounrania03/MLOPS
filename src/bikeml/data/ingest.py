"""Ingestion des données UCI Bike Sharing.

Étapes :
  1. download_raw()   : récupère hour.csv (source UCI, miroir en repli),
                        vérifie l'empreinte SHA-256 et écrit une fiche de provenance ;
  2. prepare_batches(): découpe l'historique pour simuler un flux de données
                        (socle 2011, puis un lot par mois de 2012) ;
  3. release_next_batch() / make_invalid_batch() : déposent un lot dans
                        data/incoming, comme le ferait un système amont.

Le téléchargement est idempotent : si le fichier est déjà présent avec la bonne
empreinte, rien n'est refait.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bikeml.config import Config, ensure_dirs, get_config, rel

log = logging.getLogger("bikeml.ingest")

INITIAL_BATCH = "0000_initial.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "velo-mlops/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_raw(cfg: Config | None = None, force: bool = False) -> Path:
    cfg = cfg or get_config()
    ensure_dirs(cfg)
    target = cfg.path("raw_dir") / cfg.data["file_name"]
    expected = cfg.data.get("expected_sha256")

    if target.exists() and not force and (not expected or sha256(target) == expected):
        log.info("Données brutes déjà présentes : %s", rel(target, cfg))
        return target

    used_url = None
    for url in (cfg.data["source_url"], cfg.data.get("mirror_url")):
        if not url:
            continue
        try:
            log.info("Téléchargement depuis %s", url)
            content = _fetch(url)
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    content = zf.read(cfg.data["file_name"])
            target.write_bytes(content)
            used_url = url
            break
        except Exception as exc:
            log.warning("Échec du téléchargement (%s) : %s", url, exc)
    if used_url is None:
        raise RuntimeError("Aucune source de données joignable (UCI et miroir).")

    digest = sha256(target)
    if expected and digest != expected:
        target.unlink()
        raise RuntimeError(f"Empreinte inattendue ({digest}) : fichier rejeté.")

    provenance = {
        "dataset": "Bike Sharing Dataset (Fanaee-T & Gama, 2013)",
        "official_page": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "downloaded_from": used_url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": cfg.data["file_name"],
        "sha256": digest,
        "license": cfg.data["license"],
        "citation": "Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble "
                    "detectors and background knowledge. Progress in AI, 2, 113-127.",
    }
    (cfg.path("raw_dir") / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Fichier brut enregistré (%s, sha256=%s…)", rel(target, cfg), digest[:12])
    return target


def prepare_batches(cfg: Config | None = None) -> list[Path]:
    """Découpe hour.csv : un socle initial + un fichier par mois après la date pivot."""
    cfg = cfg or get_config()
    ensure_dirs(cfg)
    raw = cfg.path("raw_dir") / cfg.data["file_name"]
    df = pd.read_csv(raw, parse_dates=["dteday"])
    df = df.drop(columns=["instant"])  # simple index technique, sans valeur métier

    pivot = pd.Timestamp(cfg.data["initial_period_end"])
    out_dir = cfg.path("batches_dir")
    for old in out_dir.glob("*.csv"):
        old.unlink()

    paths = []
    initial = df[df["dteday"] <= pivot]
    p = out_dir / INITIAL_BATCH
    initial.to_csv(p, index=False, date_format="%Y-%m-%d")
    paths.append(p)

    rest = df[df["dteday"] > pivot]
    for period, chunk in rest.groupby(rest["dteday"].dt.to_period(cfg.data["batch_frequency"])):
        p = out_dir / f"{period}.csv"
        chunk.to_csv(p, index=False, date_format="%Y-%m-%d")
        paths.append(p)
    log.info("%d lots préparés dans %s", len(paths), rel(out_dir, cfg))
    return paths


def _already_released(cfg: Config) -> set[str]:
    names = set()
    for key in ("incoming_dir", "accepted_dir", "quarantine_dir"):
        names |= {p.name for p in cfg.path(key).glob("*.csv")}
    return names


def release_next_batch(cfg: Config | None = None) -> Path | None:
    """Dépose le prochain lot historique (ordre chronologique) dans incoming/."""
    cfg = cfg or get_config()
    ensure_dirs(cfg)
    done = _already_released(cfg)
    for p in sorted(cfg.path("batches_dir").glob("*.csv")):
        if p.name not in done:
            dest = cfg.path("incoming_dir") / p.name
            shutil.copy(p, dest)
            log.info("Nouveau lot déposé : %s", dest.name)
            return dest
    log.info("Plus aucun lot historique à rejouer.")
    return None


def make_invalid_batch(cfg: Config | None = None, name: str = "corrupted_batch.csv") -> Path:
    """Fabrique un lot volontairement non conforme (incident contrôlé).

    Défauts injectés : valeurs manquantes, heure hors bornes, comptages négatifs,
    doublons et une colonne retirée.
    """
    cfg = cfg or get_config()
    ensure_dirs(cfg)
    src = sorted(cfg.path("batches_dir").glob("2012-*.csv"))[-1]
    df = pd.read_csv(src)
    df = df.head(300).copy()
    df.loc[0:9, "temp"] = None
    df.loc[10:14, "hr"] = 27
    df.loc[15:19, "cnt"] = -5
    df = pd.concat([df, df.head(20)])
    df = df.drop(columns=["windspeed"])
    dest = cfg.path("incoming_dir") / name
    df.to_csv(dest, index=False)
    log.info("Lot invalide déposé : %s", dest.name)
    return dest
