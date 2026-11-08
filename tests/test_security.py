"""Sécurité opérationnelle du dépôt (T20, T21)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = {
    "clé AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "jeton GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "clé privée": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "clé Kaggle": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "mot de passe en clair": re.compile(r"(?i)(password|passwd|secret|api_key)\s*=\s*['\"][^'\"]{6,}['\"]"),
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "models", "htmlcov"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".txt", ".toml", ".cfg", ".ini", ".json", ".sh", ".env"}


def _repo_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
            yield p


def test_no_secrets_in_repository():
    findings = []
    for p in _repo_files():
        if p.suffix not in TEXT_SUFFIXES or p.name == Path(__file__).name:
            continue
        text = p.read_text(errors="ignore")
        for label, pat in SECRET_PATTERNS.items():
            if pat.search(text):
                findings.append(f"{p.relative_to(ROOT)} : {label}")
    assert not findings, "Secrets potentiels détectés :\n" + "\n".join(findings)


def test_no_credential_files_committed():
    forbidden = {"kaggle.json", ".env", "id_rsa", "credentials.json"}
    found = [str(p.relative_to(ROOT)) for p in _repo_files() if p.name in forbidden]
    assert not found, f"Fichiers d'identifiants présents : {found}"


@pytest.mark.parametrize("pattern", ["data/creditcard.csv", "*.zip", "models/", ".env", "kaggle.json"])
def test_gitignore_protects_sensitive_and_large_files(pattern):
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert pattern in gitignore


def test_no_large_files_in_repository():
    big = [
        f"{p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} Mo)"
        for p in _repo_files()
        if p.stat().st_size > 5e6 and p.name != "creditcard.csv"
    ]
    assert not big, f"Fichiers volumineux : {big}"
