"""Lightweight parser for Pabulib .pb files.

The .pb format is a semicolon-separated text format with three sections:
META, PROJECTS, VOTES. Section headers are single lines; each section then
has a header row of column names followed by data rows.

We parse everything into plain dataclasses so the audit does not depend on
any third-party election library. pabutools is used only as a cross-check
for rule implementations, never for parsing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Project:
    pid: str
    cost: float
    selected: Optional[int]  # 1 = historically funded, 0 = not, None = unknown
    name: str = ""
    category: str = ""
    target: str = ""
    neighborhood: str = ""


@dataclass(frozen=True)
class Vote:
    vid: str
    projects: tuple  # tuple[str, ...] approved / ranked project ids
    age: Optional[int] = None
    sex: str = ""
    neighborhood: str = ""


@dataclass
class PBInstance:
    path: str
    meta: Dict[str, str] = field(default_factory=dict)
    projects: Dict[str, Project] = field(default_factory=dict)
    votes: List[Vote] = field(default_factory=list)

    @property
    def country(self) -> str:
        return self.meta.get("country", "")

    @property
    def unit(self) -> str:
        return self.meta.get("unit", "")

    @property
    def subunit(self) -> str:
        return self.meta.get("subunit", "")

    @property
    def year(self) -> Optional[int]:
        for key in ("date_begin", "date_end"):
            raw = self.meta.get(key, "")
            for token in raw.replace(".", "-").replace("/", "-").split("-"):
                if len(token) == 4 and token.isdigit():
                    return int(token)
        instance_name = self.meta.get("instance", "")
        for token in instance_name.split("_"):
            if len(token) == 4 and token.isdigit():
                return int(token)
        return None

    @property
    def budget(self) -> float:
        try:
            return float(self.meta.get("budget", "0").replace(",", "."))
        except ValueError:
            return 0.0

    @property
    def vote_type(self) -> str:
        return self.meta.get("vote_type", "")

    @property
    def rule(self) -> str:
        return self.meta.get("rule", "")

    def series_key(self) -> str:
        """Longitudinal series identifier: same place, different years."""
        return f"{self.country}/{self.unit}/{self.subunit or 'CITYWIDE'}"


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def parse_pb_file(path: Path) -> PBInstance:
    inst = PBInstance(path=str(path))
    section = None
    header: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            upper = line.strip().upper()
            if upper in ("META", "PROJECTS", "VOTES"):
                section = upper
                header = []
                continue
            fields = line.split(";")
            if section == "META":
                if len(fields) >= 2 and fields[0] != "key":
                    inst.meta[fields[0].strip()] = fields[1].strip()
            elif section == "PROJECTS":
                if not header:
                    header = [f.strip() for f in fields]
                    continue
                row = dict(zip(header, fields))
                pid = row.get("project_id", "").strip()
                if not pid:
                    continue
                try:
                    cost = float(row.get("cost", "0").replace(",", "."))
                except ValueError:
                    cost = 0.0
                inst.projects[pid] = Project(
                    pid=pid,
                    cost=cost,
                    selected=_parse_int(row.get("selected", "")),
                    name=row.get("name", ""),
                    category=row.get("category", ""),
                    target=row.get("target", ""),
                    neighborhood=row.get("neighborhood", ""),
                )
            elif section == "VOTES":
                if not header:
                    header = [f.strip() for f in fields]
                    continue
                row = dict(zip(header, fields))
                vid = row.get("voter_id", "").strip()
                raw_vote = row.get("vote", "")
                projs = tuple(p for p in raw_vote.split(",") if p)
                inst.votes.append(
                    Vote(
                        vid=vid,
                        projects=projs,
                        age=_parse_int(row.get("age", "")),
                        sex=row.get("sex", "").strip().upper(),
                        neighborhood=row.get("neighborhood", "").strip(),
                    )
                )
    return inst
