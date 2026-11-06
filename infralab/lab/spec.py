"""Schema e caricamento delle specifiche di esperimento (YAML/JSON).

Una spec descrive una campagna: scenario x agenti x griglia parametri x
repliche, con seed radice e cartella di output. Il runner la espande in una
lista deterministica di run.
"""
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class SpecError(ValueError):
    """Spec di esperimento non valida."""


@dataclass(frozen=True)
class AgentSpec:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_agent_config(self) -> Dict[str, Any]:
        config = {"type": self.type}
        config.update(self.params)
        return config


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    scenario: str                      # scenario_id (in scenarios.json) o path a un file JSON
    agents: List[AgentSpec]
    horizon_quarters: int = 8
    grid: Dict[str, List[Any]] = field(default_factory=dict)
    replicates: int = 1
    seed: int = 0
    output_dir: str = "results"
    output_formats: List[str] = field(default_factory=lambda: ["csv"])
    reward: Dict[str, float] = field(default_factory=dict)
    # Ambito del seed: "run" (storico: ogni run ha un seed figlio distinto) oppure
    # "replicate" (disegno appaiato: la replica k usa lo STESSO seed per ogni
    # agente e ogni punto della griglia, cosi' le differenze fra agenti/regimi
    # sono valutate sullo stesso ambiente stocastico).
    seed_scope: str = "run"

    def grid_points(self) -> List[Dict[str, Any]]:
        """Prodotto cartesiano della griglia parametri (lista di dict override)."""
        if not self.grid:
            return [{}]
        keys = sorted(self.grid.keys())
        points = []
        for values in itertools.product(*(self.grid[k] for k in keys)):
            points.append(dict(zip(keys, values)))
        return points

    @property
    def run_count(self) -> int:
        return len(self.agents) * len(self.grid_points()) * self.replicates


_VALID_FORMATS = {"csv", "parquet"}
_VALID_SEED_SCOPES = {"run", "replicate"}
_KNOWN_KEYS = {"name", "scenario", "agents", "horizon_quarters", "grid", "replicates", "seed",
               "seed_scope", "outputs", "reward"}
# Le chiavi della griglia devono essere override di SimParameters ("params.X").
_GRID_PREFIX = "params."


def _validate(raw: Dict[str, Any]) -> None:
    for key in ("name", "scenario", "agents"):
        if key not in raw:
            raise SpecError(f"Chiave obbligatoria '{key}' mancante nella spec.")
    unknown = sorted(k for k in raw if k not in _KNOWN_KEYS)
    if unknown:
        # Una chiave ignorata in silenzio e' una spec che mente: errore esplicito.
        raise SpecError(f"Chiavi non riconosciute nella spec: {unknown}. Ammesse: {sorted(_KNOWN_KEYS)}")
    scope = raw.get("seed_scope", "run")
    if scope not in _VALID_SEED_SCOPES:
        raise SpecError(f"'seed_scope' deve essere uno di {sorted(_VALID_SEED_SCOPES)} (trovato {scope!r}).")
    if not isinstance(raw["agents"], list) or not raw["agents"]:
        raise SpecError("'agents' deve essere una lista non vuota.")
    seen_ids = set()
    for agent in raw["agents"]:
        if "id" not in agent or "type" not in agent:
            raise SpecError(f"Ogni agente richiede 'id' e 'type': {agent!r}")
        if agent["id"] in seen_ids:
            raise SpecError(f"Id agente duplicato: {agent['id']!r}")
        seen_ids.add(agent["id"])
    grid = raw.get("grid") or {}
    for key, values in grid.items():
        if not key.startswith(_GRID_PREFIX):
            raise SpecError(f"Chiave griglia '{key}' non valida: usare '{_GRID_PREFIX}<campo>'.")
        if not isinstance(values, list) or not values:
            raise SpecError(f"Griglia '{key}': attesa lista non vuota di valori.")
    replicates = raw.get("replicates", 1)
    if not isinstance(replicates, int) or replicates < 1:
        raise SpecError(f"'replicates' deve essere un intero >= 1 (trovato {replicates!r}).")
    outputs = raw.get("outputs") or {}
    for fmt in outputs.get("formats", ["csv"]):
        if fmt not in _VALID_FORMATS:
            raise SpecError(f"Formato output '{fmt}' non supportato ({sorted(_VALID_FORMATS)}).")


def spec_from_dict(raw: Dict[str, Any]) -> ExperimentSpec:
    _validate(raw)
    outputs = raw.get("outputs") or {}
    return ExperimentSpec(
        name=raw["name"],
        scenario=raw["scenario"],
        agents=[AgentSpec(id=a["id"], type=a["type"], params=a.get("params", {})) for a in raw["agents"]],
        horizon_quarters=int(raw.get("horizon_quarters", 8)),
        grid={k: list(v) for k, v in (raw.get("grid") or {}).items()},
        replicates=int(raw.get("replicates", 1)),
        seed=int(raw.get("seed", 0)),
        output_dir=outputs.get("dir", os.path.join("results", raw["name"])),
        output_formats=list(outputs.get("formats", ["csv"])),
        reward=dict(raw.get("reward") or {}),
        seed_scope=str(raw.get("seed_scope", "run")),
    )


def load_spec(path: str) -> ExperimentSpec:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SpecError(f"Il file spec '{path}' non contiene un mapping YAML.")
    return spec_from_dict(raw)


def resolve_scenario(reference: str) -> Dict[str, Any]:
    """Risolve il riferimento scenario: id in scenarios.json oppure path JSON."""
    if os.path.exists(reference) and reference.endswith(".json"):
        with open(reference, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if len(data) != 1:
                raise SpecError(
                    f"Il file scenario '{reference}' contiene {len(data)} scenari: "
                    "riferirsi a un id di scenarios.json o a un file con un solo scenario."
                )
            return data[0]
        return data
    from infralab.persistence import load_scenarios

    for scenario in load_scenarios():
        if scenario.get("scenario_id") == reference:
            return scenario
    raise SpecError(f"Scenario '{reference}' non trovato (ne' come id ne' come path).")
