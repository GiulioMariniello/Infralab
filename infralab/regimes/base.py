"""Regime normativo: la legge come variabile dell'ambiente.

Un `RegulatoryRegime` descrive, in forma leggibile dalla macchina, la legge che
governa il parco asset:

- **obblighi** (`obligations`): trigger -> azione obbligatoria (es. ispezione
  scaduta -> L1_inspect; CdA Alta senza L4 recente -> L4_verify; rischio sopra
  soglia -> L4_verify; asset non censito -> L0_census);
- **enforcement** (`enforcement.mode`): cosa succede se l'obbligo non viene
  assolto dal gestore:
    * ``none``     : nulla, solo segnale nell'osservazione (comportamento storico);
    * ``forced``   : l'azione obbligatoria viene eseguita d'ufficio e addebitata
                     al gestore (anche a budget negativo: la legge costa);
    * ``sanction`` : misura operativa progressiva (limitazione di carico, senso
                     alternato, chiusura) finché l'obbligo non è assolto; le
                     azioni del gestore che rimuovono la misura sono rifiutate;
    * ``both``     : esecuzione d'ufficio + sanzione se comunque inadempiente;
- **sanzioni pecuniarie** (`enforcement.fine_per_quarter_m2`): multa per
  trimestre di inadempienza, proporzionale all'area;
- **tier di target** (`target_tier`): normativa dei target di affidabilità da
  imporre (override di `code_framework`), es. NEN 8700 per strutture esistenti;
- **regole informative** (`information.census_required`): senza censimento
  l'asset non è gestibile (già nel motore) e il censimento diventa obbligo;
- **regole sulle risorse** (`resources.earmarked_fraction`): quota del budget
  riservata ad azioni di analisi (ispezioni/valutazioni): le altre azioni non
  possono intaccarla;
- **override di parametri** (`param_overrides`): qualunque campo di
  SimParameters che la legge fissa (es. `l4_escalation_quarters`,
  `cda_multi_hazard`, `inspection_overdue_grace_quarters`).

Il regime ``none`` è un passthrough: con esso il motore si comporta esattamente
come prima (snapshot dorato invariato).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Trigger riconosciuti dal motore (valutati per asset, ogni trimestre).
TRIGGERS = (
    "inspection_overdue",      # quarter > next_inspection_quarter + grace
    "l4_escalation_overdue",   # CdA Alta senza L4 da piu' di l4_escalation_quarters
    "risk_above_threshold",    # risk_priority_index > obligation.threshold
    "beta_below_target",       # beta annuale (posterior se c'e') < target annuale
    "pf_above_threshold",      # Pf annuale (posterior se c'e') > obligation.threshold
    "condition_below_threshold",  # min condizione OSSERVATA < obligation.threshold
    "not_censused",            # asset non censito (L0)
    "network_relevance_above_threshold",  # rilevanza di rete > obligation.threshold (percentile 0-1)
    # Ponte esposto nell'area di un evento estremo regionale (pubblico) e non ancora
    # ispezionato dopo l'evento; threshold = intensita' locale minima (None = tutta
    # l'area). Ispezione straordinaria dopo eventi eccezionali (Linee Guida 2020).
    # Richiede regional_events_enabled (altrimenti non scatta mai).
    "regional_event_in_area",
)
ENFORCEMENT_MODES = ("none", "forced", "sanction", "both")


@dataclass(frozen=True)
class Obligation:
    """Un obbligo di legge: quando `trigger` e' vero, `action` e' dovuta."""
    trigger: str
    action: str
    threshold: Optional[float] = None       # per risk_above_threshold / condition_below_threshold
    grace_quarters: int = 0                 # trimestri di tolleranza prima dell'enforcement
    charged_to: str = "owner"               # "owner" (default) o "state" (a carico pubblico: non addebitata)
    # Come si assolve l'obbligo:
    #  "state"  : quando il trigger smette di essere vero (es. riparare fa risalire
    #             la condizione sopra la soglia). E' il comportamento storico.
    #  "action" : eseguendo l'azione dovuta, che vale per `recurrence_quarters`
    #             trimestri anche se il trigger resta vero. E' il regime di
    #             sorveglianza: il ponte ammalorato resta in esercizio PERCHE'
    #             viene ispezionato con la cadenza prescritta.
    discharge: str = "state"
    recurrence_quarters: int = 0            # solo con discharge="action": validita' dell'adempimento
    # Modalita' di applicazione di QUESTO obbligo ("forced", "sanction", "both", "none").
    # None (default) = quella della legge (`enforcement.mode`), comportamento storico.
    # Serve a scrivere una legge con obblighi di tipo diverso: per esempio un'ispezione
    # eseguita d'ufficio accanto a un obbligo di intervento fatto valere con le sanzioni.
    enforcement: Optional[str] = None

    def __post_init__(self):
        if self.enforcement is not None and self.enforcement not in ENFORCEMENT_MODES:
            raise ValueError(f"obligation.enforcement '{self.enforcement}' non valido. Validi: {ENFORCEMENT_MODES}")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"Trigger '{self.trigger}' non riconosciuto. Validi: {TRIGGERS}")
        if self.charged_to not in ("owner", "state"):
            raise ValueError("charged_to deve essere 'owner' o 'state'")
        if self.discharge not in ("state", "action"):
            raise ValueError("discharge deve essere 'state' o 'action'")
        if self.discharge == "action" and int(self.recurrence_quarters) <= 0:
            raise ValueError("con discharge='action' serve recurrence_quarters > 0 "
                             "(ogni quanti trimestri va ripetuta l'azione)")


# Grandezze osservabili su cui una legge puo' costruire la classificazione. Sono tutte
# ricavabili da cio' che il gestore e l'autorita' possono davvero sapere: evidenza
# ispettiva, dati di inventario, attributi di rete. Nessuna legge puo' leggere lo stato
# vero della struttura.
CLASSIFIER_INPUTS = (
    "observed_condition",        # condizione minima osservata (evidenza ispettiva)
    "observed_beta_margin",      # beta annuo stimato meno target annuo di norma
    "quarters_since_inspection", # anzianita' dell'informazione
    "network_relevance",         # percentile di esposizione (traffico x deviazione)
    "adt",                       # traffico giornaliero medio
    "detour_km",                 # lunghezza della deviazione in caso di chiusura
    "age_years",                 # eta' dell'opera
    "area_m2",                   # superficie di impalcato
    "crosses_water",             # attraversa un corso d'acqua (0/1)
    "scour_critical",            # fondazione a rischio erosione (0/1)
    "restricted",                # gia' soggetto a limitazione d'uso (0/1)
)
COMPARATORS = ("lt", "le", "gt", "ge", "eq", "ne")
CDA_CLASSES = ("Bassa", "Medio-Bassa", "Media", "Medio-Alta", "Alta")


@dataclass(frozen=True)
class ClassificationRule:
    """Una riga della tabella di classificazione di una legge: se TUTTE le condizioni
    sono vere, l'opera ricade nella classe indicata. Le righe si valutano in ordine e
    vince la prima che si applica, come in una tabella normativa."""
    when: Dict[str, Dict[str, float]]
    cda: str

    def __post_init__(self):
        if self.cda not in CDA_CLASSES:
            raise ValueError(f"Classe '{self.cda}' non valida. Valide: {CDA_CLASSES}")
        for name, cond in self.when.items():
            if name not in CLASSIFIER_INPUTS:
                raise ValueError(f"Grandezza '{name}' non osservabile. Disponibili: {CLASSIFIER_INPUTS}")
            if not isinstance(cond, dict) or not cond:
                raise ValueError(f"Condizione su '{name}' vuota o malformata")
            for op in cond:
                if op not in COMPARATORS:
                    raise ValueError(f"Operatore '{op}' non valido. Validi: {COMPARATORS}")

    def matches(self, values: Dict[str, float]) -> bool:
        for name, cond in self.when.items():
            v = values.get(name)
            if v is None:
                return False
            for op, ref in cond.items():
                ref = float(ref)
                if op == "lt" and not v < ref:
                    return False
                if op == "le" and not v <= ref:
                    return False
                if op == "gt" and not v > ref:
                    return False
                if op == "ge" and not v >= ref:
                    return False
                if op == "eq" and not v == ref:
                    return False
                if op == "ne" and not v != ref:
                    return False
        return True


@dataclass(frozen=True)
class Classification:
    """Come la legge attribuisce la classe di attenzione e la cadenza ispettiva.

    E' la parte che distingue una norma moderna: la classe non e' una soglia sola sulla
    condizione, ma una funzione di piu' grandezze osservabili (stato, esposizione di
    rete, natura dell'ostacolo scavalcato, anzianita' dell'informazione), come nella
    classe di attenzione multi-rischio delle Linee Guida italiane e nel metodo a base di
    rischio del 23 CFR 650.311.
    """
    rules: List[ClassificationRule] = field(default_factory=list)
    intervals: Dict[str, int] = field(default_factory=dict)   # classe -> trimestri
    default_cda: str = "Bassa"

    def __post_init__(self):
        if self.default_cda not in CDA_CLASSES:
            raise ValueError(f"default_cda '{self.default_cda}' non valida")
        for k in self.intervals:
            if k not in CDA_CLASSES:
                raise ValueError(f"Classe '{k}' non valida negli intervalli")

    def classify(self, values: Dict[str, float]) -> str:
        for rule in self.rules:
            if rule.matches(values):
                return rule.cda
        return self.default_cda

    def interval(self, cda: str, fallback: int) -> int:
        return int(self.intervals.get(cda, fallback))


@dataclass(frozen=True)
class SanctionStep:
    """Gradino della scala sanzionatoria: dopo `after_quarters` di inadempienza, `action`."""
    after_quarters: int
    action: str                              # azione operativa del catalogo (es. limitazione_carico)


@dataclass(frozen=True)
class Enforcement:
    mode: str = "none"
    sanction_ladder: List[SanctionStep] = field(default_factory=list)
    fine_per_quarter_m2: float = 0.0         # multa per trimestre di inadempienza (EUR/m2)
    lift_on_compliance: bool = True          # la misura decade quando l'obbligo e' assolto
    # La chiusura al traffico non e' una sanzione per inadempienza: nelle norme vere
    # discende dalla capacita' portante (23 CFR 650.313(m) chiude sotto le 3 tonnellate;
    # la scala italiana lega le classi alle tonnellate ammesse). Nei dati del censimento
    # federale i ponti chiusi non superano il 2% in nessuno stato e in nessun anno, e
    # sono quasi tutti nelle valutazioni da 0 a 2. Con questa soglia il gradino di
    # chiusura si applica solo se la condizione OSSERVATA e' sotto il valore indicato;
    # altrimenti la scala si ferma al gradino precedente. None = comportamento storico.
    closure_condition_threshold: Optional[float] = None

    def __post_init__(self):
        if self.mode not in ENFORCEMENT_MODES:
            raise ValueError(f"enforcement.mode '{self.mode}' non valido. Validi: {ENFORCEMENT_MODES}")
        if self.mode in ("sanction", "both") and not self.sanction_ladder and self.fine_per_quarter_m2 <= 0:
            raise ValueError("enforcement 'sanction'/'both' richiede sanction_ladder o fine_per_quarter_m2 > 0")


@dataclass(frozen=True)
class RegulatoryRegime:
    id: str
    display_name: str = ""
    source: str = ""
    hypothetical: bool = False
    target_tier: Optional[str] = None                 # nome di un CodeFramework registrato
    obligations: List[Obligation] = field(default_factory=list)
    enforcement: Enforcement = field(default_factory=Enforcement)
    census_required: bool = False
    earmarked_fraction: float = 0.0                   # quota budget riservata ad azioni 'analysis'
    param_overrides: Dict[str, Any] = field(default_factory=dict)
    # Classi di attenzione fissate dalla legge: soglie di condizione osservata e
    # intervalli di ispezione (trimestri) per classe. Chiavi di CdaConfig, es.
    # {"bassa_due_quarters": 8, "media_threshold": 0.6}. Vuoto = CdA dello scenario.
    cda_config: Dict[str, Any] = field(default_factory=dict)
    # Classificazione dichiarativa: la legge definisce essa stessa come si attribuisce
    # la classe di attenzione, a partire da piu' grandezze osservabili, e con quale
    # cadenza si ispeziona ciascuna classe. Se presente prevale su cda_config.
    classification: Optional[Classification] = None
    notes: str = ""

    @property
    def is_null(self) -> bool:
        return (not self.obligations and self.enforcement.mode == "none" and not self.target_tier
                and not self.census_required and self.earmarked_fraction <= 0 and not self.param_overrides
                and not self.cda_config and self.classification is None)

    def obligation_mode(self, ob: Obligation) -> str:
        """Modalita' di applicazione di un obbligo: la sua, se dichiarata, altrimenti quella della legge."""
        return self.enforcement.mode if getattr(ob, "enforcement", None) is None else ob.enforcement

    def obligation_forced(self, ob: Obligation) -> bool:
        return self.obligation_mode(ob) in ("forced", "both")

    def obligation_sanctioned(self, ob: Obligation) -> bool:
        return self.obligation_mode(ob) in ("sanction", "both")

    def _modes(self) -> List[str]:
        return [self.enforcement.mode] + [o.enforcement for o in self.obligations if o.enforcement is not None]

    @property
    def enforces(self) -> bool:
        return any(m != "none" for m in self._modes())

    @property
    def forces(self) -> bool:
        return any(m in ("forced", "both") for m in self._modes())

    @property
    def sanctions(self) -> bool:
        return any(m in ("sanction", "both") for m in self._modes())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "display_name": self.display_name, "source": self.source,
            "hypothetical": self.hypothetical, "target_tier": self.target_tier,
            "obligations": [{k: v for k, v in o.__dict__.items() if not (k == "enforcement" and v is None)}
                            for o in self.obligations],
            "enforcement": {
                "mode": self.enforcement.mode,
                "sanction_ladder": [s.__dict__ for s in self.enforcement.sanction_ladder],
                "fine_per_quarter_m2": self.enforcement.fine_per_quarter_m2,
                "lift_on_compliance": self.enforcement.lift_on_compliance,
            },
            "information": {"census_required": self.census_required},
            "resources": {"earmarked_fraction": self.earmarked_fraction},
            "param_overrides": dict(self.param_overrides),
            "cda_config": dict(self.cda_config),
            "classification": (None if self.classification is None else {
                "rules": [{"when": r.when, "cda": r.cda} for r in self.classification.rules],
                "intervals": dict(self.classification.intervals),
                "default_cda": self.classification.default_cda}),
            "notes": self.notes,
        }


def _classification_from_dict(data: Optional[Dict[str, Any]]) -> Optional[Classification]:
    if not data:
        return None
    rules = [ClassificationRule(when=dict(r["when"]), cda=r["cda"]) for r in (data.get("rules") or [])]
    if not rules:
        raise ValueError("classification senza regole: serve almeno una riga")
    return Classification(rules=rules,
                          intervals={k: int(v) for k, v in (data.get("intervals") or {}).items()},
                          default_cda=data.get("default_cda", "Bassa"))


def regime_from_dict(data: Dict[str, Any]) -> RegulatoryRegime:
    """Costruisce un regime da un dict (tipicamente caricato da YAML/JSON)."""
    if not isinstance(data, dict) or not data.get("id"):
        raise ValueError("Un regime richiede almeno il campo 'id'.")
    obligations = [
        Obligation(trigger=o["trigger"], action=o["action"], threshold=o.get("threshold"),
                   grace_quarters=int(o.get("grace_quarters", 0)), charged_to=o.get("charged_to", "owner"),
                   discharge=o.get("discharge", "state"),
                   recurrence_quarters=int(o.get("recurrence_quarters", 0)),
                   enforcement=o.get("enforcement"))
        for o in (data.get("obligations") or [])
    ]
    enf_raw = data.get("enforcement") or {}
    ladder = [SanctionStep(after_quarters=int(s["after_quarters"]), action=s["action"])
              for s in (enf_raw.get("sanction_ladder") or [])]
    ladder = sorted(ladder, key=lambda s: s.after_quarters)
    enforcement = Enforcement(
        mode=enf_raw.get("mode", "none"),
        sanction_ladder=ladder,
        closure_condition_threshold=(None if enf_raw.get("closure_condition_threshold") is None
                                     else float(enf_raw["closure_condition_threshold"])),
        fine_per_quarter_m2=float(enf_raw.get("fine_per_quarter_m2", 0.0) or 0.0),
        lift_on_compliance=bool(enf_raw.get("lift_on_compliance", True)),
    )
    info = data.get("information") or {}
    res = data.get("resources") or {}
    return RegulatoryRegime(
        id=str(data["id"]),
        display_name=data.get("display_name", data["id"]),
        source=data.get("source", ""),
        hypothetical=bool(data.get("hypothetical", False)),
        target_tier=data.get("target_tier"),
        obligations=obligations,
        enforcement=enforcement,
        census_required=bool(info.get("census_required", False)),
        earmarked_fraction=float(res.get("earmarked_fraction", 0.0) or 0.0),
        param_overrides=dict(data.get("param_overrides") or {}),
        cda_config=dict(data.get("cda_config") or {}),
        classification=_classification_from_dict(data.get("classification")),
        notes=data.get("notes", ""),
    )


NULL_REGIME = RegulatoryRegime(id="none", display_name="Nessun regime (comportamento storico)",
                               source="—")
