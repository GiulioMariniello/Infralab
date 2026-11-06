"""Contratto di una normativa (CodeFramework): risolve i target di affidabilità.

Un CodeFramework mappa la classe di conseguenza nel target β (e periodo di
riferimento, β SLS, incremento per rottura fragile). Selezionando la normativa
si può valutare — con un agente neutro o un'ottimizzazione semplice — l'effetto
del codice sulle decisioni, senza toccare il resto del motore.

Il framework 'custom' è un passthrough: non risolve nulla e lascia i target
manuali dei parametri (comportamento storico invariato).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class CodeFramework:
    name: str
    display_name: str
    source: str
    # Target β per classe di conseguenza (None per 'custom').
    beta_reference_by_cc: Optional[Dict[str, float]] = None
    reference_period_years: Optional[int] = None
    sls_beta_reference: Optional[float] = None
    brittle_beta_increment: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def resolve(self, consequence_class: str) -> Dict[str, object]:
        """Override da applicare ai SimParameters per questa normativa (vuoto = custom)."""
        if self.beta_reference_by_cc is None:
            return {}
        cc = (consequence_class or "CC2").upper()
        overrides: Dict[str, object] = {}
        beta = self.beta_reference_by_cc.get(cc)
        if beta is not None:
            overrides["reliability_target_beta"] = float(beta)
        if self.reference_period_years is not None:
            overrides["reliability_reference_period_years"] = int(self.reference_period_years)
        if self.sls_beta_reference is not None:
            overrides["reliability_sls_target_beta"] = float(self.sls_beta_reference)
        if self.brittle_beta_increment is not None:
            overrides["reliability_brittle_failure_beta_increment"] = float(self.brittle_beta_increment)
        # I target diventano quelli della normativa: azzera eventuali override
        # manuali dell'annuale così non mascherano la scelta del codice.
        overrides["reliability_target_beta_annual"] = None
        return overrides
