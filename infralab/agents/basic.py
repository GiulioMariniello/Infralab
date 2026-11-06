"""Strategie di gestione elementari (policy di riferimento interpretabili).

Ogni strategia e' una regola di poche righe sull'osservazione standard del
motore (`get_agent_observations`). Servono come riferimento leggibile per gli
esperimenti "stessa strategia, legge diversa": la risposta di una regola a un
cambio di regime e' riconducibile alla clausola che cambia.

Config: {"type": "Basic", "policy": <nome>, "params": {...}}
Policy disponibili: vedi BASIC_POLICIES.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from infralab.agents.legacy import AgentInterface, _calculate_cost

INTERVENTION_TYPES = ("intervention", "preventive")


def _observed(b: Dict[str, Any]) -> Optional[float]:
    return b.get("min_observed_condition")


def _feasible(b: Dict[str, Any], action: str) -> bool:
    return bool(b.get("action_feasibility", {}).get(action, {}).get("feasible"))


def _feasible_cost(b: Dict[str, Any], action: str) -> float:
    return float(b.get("action_feasibility", {}).get(action, {}).get("cost", 0.0))


def _best_intervention(b: Dict[str, Any], maint_options: Dict[str, Any], budget: float,
                       prefer_strong: bool = True) -> Optional[str]:
    """Intervento fattibile con il maggior guadagno totale (o il piu' economico se prefer_strong=False)."""
    cands: List[Tuple[float, float, str]] = []
    for name, op in maint_options.items():
        if op.get("type") not in INTERVENTION_TYPES or not _feasible(b, name):
            continue
        cost = _feasible_cost(b, name)
        if cost > budget:
            continue
        gain = float(sum((op.get("gains") or {}).values())) if op.get("type") == "intervention" else 0.05
        cands.append((gain, -cost, name))
    if not cands:
        return None
    cands.sort(reverse=prefer_strong)
    return cands[0][2]


def _cheapest_intervention(b: Dict[str, Any], maint_options: Dict[str, Any], budget: float) -> Optional[str]:
    """Intervento strutturale fattibile piu' economico (non manutenzione preventiva)."""
    cands = [(_feasible_cost(b, name), name) for name, op in maint_options.items()
             if op.get("type") == "intervention" and _feasible(b, name) and _feasible_cost(b, name) <= budget]
    return min(cands)[1] if cands else None


def _inspection_action(b: Dict[str, Any], level: int = 1) -> Optional[str]:
    for name in (f"L{level}_inspect", "L1_inspect", "L2_inspect"):
        if _feasible(b, name):
            return name
    return None


class BasicAgent(AgentInterface):
    """Agente a regola fissa. `policy` sceglie la regola, `params` i suoi parametri."""

    def __init__(self, name: str, policy: str = "do_nothing", params: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        if policy not in BASIC_POLICIES:
            raise ValueError(f"Policy '{policy}' sconosciuta. Disponibili: {sorted(BASIC_POLICIES)}")
        self.policy = policy
        self.p: Dict[str, Any] = {**BASIC_POLICIES[policy]["defaults"], **(params or {})}
        self._state: Dict[int, Any] = {}

    def decide(self, env, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = env.get_agent_observations(self.name)
        budget = float(obs["game_rules"]["my_budget"])
        quarter = int(obs["game_rules"]["quarter"])
        bridges = [b for b in obs["my_bridges"] if b.get("status") == "Operativo"]
        maint = obs["maintenance_options"]
        plan: Dict[str, str] = {}
        fn = getattr(self, f"_policy_{self.policy}")
        fn(bridges, maint, budget, quarter, plan)
        # Variante "auto-ispezionante": la regola resta la stessa, ma il gestore si
        # procura da solo l'informazione (ispezione L1 se l'ultima e' piu' vecchia di N
        # trimestri) sugli asset su cui la regola non ha gia' deciso un'azione.
        n_si = int(self.p.get("self_inspect_interval_quarters", 0) or 0)
        if n_si > 0:
            remaining = budget - sum(_feasible_cost(b, a) for b in bridges for k, a in plan.items() if str(b["id"]) == k)
            for b in bridges:
                key = str(b["id"])
                if key in plan:
                    continue
                if quarter - int(b.get("last_inspection_quarter", 0) or 0) >= n_si:
                    act = _inspection_action(b, 1)
                    if act and _feasible_cost(b, act) <= remaining:
                        plan[key] = act
                        remaining -= _feasible_cost(b, act)
        return {"maintenance": plan}, ("basic", {"policy": self.policy, "actions": len(plan)})

    # ---- regole ---------------------------------------------------------
    def _policy_do_nothing(self, bridges, maint, budget, quarter, plan):
        return

    def _policy_danger_only(self, bridges, maint, budget, quarter, plan):
        """Interviene solo quando la condizione osservata (o il beta) segnala pericolo."""
        th = float(self.p["condition_threshold"])
        beta_th = self.p.get("beta_threshold")
        for b in sorted(bridges, key=lambda x: (_observed(x) if _observed(x) is not None else 1.0)):
            c = _observed(b)
            beta = b.get("posterior_reliability_index", b.get("reliability_index"))
            danger = (c is not None and c < th) or (beta_th is not None and beta is not None and beta < float(beta_th))
            if not danger:
                continue
            act = _best_intervention(b, maint, budget)
            if act:
                plan[str(b["id"])] = act
                budget -= _feasible_cost(b, act)

    def _policy_fixed_interval(self, bridges, maint, budget, quarter, plan):
        """Ispeziona tutti ogni N trimestri; ripara chi e' sotto soglia osservata."""
        n = int(self.p["interval_quarters"])
        th = float(self.p["repair_threshold"])
        for b in bridges:
            key = str(b["id"])
            c = _observed(b)
            if c is not None and c < th:
                act = _best_intervention(b, maint, budget)
                if act:
                    plan[key] = act
                    budget -= _feasible_cost(b, act)
                    continue
            if quarter % n == 0:
                act = _inspection_action(b, 1)
                if act and _feasible_cost(b, act) <= budget:
                    plan[key] = act
                    budget -= _feasible_cost(b, act)

    def _policy_worst_first(self, bridges, maint, budget, quarter, plan):
        """Ripara dal peggiore, entro un tetto di spesa per trimestre (frazione del budget)."""
        cap = float(self.p["spend_cap_fraction"]) * max(0.0, budget)
        spent = 0.0
        for b in sorted(bridges, key=lambda x: (_observed(x) if _observed(x) is not None else 1.0)):
            c = _observed(b)
            if c is None or c >= float(self.p["repair_threshold"]):
                continue
            act = _best_intervention(b, maint, cap - spent)
            if act:
                plan[str(b["id"])] = act
                spent += _feasible_cost(b, act)

    def _policy_condition_hysteresis(self, bridges, maint, budget, quarter, plan):
        """Entra in 'intervento' sotto la soglia bassa, esce sopra quella alta (niente oscillazioni)."""
        low, high = float(self.p["low"]), float(self.p["high"])
        for b in bridges:
            bid = int(b["id"])
            c = _observed(b)
            active = self._state.get(bid, False)
            if c is not None:
                if not active and c < low:
                    active = True
                elif active and c > high:
                    active = False
            self._state[bid] = active
            if active:
                act = _best_intervention(b, maint, budget) or (
                    "manutenzione_ordinaria" if _feasible(b, "manutenzione_ordinaria") else None)
                if act:
                    plan[str(b["id"])] = act
                    budget -= _feasible_cost(b, act)

    def _policy_risk_based(self, bridges, maint, budget, quarter, plan):
        """A rischio: Pf annuale (stimata, posterior se c'e') sopra t1 ispeziona (L2), sopra t2 interviene.

        Soglie in scala di probabilita' annua (default 1e-4 ~ beta 3.7, 1e-3 ~ beta 3.1).
        """
        t1, t2 = float(self.p["inspect_threshold"]), float(self.p["repair_threshold"])

        def _pf(x):
            v = x.get("posterior_annual_failure_probability")
            if v is None:
                v = x.get("annual_failure_probability")
            return float(v or 0.0)

        for b in sorted(bridges, key=lambda x: -_pf(x)):
            r = _pf(b)
            key = str(b["id"])
            if r > t2:
                act = _best_intervention(b, maint, budget)
                if act:
                    plan[key] = act
                    budget -= _feasible_cost(b, act)
                    continue
            if r > t1:
                act = _inspection_action(b, 2)
                if act and _feasible_cost(b, act) <= budget:
                    plan[key] = act
                    budget -= _feasible_cost(b, act)

    def _policy_compliance_minimal(self, bridges, maint, budget, quarter, plan):
        """Fa esattamente cio' che la legge impone, e nient'altro."""
        for b in bridges:
            key = str(b["id"])
            act = b.get("regime_mandatory_action")
            if act is None:
                # Senza regime: segue i soli segnali Linee Guida dell'osservazione.
                if b.get("overdue_L4_assessment"):
                    act = "L4_verify"
                elif b.get("inspection_overdue") or b.get("not_censused"):
                    act = "L0_census" if b.get("not_censused") else "L1_inspect"
            if act and _feasible(b, act) and _feasible_cost(b, act) <= budget:
                plan[key] = act
                budget -= _feasible_cost(b, act)

    def _policy_compliance_gaming(self, bridges, maint, budget, quarter, plan):
        """Fa cio' che la legge impone e, in piu', tiene la condizione OSSERVATA appena
        sopra la soglia che farebbe scattare l'obbligo costoso, spendendo ogni volta
        l'intervento strutturale piu' economico disponibile.

        E' l'archetipo della conformita' creativa descritto nella letteratura sulla
        regolazione (McBarnet e Whelan 1991; Edelman 1992): non l'inerzia, ma il
        rispetto della lettera della regola al minimo costo, che e' cosa diversa dal
        rispetto del suo scopo. Parametri: `legal_threshold` (la soglia di condizione
        che fa scattare l'obbligo nella legge in vigore) e `margin` (di quanto sopra
        la soglia il gestore si tiene).
        """
        th, margin = float(self.p["legal_threshold"]), float(self.p["margin"])
        for b in bridges:
            key = str(b["id"])
            act = b.get("regime_mandatory_action")
            if act is None:
                if b.get("overdue_L4_assessment"):
                    act = "L4_verify"
                elif b.get("inspection_overdue") or b.get("not_censused"):
                    act = "L0_census" if b.get("not_censused") else "L1_inspect"
            if act is None:
                seen = _observed(b)
                if seen is not None and th <= float(seen) <= th + margin:
                    act = _cheapest_intervention(b, maint, budget)
            if act and _feasible(b, act) and _feasible_cost(b, act) <= budget:
                plan[key] = act
                budget -= _feasible_cost(b, act)

    def _policy_target_keeper(self, bridges, maint, budget, quarter, plan):
        """Legge il target della norma in vigore: interviene quando il beta annuo STIMATO
        scende sotto il target annuo (+ margine), ispeziona quando manca informazione."""
        margin = float(self.p["margin"])
        for b in bridges:
            key = str(b["id"])
            det = b.get("posterior_reliability_details") or b.get("reliability_details") or {}
            target = det.get("target_beta_annual")
            beta = b.get("posterior_annual_reliability_index")
            if beta is None:
                beta = b.get("annual_reliability_index")
            if target is None or beta is None:
                continue
            if float(beta) < float(target) + margin:
                act = _best_intervention(b, maint, budget)
                if act is None and _observed(b) is None:
                    act = _inspection_action(b, 1)
                if act and _feasible_cost(b, act) <= budget:
                    plan[key] = act
                    budget -= _feasible_cost(b, act)

    def _policy_inspect_and_repair(self, bridges, maint, budget, quarter, plan):
        """Regola 'tecnico prudente': ispeziona quando dovuto, ripara sotto soglia, altrimenti manutenzione ordinaria periodica."""
        th = float(self.p["repair_threshold"])
        every = int(self.p["preventive_every_quarters"])
        for b in sorted(bridges, key=lambda x: (_observed(x) if _observed(x) is not None else 1.0)):
            key = str(b["id"])
            c = _observed(b)
            if b.get("inspection_overdue") or quarter >= int(b.get("next_inspection_quarter", 10**9)):
                act = _inspection_action(b, 1)
            elif c is not None and c < th:
                act = _best_intervention(b, maint, budget)
            elif every > 0 and quarter % every == 0 and _feasible(b, "manutenzione_ordinaria"):
                act = "manutenzione_ordinaria"
            else:
                act = None
            if act and _feasible(b, act) and _feasible_cost(b, act) <= budget:
                plan[key] = act
                budget -= _feasible_cost(b, act)

    def _policy_typical_owner(self, bridges, maint, budget, quarter, plan):
        """Gestore tipico: ogni anno interviene su un ponte con una probabilita' che dipende
        solo dal voto osservato (0-9, voto = 9 x centro dell'intervallo osservato del
        componente peggiore). Regola veloce, senza simulazioni, pensata per riprodurre
        la frequenza di interventi di un inventario reale: la tabella di default e' la
        probabilita' annua che il voto peggiore salga di almeno un punto nell'inventario
        americano dei ponti (PA, TX, OH, IA, 2015-2022).

        Parametri: `annual_rate_by_rating` (voto -> probabilita' annua), `rate_scale`
        (moltiplicatore), `major_share` (quota di interventi maggiori: quello col maggior
        guadagno totale; gli altri sostituiscono il solo componente peggiore), `seed`.
        L'estrazione e' a chiave (seme, ponte, trimestre): ripetibile e indipendente da
        cio' che la regola ha deciso prima. La policy non conosce il seme della partita:
        chi lancia piu' repliche passi un `seed` diverso per replica.
        """
        rates = {int(k): float(v) for k, v in (self.p["annual_rate_by_rating"] or {}).items()}
        scale, major, seed = float(self.p["rate_scale"]), float(self.p["major_share"]), int(self.p["seed"])
        for b in bridges:
            centres = {n: 0.5 * (float(c["observed_condition_min"]) + float(c["observed_condition_max"]))
                       for n, c in (b.get("components") or {}).items()
                       if c.get("observed_condition_min") is not None and c.get("observed_condition_max") is not None}
            if not centres:
                continue
            worst = min(centres, key=lambda n: (centres[n], n))
            rating = int(min(9, max(0, round(9.0 * centres[worst]))))
            p_year = min(1.0, max(0.0, scale * rates.get(rating, 0.0)))
            if p_year <= 0.0:
                continue
            u = np.random.default_rng([seed, 7701, int(b["id"]), int(quarter)]).random(2)
            if u[0] >= 1.0 - (1.0 - p_year) ** 0.25:
                continue
            act = None
            if u[1] >= major:
                act = "replace_" + worst
                if not (_feasible(b, act) and _feasible_cost(b, act) <= budget):
                    act = None
            if act is None:
                act = _best_intervention(b, maint, budget)
            if act:
                plan[str(b["id"])] = act
                budget -= _feasible_cost(b, act)


# P(il voto peggiore sale di almeno 1 in un anno), inventario americano dei ponti (NBI),
# PA+TX+OH+IA, coppie di anni consecutivi 2015-2022.
TYPICAL_OWNER_RATES = {0: 0.17, 1: 0.17, 2: 0.17, 3: 0.073, 4: 0.050, 5: 0.031, 6: 0.021, 7: 0.005, 8: 0.002, 9: 0.0}

BASIC_POLICIES: Dict[str, Dict[str, Any]] = {
    "do_nothing": {"label": "Nessuna azione", "defaults": {}},
    "danger_only": {"label": "Agisci solo in caso di pericolo",
                    "defaults": {"condition_threshold": 0.4, "beta_threshold": None}},
    "fixed_interval": {"label": "Ispezione a intervallo fisso + riparazione sotto soglia",
                       "defaults": {"interval_quarters": 4, "repair_threshold": 0.6}},
    "worst_first": {"label": "Ripara il peggio entro un tetto di spesa",
                    "defaults": {"spend_cap_fraction": 0.25, "repair_threshold": 0.7}},
    "condition_hysteresis": {"label": "A condizione con isteresi", "defaults": {"low": 0.5, "high": 0.75}},
    "risk_based": {"label": "A rischio (Pf annua: ispeziona sopra t1, interviene sopra t2)",
                   "defaults": {"inspect_threshold": 1e-4, "repair_threshold": 1e-3}},
    "compliance_minimal": {"label": "Conformita' minima (solo cio' che la legge impone)", "defaults": {}},
    "compliance_gaming": {"label": "Conformita' creativa (resta appena sopra la soglia di legge al minimo costo)",
                          "defaults": {"legal_threshold": 0.50, "margin": 0.06}},
    "target_keeper": {"label": "Tiene il target della norma (legge il tier, agisce sul beta stimato)",
                      "defaults": {"margin": 0.0}},
    "typical_owner": {"label": "Gestore tipico (probabilita' annua di intervento per voto osservato)",
                      "defaults": {"annual_rate_by_rating": TYPICAL_OWNER_RATES, "rate_scale": 1.0,
                                   "major_share": 0.3, "seed": 0}},
    "inspect_and_repair": {"label": "Tecnico prudente (ispeziona quando dovuto, ripara, manutenzione periodica)",
                           "defaults": {"repair_threshold": 0.6, "preventive_every_quarters": 8}},
}
