# agents.py
import json
import os
import random
import copy
from typing import Dict, Any, Tuple, List, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import openai
except ImportError:
    openai = None
import pandas as pd
import numpy as np

from infralab.settings.config import NumpyJSONEncoder
from infralab.core.environment import BridgeEnv, Component, _calculate_cost
from infralab.persistence import load_player_action


# ---------------------------------------------------------------------------
# Obiettivo dei pianificatori (planner_objective): helper comuni a GA, memetico, MCTS
# ---------------------------------------------------------------------------

def _planner_objective(agent: 'AgentInterface', env: BridgeEnv) -> Tuple[str, float]:
    """(obiettivo, lambda): override dell'agente se dato, altrimenti SimParameters."""
    objective = getattr(agent, "objective", None) or getattr(env.params, "planner_objective", "asset_value")
    weight = getattr(agent, "safety_weight", None)
    if weight is None:
        weight = getattr(env.params, "planner_safety_weight", 0.0)
    if objective not in ("asset_value", "risk_adjusted"):
        raise ValueError(f"planner_objective non valido: {objective!r}")
    return objective, float(weight or 0.0)


def planner_safety_deficit(sim_env: BridgeEnv, agent_name: str) -> float:
    """Somma sui ponti dell'agente di max(0, beta_target_annuo - beta_annuo), con le stesse
    grandezze del margine esportato (final_min_beta_margin): beta annuo posterior se c'e',
    target da reliability_details["target_beta_annual"]. Senza clamp."""
    total = 0.0
    for b in sim_env.bridges.values():
        if b.manager != agent_name:
            continue
        beta = (b.posterior_annual_reliability_index
                if b.posterior_annual_reliability_index is not None else b.annual_reliability_index)
        target = (b.reliability_details or {}).get("target_beta_annual")
        if beta is None or target is None:
            continue
        total += max(0.0, float(target) - float(beta))
    return total


def planner_step_cost(sim_env: BridgeEnv, agent_name: str, action_costs: Dict[str, Any], safety_weight: float) -> float:
    """Costo di un trimestre simulato per l'obiettivo "risk_adjusted": spesa del gestore
    (costi restituiti da step, come spent_quarter) + perdita attesa da collasso del
    trimestre (last_expected_loss, come metric_risk_cost_quarter) + lambda x deficit di
    sicurezza sullo stato dopo il passo."""
    spent = float(sum(c.get("cost", 0.0) for c in (action_costs or {}).get(agent_name, []) or []))
    loss = (getattr(sim_env, "last_expected_loss", {}) or {}).get(agent_name) \
        or sim_env.expected_quarterly_failure_loss(agent_name)
    cost = spent + float(loss["total"])
    if safety_weight:
        cost += safety_weight * planner_safety_deficit(sim_env, agent_name)
    return cost


def planner_terminal_cost(sim_env: BridgeEnv, agent_name: str, safety_weight: float) -> float:
    """Valore terminale dell'obiettivo "risk_adjusted" (planner_terminal_quarters = N > 0):
    N trimestri di perdita attesa e di lambda x deficit di sicurezza calcolati sullo stato
    a fine rollout, tenuto fermo. Senza di esso un rollout corto vede il costo di un
    intervento ma non il rischio che evita dopo l'orizzonte, e premia il non fare nulla.
    Stato fermo = niente degrado ulteriore: e' una stima per difetto del rischio futuro."""
    n = int(getattr(sim_env.params, "planner_terminal_quarters", 0) or 0)
    if n <= 0:
        return 0.0
    loss = sim_env.expected_quarterly_failure_loss(agent_name)
    per_q = float(loss["total"])
    if safety_weight:
        per_q += safety_weight * planner_safety_deficit(sim_env, agent_name)
    return n * per_q


class AgentInterface:
    def __init__(self, name: str):
        self.name = name

    def decide(self, env: 'BridgeEnv', **kwargs) -> Tuple[Dict[str, Any], Any]:
        """
        Metodo di decisione base per un agente.
        **kwargs può contenere argomenti opzionali come 'session_id' o 'history_dfs'.
        """
        raise NotImplementedError


class HumanAgent(AgentInterface):
    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        session_id = kwargs.get("session_id")
        if not session_id:
            return {}, "Errore: ID di sessione mancante per l'agente umano."

        plan = load_player_action(session_id, self.name)
        return (plan, "Umano") if plan else ({}, "In attesa...")


class HeuristicAgent(AgentInterface):
    """Gestore a regole: ispeziona quando dovuto, prova di carico sotto target, ripara il peggio.

    `follow_guidelines` (default True) governa la regola "ispeziona alla scadenza".
    Storicamente era legata a `params.enforce_guidelines`, che pero' controlla anche il
    throttle sui ricavi del motore: le due semantiche sono ora separate, cosi' l'agente
    resta lo stesso quando il regime normativo cambia (config: heuristic_params).
    """

    def __init__(self, name: str, follow_guidelines: bool = True):
        super().__init__(name)
        self.follow_guidelines = bool(follow_guidelines)

    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = env.get_agent_observations(self.name)
        plan = {"maintenance": {}}
        budget = obs['game_rules']['my_budget']
        my_bridges = [b for b in obs['my_bridges'] if b.get('status') == 'Operativo']
        current_quarter = obs['game_rules']['quarter']
        available_budget = budget
        actioned_bridges = set()

        maint_options = obs['maintenance_options'].copy()

        if self.follow_guidelines:
            due_bridges = sorted([b for b in my_bridges if b['next_inspection_quarter'] <= current_quarter],
                                 key=lambda b: b.get('min_observed_condition') or 1.0)
            for bridge in due_bridges:
                if bridge.get('last_action_name') == 'L1_inspect' and bridge.get(
                        'last_action_quarter') == current_quarter - 1:
                    continue

                action = 'L1_inspect'
                cost = _calculate_cost(maint_options.get(action, {}), bridge['area'])
                if available_budget >= cost:
                    plan["maintenance"][str(bridge['id'])] = action
                    available_budget -= cost
                    actioned_bridges.add(bridge['id'])

        remaining_bridges = [b for b in my_bridges if b['id'] not in actioned_bridges]
        proof_load_action = 'proof_load_test'
        proof_load_budget = None
        if proof_load_action in maint_options:
            def _needs_proof_load(bridge_data):
                beta_ref = bridge_data.get('posterior_reliability_index', bridge_data.get('reliability_index', 99))
                beta_ann = bridge_data.get('posterior_annual_reliability_index', bridge_data.get('annual_reliability_index', 99))
                rel_details = bridge_data.get('posterior_reliability_details') or bridge_data.get('reliability_details') or {}
                target_beta_ann = rel_details.get('target_beta_annual')
                life_cycle = rel_details.get('life_cycle_profile') or {}
                min_future_beta = life_cycle.get('min_remaining_structural_annual_beta')
                below_reference = beta_ref < env.params.reliability_target_beta
                below_annual = target_beta_ann is not None and beta_ann < target_beta_ann
                below_future = target_beta_ann is not None and min_future_beta is not None and min_future_beta < target_beta_ann
                return below_reference or below_annual or below_future

            candidate_bridges = sorted(
                [b for b in remaining_bridges if _needs_proof_load(b)],
                key=lambda b: (
                    b.get('posterior_reliability_index', b.get('reliability_index', 99)),
                    -b.get('risk_priority_index', 0.0),
                )
            )
            if candidate_bridges:
                candidate = candidate_bridges[0]
                proof_load_budget = _calculate_cost(maint_options[proof_load_action], candidate['area'])
                if (
                    available_budget >= proof_load_budget
                    and candidate.get('min_observed_condition') is not None
                    and candidate.get('min_observed_condition') > 0.45
                    and candidate.get('last_action_name') != proof_load_action
                ):
                    plan["maintenance"][str(candidate['id'])] = proof_load_action
                    available_budget -= proof_load_budget
                    actioned_bridges.add(candidate['id'])

        remaining_bridges = [b for b in my_bridges if b['id'] not in actioned_bridges]
        if remaining_bridges:
            all_components = []
            for b in remaining_bridges:
                for c_name, c in b['components'].items():
                    if c['observed_condition_min'] is not None:
                        all_components.append((b['id'], c_name, c['observed_condition_min'], b['area']))

            if all_components:
                worst_comp_info = min(all_components, key=lambda x: x[2], default=None)
                if worst_comp_info and worst_comp_info[2] < 0.8:
                    bridge_id, comp_name, _, bridge_area = worst_comp_info
                    bridge_data = next((b for b in remaining_bridges if b['id'] == bridge_id), None)
                    if bridge_data:
                        weighted_candidates = [
                            item for item in all_components
                            if next((b for b in remaining_bridges if b['id'] == item[0]), {}).get('risk_priority_index', 0.0) >= bridge_data.get('risk_priority_index', 0.0) * 0.85
                        ]
                        if weighted_candidates:
                            worst_comp_info = min(
                                weighted_candidates,
                                key=lambda x: x[2] / max(
                                    0.25,
                                    next((b for b in remaining_bridges if b['id'] == x[0]), {}).get('network_importance_factor', 1.0),
                                ),
                            )
                            bridge_id, comp_name, _, bridge_area = worst_comp_info
                            bridge_data = next((b for b in remaining_bridges if b['id'] == bridge_id), None)

                    possible_actions = []
                    for action_name, op in maint_options.items():
                        if bridge_data and action_name != 'L1_inspect' and action_name == bridge_data.get(
                                'last_action_name') and bridge_data.get('last_action_quarter') == current_quarter - 1:
                            continue
                        if bridge_data and action_name == 'monitoring' and bridge_data.get('monitoring_ends_quarter',
                                                                                           0) > current_quarter:
                            continue

                        gains = op.get('gains', {})
                        if op.get('type') == 'intervention' and comp_name in gains:
                            cost = _calculate_cost(op, bridge_area)
                            if available_budget >= cost:
                                roi = gains.get(comp_name, 0) / cost if cost > 0 else 0
                                possible_actions.append((roi, action_name))

                    if possible_actions:
                        _, best_action_name = max(possible_actions)
                        if str(bridge_id) not in plan['maintenance']:
                            plan["maintenance"][str(bridge_id)] = best_action_name

        return plan, "Euristico: Ispezioni e/o intervento a ROI max."


class RollingHorizonGAAgent(AgentInterface):
    def __init__(self, name: str, horizon: int = 4, pop_size: int = 50, generations: int = 20, tournament_size: int = 3,
                 mutation_rate: float = 0.1, objective: Optional[str] = None, safety_weight: Optional[float] = None):
        super().__init__(name)
        self.horizon = horizon
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        # None = da SimParameters (planner_objective, planner_safety_weight)
        self.objective = objective
        self.safety_weight = safety_weight

    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = env.get_agent_observations(self.name)
        population = [self._random_chromosome(obs, env) for _ in range(self.pop_size)]

        for _ in range(self.generations):
            fitness = [self._evaluate(ch, env) for ch in population]
            new_population = []
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1, obs, env))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(c2, obs, env))
            population = new_population

        final_fitness = [self._evaluate(ch, env) for ch in population]
        best_chromosome = population[final_fitness.index(max(final_fitness))]

        first_plan = best_chromosome[0]
        return {"maintenance": first_plan}, f"GA (dati osservati): 1 di {self.horizon} step"

    def _random_chromosome(self, obs: Dict, env: BridgeEnv):
        return [self._random_plan(obs, env) for _ in range(self.horizon)]

    def _random_plan(self, obs: Dict, env: BridgeEnv) -> Dict[str, str]:
        plan = {}
        budget = obs['game_rules']['my_budget']
        bridges = [b for b in obs['my_bridges'] if b.get('status') == 'Operativo']
        options = obs['maintenance_options']
        random.shuffle(bridges)
        current_quarter = obs['game_rules']['quarter']

        for b in bridges:
            if random.random() < 0.5:
                possible_actions = list(options.keys())
                random.shuffle(possible_actions)
                for act in possible_actions:
                    if act == 'monitoring' and b.get('monitoring_ends_quarter', 0) > current_quarter: continue
                    if act != 'L1_inspect' and act == b.get('last_action_name') and b.get(
                        'last_action_quarter') == current_quarter - 1: continue

                    op = options[act]
                    cost = _calculate_cost(op, b.get('area', 1))
                    if cost <= budget:
                        plan[str(b['id'])] = act
                        budget -= cost
                        break
        return plan

    def _evaluate(self, chromosome: list, initial_env: BridgeEnv) -> float:
        sim_env = initial_env.planning_clone()
        objective, safety_weight = _planner_objective(self, initial_env)

        # Copia "belief" (planner_information="belief"): lo stato e' gia' quello del
        # gestore (o il vero per l'oracolo). La sovrascrittura storica qui sotto resta
        # solo per planner_information="truth".
        real_bridges_obs = [] if getattr(sim_env, "planner_view", None) \
            else initial_env.get_agent_observations(self.name)['my_bridges']

        for bridge_id, sim_bridge in sim_env.bridges.items():
            if sim_bridge.manager != self.name: continue

            real_bridge_data = next((b for b in real_bridges_obs if b['id'] == bridge_id), None)
            if not real_bridge_data: continue

            for comp_name, sim_comp in sim_bridge.components.items():
                real_comp_data = real_bridge_data['components'][comp_name]
                obs_min = real_comp_data.get('observed_condition_min')
                obs_max = real_comp_data.get('observed_condition_max')

                start_condition = 0
                if obs_min is not None and obs_max is not None:
                    start_condition = (obs_min + obs_max) / 2.0
                else:
                    # RNG derivato dal random globale (seedato per run): senza, il
                    # componente campionava parametri da entropia di sistema e la
                    # valutazione del GA non era riproducibile.
                    temp_comp = Component(name=comp_name,
                                          eta_iniziale=real_comp_data['effective_age_years'],
                                          vita_utile=real_comp_data['vita_utile_anni'],
                                          exponent=initial_env.params.deterioration_exponent,
                                          rng=np.random.default_rng(random.getrandbits(32)))
                    start_condition = temp_comp.current_condition

                sim_comp.current_condition = start_condition
                sim_comp.apply_maintenance(0, initial_env.params.deterioration_exponent)

        rollout_cost = 0.0
        for plan in chromosome:
            full_plan = {self.name: {"maintenance": plan}}
            action_costs, _ = sim_env.step(full_plan)
            if objective == "risk_adjusted":
                rollout_cost += planner_step_cost(sim_env, self.name, action_costs, safety_weight)

        if objective == "risk_adjusted":
            fitness = -(rollout_cost + planner_terminal_cost(sim_env, self.name, safety_weight))
        else:
            my_bridges = [b for b in sim_env.bridges.values() if b.manager == self.name]
            final_asset_value = sum(b.current_asset_value for b in my_bridges)
            final_budget = sim_env.budgets.get(self.name, 0)
            fitness = final_budget + final_asset_value

        if self.name not in sim_env.active_agents:
            fitness = -1e12

        return fitness

    def _tournament_select(self, population: list, fitness: list) -> list:
        contenders_indices = random.sample(range(len(population)), self.tournament_size)
        best_index = max(contenders_indices, key=lambda i: fitness[i])
        return population[best_index]

    def _crossover(self, p1: list, p2: list) -> tuple:
        if len(p1) < 2 or len(p2) < 2: return p1, p2
        point = random.randint(1, self.horizon - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def _mutate(self, chromosome: list, obs: Dict, env: BridgeEnv) -> list:
        if random.random() < self.mutation_rate:
            quarter_to_mutate = random.randrange(self.horizon)
            chromosome[quarter_to_mutate] = self._random_plan(obs, env)
        return chromosome


class MemeticAgent(AgentInterface):
    def __init__(self, name: str, horizon: int = 4, pop_size: int = 40, generations: int = 15,
                 tournament_size: int = 3, mutation_rate: float = 0.1,
                 local_search_steps: int = 5, objective: Optional[str] = None,
                 safety_weight: Optional[float] = None):
        super().__init__(name)
        self.horizon = horizon
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.local_search_steps = local_search_steps
        # None = da SimParameters (planner_objective, planner_safety_weight)
        self.objective = objective
        self.safety_weight = safety_weight

    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = env.get_agent_observations(self.name)
        action_sets = self._build_action_sets(obs, env)
        population = self._initialize_population(obs, env, action_sets)

        for _ in range(self.generations):
            fitness = [self._evaluate(ch, env) for ch in population]
            new_population = []
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitness)
                p2 = self._tournament_select(population, fitness)
                child, _ = self._crossover(p1, p2)
                child = self._mutate(child, obs, env, action_sets)

                child = self._local_search(child, env, action_sets)

                new_population.append(child)

            population = new_population

        final_fitness = [self._evaluate(ch, env) for ch in population]
        best_chromosome = population[final_fitness.index(max(final_fitness))]

        first_plan = best_chromosome[0]
        return {"maintenance": first_plan}, f"Algoritmo Memetico guidato: 1 di {self.horizon} step"

    def _initialize_population(self, obs: Dict, env: BridgeEnv, action_sets: Dict[str, List[Optional[str]]]) -> List[list]:
        population: List[list] = []
        empty = [dict() for _ in range(self.horizon)]
        population.append(empty)

        heuristic_plan, _ = HeuristicAgent(self.name).decide(env)
        heuristic_chromosome = [copy.deepcopy(heuristic_plan.get("maintenance", {}))] + [dict() for _ in range(max(0, self.horizon - 1))]
        population.append(heuristic_chromosome)

        while len(population) < self.pop_size:
            population.append(self._random_chromosome(obs, env, action_sets))
        return population

    def _build_action_sets(self, obs: Dict, env: BridgeEnv) -> Dict[str, List[Optional[str]]]:
        action_sets: Dict[str, List[Optional[str]]] = {}
        current_quarter = obs["game_rules"]["quarter"]
        for bridge in [b for b in obs["my_bridges"] if b.get("status") == "Operativo"]:
            bridge_id = str(bridge["id"])
            feasible = bridge.get("action_feasibility", {})
            candidates: List[tuple[float, str]] = []
            rel_details = bridge.get("posterior_reliability_details") or bridge.get("reliability_details") or {}
            life_cycle = rel_details.get("life_cycle_profile") or {}
            target_ann = rel_details.get("target_beta_annual") or env.params.reliability_target_beta
            beta_ref = bridge.get("posterior_reliability_index", bridge.get("reliability_index", 99.0))
            beta_ann = bridge.get("posterior_annual_reliability_index", bridge.get("annual_reliability_index", 99.0))
            min_future_beta = life_cycle.get("min_remaining_structural_annual_beta", beta_ann)
            min_obs_cond = bridge.get("min_observed_condition")
            backlog = int(bridge.get("next_inspection_quarter", 9999) <= current_quarter)
            knowledge_gap = float(min_obs_cond is None)
            risk_priority = float(bridge.get("risk_priority_index", 0.0))

            if feasible.get("L1_inspect", {}).get("feasible") and (knowledge_gap > 0 or backlog):
                candidates.append((8.0 + 4.0 * knowledge_gap + 2.0 * backlog, "L1_inspect"))
            if feasible.get("L4_verify", {}).get("feasible") and (
                knowledge_gap > 0 or beta_ref < env.params.reliability_target_beta or min_future_beta < target_ann
            ):
                candidates.append((10.0 + 6.0 * knowledge_gap + max(0.0, target_ann - min_future_beta), "L4_verify"))
            if feasible.get("proof_load_test", {}).get("feasible") and min_obs_cond is not None and 0.45 <= min_obs_cond <= 0.90:
                if beta_ref < env.params.reliability_target_beta or min_future_beta < target_ann:
                    candidates.append((14.0 + max(0.0, target_ann - min_future_beta) + 8.0 * risk_priority, "proof_load_test"))
            if feasible.get("monitoring", {}).get("feasible") and knowledge_gap == 0 and min_future_beta < target_ann:
                candidates.append((7.0 + 0.5 * max(0.0, target_ann - min_future_beta), "monitoring"))

            for action_name, op in obs["maintenance_options"].items():
                gains = op.get("gains", {})
                if not feasible.get(action_name, {}).get("feasible"):
                    continue
                if op.get("type") != "intervention":
                    continue
                gain_total = sum(float(v) for v in gains.values())
                if gain_total <= 0.0:
                    continue
                bridge_area = bridge.get("area", 1.0)
                cost = _calculate_cost(op, bridge_area)
                if cost <= 0:
                    continue
                condition_pressure = max(0.0, 0.88 - (min_obs_cond if min_obs_cond is not None else bridge.get("min_condition", 0.5)))
                reliability_pressure = max(0.0, target_ann - min_future_beta)
                score = (gain_total * (1.0 + 4.0 * condition_pressure + 2.5 * reliability_pressure + 4.0 * risk_priority)) / cost
                candidates.append((score, action_name))

            ranked_actions = [name for _, name in sorted(candidates, key=lambda item: item[0], reverse=True)]
            deduped_actions: List[Optional[str]] = [None]
            for action_name in ranked_actions:
                if action_name not in deduped_actions:
                    deduped_actions.append(action_name)
                if len(deduped_actions) >= 5:
                    break
            action_sets[bridge_id] = deduped_actions
        return action_sets

    def _local_search(self, chromosome: list, env: BridgeEnv, action_sets: Dict[str, List[Optional[str]]]) -> list:
        current_fitness = self._evaluate(chromosome, env)
        bridge_ids = list(action_sets.keys())
        if not bridge_ids:
            return chromosome

        for _ in range(self.local_search_steps):
            neighbor = copy.deepcopy(chromosome)
            quarter_to_tweak = 0 if self.horizon == 1 else random.choice([0, random.randrange(self.horizon)])
            plan_to_tweak = neighbor[quarter_to_tweak]
            bridge_id_str = random.choice(bridge_ids)
            possible_actions = action_sets.get(bridge_id_str, [None])
            new_action = random.choice(possible_actions)

            if new_action is None:
                plan_to_tweak.pop(bridge_id_str, None)
            else:
                plan_to_tweak[bridge_id_str] = new_action

            neighbor_fitness = self._evaluate(neighbor, env)
            if neighbor_fitness > current_fitness:
                chromosome = neighbor
                current_fitness = neighbor_fitness

        return chromosome

    def _random_chromosome(self, obs: Dict, env: BridgeEnv, action_sets: Dict[str, List[Optional[str]]]):
        chromosome = [self._random_plan(obs, env, action_sets)]
        for _ in range(max(0, self.horizon - 1)):
            chromosome.append({})
        return chromosome

    def _random_plan(self, obs: Dict, env: BridgeEnv, action_sets: Dict[str, List[Optional[str]]]) -> Dict[str, str]:
        plan = {}
        budget = obs['game_rules']['my_budget']
        bridges = [b for b in obs['my_bridges'] if b.get('status') == 'Operativo']
        priority = sorted(
            bridges,
            key=lambda b: (
                (b.get('posterior_reliability_details') or b.get('reliability_details') or {}).get('life_cycle_profile', {}).get('min_remaining_structural_annual_beta', 99.0),
                -b.get('risk_priority_index', 0.0),
                b.get('min_observed_condition') if b.get('min_observed_condition') is not None else 0.5,
            )
        )

        for b in priority:
            bridge_id = str(b['id'])
            if random.random() < 0.35:
                continue
            for act in action_sets.get(bridge_id, [None]):
                if act is None:
                    continue
                op = obs['maintenance_options'].get(act, {})
                cost = _calculate_cost(op, b.get('area', 1))
                if cost <= budget:
                    plan[bridge_id] = act
                    budget -= cost
                    break
        return plan

    def _evaluate(self, chromosome: list, initial_env: BridgeEnv) -> float:
        sim_env = initial_env.planning_clone()
        objective, safety_weight = _planner_objective(self, initial_env)
        if objective == "risk_adjusted":
            # Obiettivo allineato al benchmark: tutto il cromosoma (orizzonte), come il GA.
            rollout_cost = 0.0
            for plan in (chromosome or [{}]):
                action_costs, _ = sim_env.step({self.name: {"maintenance": plan}})
                rollout_cost += planner_step_cost(sim_env, self.name, action_costs, safety_weight)
            if self.name not in sim_env.active_agents:
                return -1e12
            return -(rollout_cost + planner_terminal_cost(sim_env, self.name, safety_weight))
        first_plan = chromosome[0] if chromosome else {}
        sim_env.step({self.name: {"maintenance": first_plan}})

        my_bridges = [b for b in sim_env.bridges.values() if b.manager == self.name]
        if not my_bridges or self.name not in sim_env.active_agents:
            return -1e12

        final_asset_value = sum(b.current_asset_value for b in my_bridges)
        final_budget = sim_env.budgets.get(self.name, 0.0)
        net_worth = final_budget + final_asset_value
        total_adt = sum(b.adt for b in my_bridges)
        health_index = sum(b.min_condition * b.adt for b in my_bridges) / max(total_adt, 1)

        current_annual_betas = []
        remaining_annual_betas = []
        remaining_annual_pfs = []
        target_annual_betas = []
        proof_load_count = 0
        l4_count = 0
        for bridge in my_bridges:
            details = bridge.posterior_reliability_details or bridge.reliability_details or {}
            life_cycle = details.get("life_cycle_profile") or {}
            current_annual_betas.append(bridge.posterior_annual_reliability_index if bridge.posterior_annual_reliability_index is not None else bridge.annual_reliability_index or -7.0)
            remaining_annual_betas.append(life_cycle.get("min_remaining_structural_annual_beta", -7.0))
            remaining_annual_pfs.append(life_cycle.get("max_remaining_structural_annual_pf", 1.0))
            target_annual_betas.append(details.get("target_beta_annual", initial_env.params.reliability_target_beta))
            if bridge.last_proof_load_report and bridge.last_proof_load_report.get("quarter") == sim_env.quarter - 1:
                proof_load_count += 1
            if bridge.last_action_name == "L4_verify" and bridge.last_action_quarter == sim_env.quarter - 1:
                l4_count += 1

        min_current_annual_beta = min(current_annual_betas) if current_annual_betas else -7.0
        min_remaining_annual_beta = min(remaining_annual_betas) if remaining_annual_betas else -7.0
        max_remaining_annual_pf = max(remaining_annual_pfs) if remaining_annual_pfs else 1.0
        target_annual_beta = max(target_annual_betas) if target_annual_betas else initial_env.params.reliability_target_beta

        annual_shortfall = max(0.0, target_annual_beta - min_remaining_annual_beta)
        current_shortfall = max(0.0, target_annual_beta - min_current_annual_beta)

        score = 0.15 * net_worth
        score += 4.5e7 * health_index
        score += 5.5e7 * min_remaining_annual_beta
        score += 2.0e7 * min_current_annual_beta
        score -= 1.2e8 * max_remaining_annual_pf
        score -= 8.0e7 * annual_shortfall
        score -= 4.0e7 * current_shortfall
        score += 3.0e6 * proof_load_count
        score -= 1.0e6 * l4_count
        return score

    def _tournament_select(self, population: list, fitness: list) -> list:
        contenders_indices = random.sample(range(len(population)), self.tournament_size)
        best_index = max(contenders_indices, key=lambda i: fitness[i])
        return population[best_index]

    def _crossover(self, p1: list, p2: list) -> tuple:
        if len(p1) < 2 or len(p2) < 2: return p1, p2
        point = random.randint(1, self.horizon - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def _mutate(self, chromosome: list, obs: Dict, env: BridgeEnv, action_sets: Dict[str, List[Optional[str]]]) -> list:
        if random.random() < self.mutation_rate:
            quarter_to_mutate = 0 if self.horizon == 1 else random.choice([0, random.randrange(self.horizon)])
            chromosome[quarter_to_mutate] = self._random_plan(obs, env, action_sets)
        return chromosome


class LLMAgent(AgentInterface):
    """Agente LLM multi-provider.

    Provider supportati:
    - "OpenAI":      API OpenAI (chiave da OPENAI_API_KEY)
    - "Google":      API Gemini (chiave da GOOGLE_API_KEY)
    - "HuggingFace": Hugging Face Hub (token da HF_TOKEN, modello via InferenceClient)
    - "Locale":      qualunque server OpenAI-compatibile (Ollama, LM Studio, vLLM…)
                     via INFRALAB_LLM_BASE_URL (default http://localhost:11434/v1)
                     e INFRALAB_LLM_MODEL (default llama3.1)
    Il modello e' sovrascrivibile per-agente (parametro `model`) o via env
    OPENAI_LLM_MODEL / GOOGLE_LLM_MODEL / HF_LLM_MODEL.
    """

    def __init__(self, name: str, profile: str, provider: str, model: str | None = None, hf_token: str | None = None):
        super().__init__(name)
        self.profile, self.provider = profile, provider
        self.openai_model = model or os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
        self.google_model = model or os.environ.get("GOOGLE_LLM_MODEL", "gemini-2.0-flash")
        self.hf_model = model or os.environ.get("HF_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.local_model = model or os.environ.get("INFRALAB_LLM_MODEL", "llama3.1")
        self.local_base_url = os.environ.get("INFRALAB_LLM_BASE_URL", "http://localhost:11434/v1")

    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        history_dfs = kwargs.get("history_dfs", {})
        obs = env.get_agent_observations(self.name)
        if self.provider == "Google":
            return self._decide_with_gemini(obs, history_dfs)
        elif self.provider == "OpenAI":
            return self._decide_with_openai(obs, history_dfs)
        elif self.provider == "HuggingFace":
            return self._decide_with_huggingface(obs, history_dfs)
        elif self.provider == "Locale":
            return self._decide_with_local(obs, history_dfs)
        return {}, ("", f"Provider LLM non supportato: {self.provider}")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Estrae il primo blocco JSON dal testo (con o senza fence markdown)."""
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```json")[-1] if "```json" in cleaned else cleaned.split("```")[1]
            cleaned = cleaned.replace("```", "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)

    def _profile_instruction(self) -> str:
        instructions = {
            "Molto Prudente": "Privilegia sicurezza strutturale e affidabilità futura. Evita politiche aggressive sul budget. Anticipa ispezioni e monitoraggio.",
            "Prudente": "Mantieni alta affidabilità e riduci rischio di failure. Accetta spesa preventiva se giustificata.",
            "Bilanciato": "Bilancia affidabilità, budget e valore finale del portafoglio senza estremizzare il comportamento.",
            "Dinamico": "Accetta rischio moderato per proteggere il valore complessivo. Usa interventi selettivi e rialloca budget dove il beneficio marginale è maggiore.",
            "Aggressivo": "Massimizza efficienza economica di medio termine. Tolleranza al rischio più alta, ma senza ignorare failure imminente.",
        }
        return instructions.get(self.profile, instructions["Bilanciato"])

    def _history_memory(self, history_df_kpi, history_df_budget) -> str:
        memory_lines = []
        if history_df_kpi is not None and not history_df_kpi.empty:
            agent_kpis = history_df_kpi[history_df_kpi['agent'] == self.name].tail(4)
            for _, row in agent_kpis.iterrows():
                memory_lines.append(
                    f"T{int(row['quarter'])}: HI={row.get('health_index', 0):.3f}, "
                    f"backlog={row.get('backlog_count', 0)}, "
                    f"beta_net={row.get('network_reliability_index', 0):.3f}, "
                    f"pf_net={row.get('network_failure_probability', 0):.2e}"
                )
        if history_df_budget is not None and not history_df_budget.empty:
            agent_budget = history_df_budget[history_df_budget['agent'] == self.name].tail(2)
            for _, row in agent_budget.iterrows():
                memory_lines.append(
                    f"T{int(row['quarter'])}: budget={row.get('budget', 0):,.0f}, revenue={row.get('revenue', 0):,.0f}, costs={row.get('costs', 0):,.0f}"
                )
        return "\n".join(memory_lines[-6:]) if memory_lines else "Nessuna memoria storica disponibile."

    def _get_prompt(self, obs, history_df_kpi, history_df_budget):
        simplified_bridges = []
        for b in obs['my_bridges']:
            s_b = {k: v for k, v in b.items() if
                   k not in ['components', 'future_curve_range', 'last_action_name', 'last_action_quarter']}
            s_b['components_observed_range'] = {
                c_name: [round(c['observed_condition_min'], 3), round(c['observed_condition_max'], 3)]
                if c['observed_condition_min'] is not None else "Sconosciuta"
                for c_name, c in b['components'].items()
            }
            s_b['available_actions'] = {
                action_name: item for action_name, item in b.get('action_feasibility', {}).items() if item.get('feasible')
            }
            s_b['blocked_actions'] = {
                action_name: item.get('reason') for action_name, item in b.get('action_feasibility', {}).items() if not item.get('feasible')
            }
            simplified_bridges.append(s_b)

        history_summary = ""
        if history_df_kpi is not None and not history_df_kpi.empty:
            agent_kpis = history_df_kpi[history_df_kpi['agent'] == self.name].tail(4)
            if not agent_kpis.empty:
                kpi_start = agent_kpis.iloc[0]
                kpi_end = agent_kpis.iloc[-1]
                health_change = kpi_end['health_index'] - kpi_start['health_index']
                budget_end = obs['game_rules']['my_budget']
                history_summary = f"Riepilogo ultimi trimestri: Variazione salute: {health_change:+.2%}, Budget attuale: {budget_end:,.0f}€."
        memory_summary = self._history_memory(history_df_kpi, history_df_budget)

        params = obs['game_settings']
        guidelines_text = "Le linee guida sono ATTIVE." if params.get(
            'enforce_guidelines') else "Le linee guida sono DISATTIVATE."

        return f"""Sei un Bridge Manager con profilo '{self.profile}'.
OBIETTIVO: Massimizzare il 'Patrimonio Netto Complessivo' (Budget + Valore Asset) a fine partita.
STILE DECISIONALE: {self._profile_instruction()}
{history_summary}
MEMORIA OPERATIVA:
{memory_summary}
CONTESTO:
- Regole Finanziarie: Budget non speso cresce del {params.get('unallocated_budget_return_annual', 0):.2%} all'anno.
- Regole di Ispezione: {guidelines_text}
- Degrado: Accelerato da traffico (ADT) e ambiente.
- Osservazioni: Le ispezioni danno una forchetta di affidabilità [min, max].
- Affidabilità: considera prioritari ponti con beta basso, Pf alto o backlog ispettivo.
- Layer di gestione: il piano viene validato prima dell'esecuzione. Evita azioni bloccate o non finanziabili.
- Vincoli operativi: {json.dumps(obs.get('management_constraints', {}), indent=2, cls=NumpyJSONEncoder)}
- Explainability: la spiegazione deve motivare trade-off tra rischio, costo e orizzonte.

STATO ATTUALE (Trimestre {obs['game_rules']['quarter']}):
- Budget: {obs['game_rules']['my_budget']:,.2f}€
- Ponti: {json.dumps(simplified_bridges, indent=2, cls=NumpyJSONEncoder)}
- Azioni Disponibili: {json.dumps(obs['maintenance_options'], indent=2, cls=NumpyJSONEncoder)}
- Diagnostica ultimo validatore: {json.dumps(obs.get('last_plan_diagnostics', []), indent=2, cls=NumpyJSONEncoder)}

COMPITO: Analizza lo stato e decidi un piano di manutenzione. Formula la decisione in formato JSON.

FORMATO RISPOSTA (SOLO JSON):
```json
{{
  "explanation": "La mia logica...",
  "plan": {{
    "maintenance": {{
      "ID_PONTE_1": "azione_scelta_1"
    }}
  }}
}}
```"""

    def _decide_with_openai(self, obs: Dict, history_dfs: Dict):
        prompt = self._get_prompt(obs, history_dfs.get('kpi'), history_dfs.get('budget'))
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(model=self.openai_model,
                                                      messages=[{"role": "system",
                                                                 "content": "Rispondi solo in formato JSON, seguendo lo schema richiesto."},
                                                                {"role": "user", "content": prompt}], temperature=0.5,
                                                      response_format={"type": "json_object"})
            data = json.loads(response.choices[0].message.content)
            return data.get("plan", {}), (prompt, (data, response.choices[0].message.content))
        except Exception as e:
            return {}, ("", f"Errore API OpenAI: {e}")

    def _decide_with_gemini(self, obs: Dict, history_dfs: Dict):
        prompt = self._get_prompt(obs, history_dfs.get('kpi'), history_dfs.get('budget'))
        try:
            model = genai.GenerativeModel(self.google_model)
            response = model.generate_content(prompt)
            data = self._extract_json(response.text)
            return data.get("plan", {}), (prompt, (data, response.text))
        except Exception as e:
            return {}, (
            "", f"Errore API Gemini: {e}\nTesto ricevuto: {response.text if 'response' in locals() else 'N/A'}")

    def _decide_with_huggingface(self, obs: Dict, history_dfs: Dict):
        """Invocazione serverless/endpoint di Hugging Face via InferenceClient."""
        prompt = self._get_prompt(obs, history_dfs.get('kpi'), history_dfs.get('budget'))
        try:
            from huggingface_hub import InferenceClient
            token = self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            client = InferenceClient(token=token)
            response = client.chat.completions.create(
                model=self.hf_model,
                messages=[
                    {"role": "system", "content": "Rispondi esclusivamente con un oggetto JSON valido, seguendo esattamente lo schema indicato, senza testo né markdown extra."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content or ""
            data = self._extract_json(raw)
            return data.get("plan", {}), (prompt, (data, raw))
        except Exception as e:
            return {}, ("", f"Errore API HuggingFace ({self.hf_model}): {e}")

    def _decide_with_local(self, obs: Dict, history_dfs: Dict):
        """Server OpenAI-compatibile locale (Ollama, LM Studio, vLLM, llama.cpp…)."""
        prompt = self._get_prompt(obs, history_dfs.get('kpi'), history_dfs.get('budget'))
        try:
            client = openai.OpenAI(
                base_url=self.local_base_url,
                api_key=os.environ.get("INFRALAB_LLM_API_KEY", "not-needed"),
            )
            # Niente response_format: molti server locali non lo supportano.
            response = client.chat.completions.create(
                model=self.local_model,
                messages=[
                    {"role": "system", "content": "Rispondi solo in formato JSON, seguendo lo schema richiesto."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            text = response.choices[0].message.content or ""
            data = self._extract_json(text)
            return data.get("plan", {}), (prompt, (data, text))
        except Exception as e:
            return {}, ("", f"Errore LLM locale ({self.local_base_url}, {self.local_model}): {e}")


# ---------------------------------------------------------------------------
# MCTSAgent — Monte Carlo Tree Search per pianificazione a orizzonte
# ---------------------------------------------------------------------------

class MCTSAgent(AgentInterface):
    """
    Monte Carlo Tree Search per la pianificazione della manutenzione.
    Genera N piani candidati e li valuta con rollout stocastici nel simulatore,
    selezionando il piano con il valore atteso massimo.
    """

    def __init__(
        self,
        name: str,
        n_rollouts: int = 50,
        rollout_depth: int = 4,
        n_candidates: int = 12,
        objective: Optional[str] = None,
        safety_weight: Optional[float] = None,
    ):
        super().__init__(name)
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.n_candidates = n_candidates
        # None = da SimParameters (planner_objective, planner_safety_weight)
        self.objective = objective
        self.safety_weight = safety_weight

    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = env.get_agent_observations(self.name)
        candidates = self._generate_candidates(obs, env)

        if not candidates:
            return {"maintenance": {}}, "MCTS: nessun piano candidato disponibile"

        rollouts_each = max(1, self.n_rollouts // len(candidates))
        best_plan, best_value = candidates[0], -float("inf")

        for plan in candidates:
            values = [self._rollout(plan, env) for _ in range(rollouts_each)]
            mean_val = float(np.mean(values))
            if mean_val > best_value:
                best_value = mean_val
                best_plan = plan

        return (
            {"maintenance": best_plan},
            f"MCTS ({self.n_rollouts} rollout, depth={self.rollout_depth}, valore={best_value:,.0f})",
        )

    def _generate_candidates(self, obs: Dict, env: BridgeEnv) -> List[Dict]:
        candidates = [{}]  # piano vuoto (do-nothing) sempre incluso
        for _ in range(self.n_candidates - 1):
            candidates.append(self._random_plan(obs, env))
        return candidates

    def _random_plan(self, obs: Dict, env: BridgeEnv) -> Dict[str, str]:
        plan: Dict[str, str] = {}
        budget = obs["game_rules"]["my_budget"]
        bridges = [b for b in obs["my_bridges"] if b.get("status") == "Operativo"]
        options = obs["maintenance_options"]
        current_q = obs["game_rules"]["quarter"]
        random.shuffle(bridges)

        for b in bridges:
            if random.random() < 0.55:
                acts = list(options.keys())
                random.shuffle(acts)
                for act in acts:
                    if act == "monitoring" and b.get("monitoring_ends_quarter", 0) > current_q:
                        continue
                    if (act != "L1_inspect"
                            and act == b.get("last_action_name")
                            and b.get("last_action_quarter") == current_q - 1):
                        continue
                    cost = _calculate_cost(options[act], b.get("area", 1))
                    if cost <= budget:
                        plan[str(b["id"])] = act
                        budget -= cost
                        break
        return plan

    def _rollout(self, initial_plan: Dict, env: BridgeEnv) -> float:
        # planner_information="belief": ogni rollout parte da una nuova estrazione
        # dalla credenza del gestore (si integra sull'incertezza dello stato).
        sim_env = env.planning_clone()
        objective, safety_weight = _planner_objective(self, env)
        rollout_cost = 0.0
        action_costs, _ = sim_env.step({self.name: {"maintenance": initial_plan}})
        if objective == "risk_adjusted":
            rollout_cost += planner_step_cost(sim_env, self.name, action_costs, safety_weight)

        for _ in range(self.rollout_depth - 1):
            obs = sim_env.get_agent_observations(self.name)
            rp = self._random_plan(obs, sim_env)
            action_costs, _ = sim_env.step({self.name: {"maintenance": rp}})
            if objective == "risk_adjusted":
                rollout_cost += planner_step_cost(sim_env, self.name, action_costs, safety_weight)

        if self.name not in sim_env.active_agents:
            return -1e12

        if objective == "risk_adjusted":
            return -(rollout_cost + planner_terminal_cost(sim_env, self.name, safety_weight))
        my_bridges = [b for b in sim_env.bridges.values() if b.manager == self.name]
        return sum(b.current_asset_value for b in my_bridges) + sim_env.budgets.get(self.name, 0.0)


# ---------------------------------------------------------------------------
# RLAgent — Q-Learning con approssimazione lineare della funzione valore
# ---------------------------------------------------------------------------

class RLAgent(AgentInterface):
    """
    Agente di Reinforcement Learning basato su Q-Learning online
    con approssimazione lineare della funzione Q.

    Impara durante la simulazione (online) aggiornando i pesi a ogni turno
    tramite l'errore TD (Temporal Difference).
    """

    FEATURE_DIM = 14  # budget + quarter + n_bridges + backlog + 10 condizioni

    def __init__(
        self,
        name: str,
        learning_rate: float = 0.04,
        discount: float = 0.95,
        epsilon_start: float = 0.30,
        epsilon_decay: float = 0.993,
        epsilon_min: float = 0.05,
    ):
        super().__init__(name)
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Pesi Q: {action_key: np.array(FEATURE_DIM)}
        self.Q_weights: Dict[str, np.ndarray] = {}

        self._last_features: Optional[np.ndarray] = None
        self._last_action_key: Optional[str] = None
        self._last_budget: float = 0.0

    # --- Feature engineering ---

    def _featurize(self, obs: Dict) -> np.ndarray:
        budget_norm = min(obs["game_rules"]["my_budget"] / 2_000_000.0, 4.0)
        quarter_norm = obs["game_rules"]["quarter"] / 200.0
        bridges = sorted(obs["my_bridges"], key=lambda b: b["id"])
        n_bridges_norm = len(bridges) / 20.0
        backlog = sum(
            1 for b in bridges
            if b.get("next_inspection_quarter", 9999) < obs["game_rules"]["quarter"]
        ) / max(len(bridges), 1)

        conditions = []
        for b in bridges[:10]:
            mc = b.get("min_observed_condition")
            conditions.append(float(mc) if mc is not None else 0.5)
        while len(conditions) < 10:
            conditions.append(0.0)

        return np.array(
            [budget_norm, quarter_norm, n_bridges_norm, backlog] + conditions,
            dtype=np.float32,
        )

    def _action_key(self, plan: Dict) -> str:
        return json.dumps(sorted(plan.items()), sort_keys=True)

    def _get_q(self, features: np.ndarray, action_key: str) -> float:
        if action_key not in self.Q_weights:
            self.Q_weights[action_key] = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        return float(np.dot(self.Q_weights[action_key], features[: self.FEATURE_DIM]))

    def _update_q(self, features: np.ndarray, action_key: str, td_error: float) -> None:
        if action_key not in self.Q_weights:
            self.Q_weights[action_key] = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        self.Q_weights[action_key] += self.lr * td_error * features[: self.FEATURE_DIM]

    # --- Azione ---

    def _build_candidates(self, obs: Dict) -> List[Dict]:
        candidates: List[Dict] = [{}]  # do-nothing
        budget = obs["game_rules"]["my_budget"]
        current_q = obs["game_rules"]["quarter"]
        bridges = [b for b in obs["my_bridges"] if b.get("status") == "Operativo"]

        for b in bridges:
            for act_name, op in obs["maintenance_options"].items():
                if act_name == "monitoring" and b.get("monitoring_ends_quarter", 0) > current_q:
                    continue
                cost = _calculate_cost(op, b.get("area", 1))
                if cost <= budget:
                    candidates.append({str(b["id"]): act_name})
        return candidates

    def decide(self, env: BridgeEnv, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = env.get_agent_observations(self.name)
        features = self._featurize(obs)
        candidates = self._build_candidates(obs)

        # --- Aggiornamento TD dal passo precedente ---
        if self._last_features is not None and self._last_action_key is not None:
            reward = (obs["game_rules"]["my_budget"] - self._last_budget) / 100_000.0
            next_q_vals = [self._get_q(features, self._action_key(p)) for p in candidates]
            max_next_q = max(next_q_vals) if next_q_vals else 0.0
            td_error = reward + self.gamma * max_next_q - self._get_q(self._last_features, self._last_action_key)
            self._update_q(self._last_features, self._last_action_key, td_error)

        # --- Selezione epsilon-greedy ---
        if random.random() < self.epsilon:
            chosen = random.choice(candidates)
        else:
            q_vals = [self._get_q(features, self._action_key(p)) for p in candidates]
            chosen = candidates[int(np.argmax(q_vals))]

        # Salva stato per il prossimo aggiornamento
        self._last_features = features.copy()
        self._last_action_key = self._action_key(chosen)
        self._last_budget = obs["game_rules"]["my_budget"]
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return (
            {"maintenance": chosen},
            f"RL Q-Learning (ε={self.epsilon:.3f}, politiche note: {len(self.Q_weights)})",
        )


# ---------------------------------------------------------------------------
# Factory aggiornata
# ---------------------------------------------------------------------------

def agent_factory(name: str, config: Dict) -> AgentInterface:
    agent_type = config.get("type")
    if agent_type == "Umano": return HumanAgent(name)
    if agent_type == "Euristico": return HeuristicAgent(name, **config.get("heuristic_params", {}))
    if agent_type == "Basic":
        from infralab.agents.basic import BasicAgent
        return BasicAgent(name, policy=config.get("policy", "do_nothing"), params=config.get("params"))
    if agent_type == "LLM":
        return LLMAgent(
            name,
            config.get("profile", "Bilanciato"),
            config.get("provider", "Google"),
            model=config.get("model"),
            hf_token=config.get("hf_token"),
        )
    if agent_type == "LLM Agentico":
        from infralab.agents.agentic import AgenticLLMAgent
        return AgenticLLMAgent(
            name,
            config.get("profile", "Bilanciato"),
            config.get("provider", "Locale"),
            model=config.get("model"),
            max_rounds=int(config.get("max_rounds", 3)),
            hf_token=config.get("hf_token"),
        )
    if agent_type == "GA Orizzonte Variabile":
        return RollingHorizonGAAgent(name, **config.get("ga_params", {}))
    if agent_type == "Algoritmo Memetico":
        return MemeticAgent(name, **config.get("ga_params", {}))
    if agent_type == "MCTS":
        return MCTSAgent(name, **config.get("mcts_params", {}))
    if agent_type == "RL Q-Learning":
        return RLAgent(name, **config.get("rl_params", {}))
    if agent_type == "Custom":
        from infralab.agents.scripted import ScriptedPluginAgent
        return ScriptedPluginAgent(name, plugin=config.get("plugin"), code=config.get("code"))
    raise ValueError(f"Tipo agente non riconosciuto: {agent_type}")
