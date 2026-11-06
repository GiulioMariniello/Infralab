# analysis.py
import io
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

from infralab.settings.config import UNINA_PALETTE
from infralab.core.environment import BridgeEnv

_FALLBACK_COLORS = ["#0077c8", "#f9a825", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def _agent_colors(agent_names: list) -> dict:
    """Restituisce un dizionario agente→colore garantendo nessun valore None."""
    all_colors = UNINA_PALETTE + _FALLBACK_COLORS
    return {name: all_colors[i % len(all_colors)] for i, name in enumerate(agent_names)}


def _plot_lines_by_agent(ax, df_pivoted: pd.DataFrame, palette: dict, marker: str = "o"):
    """Plotta ogni colonna (agente) del DataFrame pivotato come linea separata."""
    for col in df_pivoted.columns:
        color = palette.get(col, _FALLBACK_COLORS[0])
        ax.plot(df_pivoted.index, df_pivoted[col], marker=marker, label=col, color=color, linewidth=1.8)


def create_kpi_charts(df_kpi: pd.DataFrame, agent_names: list) -> tuple:
    """Crea e restituisce le figure per i grafici KPI."""
    if df_kpi.empty:
        return None, None

    palette = _agent_colors(agent_names)

    # Figura 1: Indice di Salute
    fig_health, ax_health = plt.subplots(figsize=(10, 4))
    if "health_index" in df_kpi.columns:
        df_plot_health = df_kpi.pivot_table(index="quarter", columns="agent", values="health_index", aggfunc='mean')
        _plot_lines_by_agent(ax_health, df_plot_health, palette)
    ax_health.set_title("Indice di Salute della Rete")
    ax_health.set_ylabel("Salute (0-1)")
    ax_health.set_xlabel("Trimestre")
    ax_health.legend(title="Agente")

    # Figura 2: Ispezioni Scadute
    fig_backlog, ax_backlog = plt.subplots(figsize=(10, 4))
    if "backlog_count" in df_kpi.columns:
        df_plot_backlog = df_kpi.pivot_table(index="quarter", columns="agent", values="backlog_count", aggfunc='mean')
        _plot_lines_by_agent(ax_backlog, df_plot_backlog, palette)
    ax_backlog.set_title("Numero di Ispezioni Scadute")
    ax_backlog.set_ylabel("N. Ispezioni")
    ax_backlog.set_xlabel("Trimestre")
    ax_backlog.legend(title="Agente")

    return fig_health, fig_backlog


def create_financial_charts(df_budget: pd.DataFrame, df_kpi: pd.DataFrame, agent_names: list) -> tuple:
    """Crea e restituisce le figure per i grafici finanziari."""
    if df_budget.empty:
        return None, None

    palette = _agent_colors(agent_names)

    # Figura 1: Budget Liquido
    fig_budget, ax_budget = plt.subplots(figsize=(10, 4))
    df_plot_budget = df_budget.pivot_table(index="quarter", columns="agent", values="budget", aggfunc='mean')
    if not df_plot_budget.empty:
        _plot_lines_by_agent(ax_budget, df_plot_budget, palette)
        ax_budget.set_title("Andamento Budget Liquido")
        ax_budget.grid(True)
    ax_budget.set_ylabel("Budget (€)")
    ax_budget.set_xlabel("Trimestre")
    ax_budget.legend(title="Agente")

    # Figura 2: Patrimonio Netto
    fig_net_worth, ax_net_worth = plt.subplots(figsize=(10, 4))
    if not df_kpi.empty:
        df_merged = pd.merge(df_budget, df_kpi, on=["quarter", "agent"], how="inner")
        if not df_merged.empty and "total_asset_value" in df_merged.columns:
            df_merged["net_worth"] = df_merged["budget"] + df_merged["total_asset_value"]
            df_plot_net_worth = df_merged.pivot_table(index="quarter", columns="agent", values="net_worth", aggfunc='mean')
            if not df_plot_net_worth.empty:
                _plot_lines_by_agent(ax_net_worth, df_plot_net_worth, palette)
                ax_net_worth.set_title("Andamento Patrimonio Netto Complessivo")
                ax_net_worth.grid(True)
    ax_net_worth.set_ylabel("Patrimonio (€)")
    ax_net_worth.set_xlabel("Trimestre")
    ax_net_worth.legend(title="Agente")

    return fig_budget, fig_net_worth

def create_summary_df(env: BridgeEnv, history_dfs: dict) -> pd.DataFrame:
    """Crea un DataFrame di riepilogo dello stato finale."""
    summary_data = []
    df_kpi = history_dfs.get('kpi')

    for agent_name in env.params.agent_names:
        liquid_budget = env.budgets.get(agent_name, 0)
        my_bridges = [b for b in env.bridges.values() if b.manager == agent_name]
        total_asset_value = sum(b.current_asset_value for b in my_bridges)
        comprehensive_net_worth = liquid_budget + total_asset_value

        final_health_index = 0.0
        if df_kpi is not None and not df_kpi.empty:
            agent_kpi = df_kpi[df_kpi['agent'] == agent_name]
            if not agent_kpi.empty:
                last_kpi_entry = agent_kpi.loc[agent_kpi['quarter'].idxmax()]
                final_health_index = last_kpi_entry['health_index']

        summary_data.append({
            "Agente": agent_name,
            "Stato": "Attivo" if agent_name in env.active_agents else "Fallito",
            "Patrimonio Netto Complessivo": comprehensive_net_worth,
            "Budget Liquido": liquid_budget,
            "Valore Asset Ponti": total_asset_value,
            "Indice di Salute": final_health_index,
            "Beta Affidabilità Min": min(
                [b.posterior_reliability_index if b.posterior_reliability_index is not None else b.reliability_index for b in my_bridges],
                default=0.0,
            ),
            "Beta Annuale Min": min(
                [b.posterior_annual_reliability_index if b.posterior_annual_reliability_index is not None else b.annual_reliability_index for b in my_bridges],
                default=0.0,
            ),
            "Pf Max": max(
                [b.posterior_failure_probability if b.posterior_failure_probability is not None else b.failure_probability for b in my_bridges],
                default=0.0,
            ),
            "Traffic Lambda Max": max(
                [b.traffic_fragility.get("annual_failure_rate", 0.0) for b in my_bridges if b.traffic_fragility],
                default=0.0,
            ),
            "Seismic Pf Max": max(
                [b.seismic_fragility.get("failure_probability", 0.0) for b in my_bridges if b.seismic_fragility],
                default=0.0,
            ),
        })
    return pd.DataFrame(summary_data)


def create_paper_experiment_df(env: BridgeEnv) -> pd.DataFrame:
    rows = []
    for bridge in env.bridges.values():
        traffic = bridge.traffic_fragility or {}
        seismic = bridge.seismic_fragility or {}
        rows.append({
            "Bridge ID": bridge.id,
            "Bridge Name": bridge.name,
            "Manager": bridge.manager,
            "Min Condition": bridge.min_condition,
            "Posterior Beta": bridge.posterior_reliability_index if bridge.posterior_reliability_index is not None else bridge.reliability_index,
            "Posterior Annual Beta": bridge.posterior_annual_reliability_index if bridge.posterior_annual_reliability_index is not None else bridge.annual_reliability_index,
            "Posterior Pf": bridge.posterior_failure_probability if bridge.posterior_failure_probability is not None else bridge.failure_probability,
            "Posterior Annual Pf": bridge.posterior_annual_failure_probability if bridge.posterior_annual_failure_probability is not None else bridge.annual_failure_probability,
            "Min Remaining Annual Beta": ((bridge.posterior_reliability_details or bridge.reliability_details or {}).get("life_cycle_profile") or {}).get("min_remaining_structural_annual_beta"),
            "Year Of Min Remaining Beta": ((bridge.posterior_reliability_details or bridge.reliability_details or {}).get("life_cycle_profile") or {}).get("year_of_min_structural_beta"),
            "Traffic Model": traffic.get("traffic_model"),
            "Traffic Eta": traffic.get("eta"),
            "Traffic Beta": traffic.get("beta"),
            "Traffic Annual Failure Rate": traffic.get("annual_failure_rate"),
            "Seismic Class": seismic.get("bridge_class"),
            "Seismic Limit State": seismic.get("limit_state"),
            "Seismic Pf": seismic.get("failure_probability"),
            "Seismic Annual Exceedance Proxy": seismic.get("annual_exceedance_proxy"),
            "Last Proof Load Passed": bridge.last_proof_load_report.get("passed") if bridge.last_proof_load_report else None,
            "Last Proof Load Alpha": bridge.last_proof_load_report.get("alpha") if bridge.last_proof_load_report else None,
            "Last Management Feedback": bridge.last_management_feedback.get("source") if bridge.last_management_feedback else None,
            "Action History Length": len(bridge.action_history),
            "Network Importance Factor": bridge.network_importance_factor,
            "Network Centrality": bridge.network_centrality,
            "Risk Priority Index": bridge.risk_priority_index,
            "Calibration Profile": env.params.calibration_profile,
            "Portfolio Size": len(env.bridges),
            "Planning Horizon Years": env.params.years,
            "Management Objective Mode": env.params.management_objective_mode,
        })
    return pd.DataFrame(rows)


def create_management_diagnostics_df(history_dfs: dict, agent_names: list) -> pd.DataFrame:
    df_kpi = history_dfs.get('kpi') if history_dfs else None
    df_action = history_dfs.get('action') if history_dfs else None
    df_kpi = df_kpi if isinstance(df_kpi, pd.DataFrame) else pd.DataFrame()
    df_action = df_action if isinstance(df_action, pd.DataFrame) else pd.DataFrame()
    rows = []
    for agent in agent_names:
        kpi_agent = df_kpi[df_kpi['agent'] == agent] if not df_kpi.empty and 'agent' in df_kpi.columns else pd.DataFrame()
        action_agent = df_action[df_action['agent'] == agent] if not df_action.empty and 'agent' in df_action.columns else pd.DataFrame()
        if kpi_agent.empty and action_agent.empty:
            continue
        blocked_actions = int((action_agent['action_status'] == 'blocked').sum()) if 'action_status' in action_agent.columns else 0
        requested_actions = int(action_agent['requested_action'].notna().sum()) if 'requested_action' in action_agent.columns else 0
        dominant_block = None
        if 'action_block_reason' in action_agent.columns:
            reasons = action_agent['action_block_reason'].dropna()
            if not reasons.empty:
                dominant_block = reasons.value_counts().idxmax()
        rows.append({
            "Agente": agent,
            "Inspection Coverage Finale": kpi_agent['inspection_coverage'].iloc[-1] if not kpi_agent.empty and 'inspection_coverage' in kpi_agent.columns else np.nan,
            "Knowledge Gap Max": kpi_agent['knowledge_gap_count'].max() if not kpi_agent.empty and 'knowledge_gap_count' in kpi_agent.columns else np.nan,
            "Backlog Max": kpi_agent['backlog_count'].max() if not kpi_agent.empty and 'backlog_count' in kpi_agent.columns else np.nan,
            "Monitoring Attivi Max": kpi_agent['active_monitoring_count'].max() if not kpi_agent.empty and 'active_monitoring_count' in kpi_agent.columns else np.nan,
            "Blocked Actions": blocked_actions,
            "Blocked Action Rate": (blocked_actions / requested_actions) if requested_actions > 0 else 0.0,
            "Dominant Block Reason": dominant_block,
        })
    return pd.DataFrame(rows)

def create_statistical_analysis(history_dfs: dict, agent_names: list) -> dict:
    """
    Analisi statistica descrittiva per agente.
    Restituisce un dict con DataFrames pronti per la visualizzazione.
    """
    df_kpi = history_dfs.get('kpi', pd.DataFrame())
    df_budget = history_dfs.get('budget', pd.DataFrame())
    df_action = history_dfs.get('action', pd.DataFrame())
    results = {}

    # --- Statistiche Health Index per agente ---
    if not df_kpi.empty and 'health_index' in df_kpi.columns:
        hi_stats = []
        for agent in agent_names:
            ag = df_kpi[df_kpi['agent'] == agent]['health_index'].dropna()
            if ag.empty:
                continue
            hi_stats.append({
                'Agente': agent,
                'Media': ag.mean(),
                'Std': ag.std(),
                'Min': ag.min(),
                'Max': ag.max(),
                'Mediana': ag.median(),
                'CoV (%)': (ag.std() / ag.mean() * 100) if ag.mean() > 0 else 0,
                'Trimestre Peggiore': int(df_kpi[(df_kpi['agent'] == agent)].loc[
                    df_kpi[df_kpi['agent'] == agent]['health_index'].idxmin(), 'quarter'
                ]) if not ag.empty else 0,
            })
        results['health_stats'] = pd.DataFrame(hi_stats)

    # --- Statistiche Budget per agente ---
    if not df_budget.empty and 'budget' in df_budget.columns:
        bud_stats = []
        for agent in agent_names:
            ag = df_budget[df_budget['agent'] == agent]['budget'].dropna()
            if ag.empty:
                continue
            bud_stats.append({
                'Agente': agent,
                'Budget Medio': ag.mean(),
                'Budget Min': ag.min(),
                'Budget Max': ag.max(),
                'Budget Finale': ag.iloc[-1] if len(ag) > 0 else 0,
                'Trimestri Negativi': int((ag < 0).sum()),
            })
        results['budget_stats'] = pd.DataFrame(bud_stats)

    # --- Analisi azioni per agente ---
    if not df_action.empty and 'action' in df_action.columns:
        action_cols = ['agent', 'action'] if 'agent' in df_action.columns else ['action']
        action_counts = df_action.groupby(action_cols).size().reset_index(name='Conteggio')
        results['action_counts'] = action_counts

        # Costo totale per agente
        if 'cost' in df_action.columns and 'agent' in df_action.columns:
            cost_by_agent = df_action.groupby('agent')['cost'].agg(['sum', 'mean', 'count']).reset_index()
            cost_by_agent.columns = ['Agente', 'Costo Totale', 'Costo Medio Azione', 'N. Azioni']
            results['cost_stats'] = cost_by_agent

    # --- Correlazione Health Index vs Budget ---
    if not df_kpi.empty and not df_budget.empty:
        merged = pd.merge(
            df_kpi[['quarter', 'agent', 'health_index']],
            df_budget[['quarter', 'agent', 'budget']],
            on=['quarter', 'agent'], how='inner'
        )
        if not merged.empty:
            corr_data = []
            for agent in agent_names:
                ag = merged[merged['agent'] == agent][['health_index', 'budget']].dropna()
                if len(ag) > 2:
                    corr = ag['health_index'].corr(ag['budget'])
                    corr_data.append({'Agente': agent, 'Correlazione HI-Budget': round(corr, 3)})
            results['correlations'] = pd.DataFrame(corr_data)

    if not df_kpi.empty and 'network_reliability_index' in df_kpi.columns:
        rel_stats = []
        for agent in agent_names:
            ag = df_kpi[df_kpi['agent'] == agent]['network_reliability_index'].dropna()
            if ag.empty:
                continue
            rel_stats.append({
                'Agente': agent,
                'Beta Medio': ag.mean(),
                'Beta Min': ag.min(),
                'Beta Finale': ag.iloc[-1] if len(ag) > 0 else 0,
            })
        results['reliability_stats'] = pd.DataFrame(rel_stats)

    management_diag = create_management_diagnostics_df(history_dfs, agent_names)
    if not management_diag.empty:
        results['management_diagnostics'] = management_diag

    return results


def create_degradation_model_df(vita_utile: int = 100, exponent: float = 3.0) -> pd.DataFrame:
    """
    Genera un DataFrame con le curve di degrado analitiche per diversi modelli.
    Utile per confronto visivo dei modelli di degrado supportati.
    """
    ages = np.linspace(0, vita_utile, 200)
    rows = []
    for age in ages:
        r = min(age / vita_utile, 1.0)
        rows.append({
            'Età (anni)': age,
            'Power Law (attuale)': max(0.0, 1.0 - r ** exponent),
            'Lineare': max(0.0, 1.0 - r),
            'Weibull': max(0.0, math.exp(-(r ** exponent))),
            'Bilineare': max(0.0, 1.0 - (r * 0.4 if r < 0.5 else 0.2 + (r - 0.5) * 1.6)),
            'Esponenziale': max(0.0, math.exp(-exponent * r) / math.exp(0)),
        })
    return pd.DataFrame(rows)


def create_final_report_excel(env: BridgeEnv, history_dfs: dict) -> bytes:
    """Crea un file Excel con il report finale completo."""
    output = io.BytesIO()
    engine = "openpyxl"
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"

    df_action = history_dfs.get('action') if history_dfs else None
    df_budget = history_dfs.get('budget') if history_dfs else None
    df_kpi = history_dfs.get('kpi') if history_dfs else None

    with pd.ExcelWriter(output, engine=engine) as writer:
        pd.DataFrame(asdict(env.params).items(), columns=['Parametro', 'Valore']).to_excel(writer,
                                                                                           sheet_name='Parametri_Simulazione',
                                                                                           index=False)
        (df_action if isinstance(df_action, pd.DataFrame) else pd.DataFrame()).to_excel(writer, sheet_name='Log_Azioni', index=False)
        (df_budget if isinstance(df_budget, pd.DataFrame) else pd.DataFrame()).to_excel(writer, sheet_name='Log_Budget', index=False)
        (df_kpi if isinstance(df_kpi, pd.DataFrame) else pd.DataFrame()).to_excel(writer, sheet_name='Log_KPI', index=False)
        create_paper_experiment_df(env).to_excel(writer, sheet_name='Output_Paper', index=False)
        create_management_diagnostics_df(history_dfs, env.params.agent_names).to_excel(writer, sheet_name='Diagnostica_Gestione', index=False)
    return output.getvalue()
