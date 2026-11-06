import io
import os
import json
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from dataclasses import asdict
import matplotlib.style as mplstyle

import infralab.persistence as persistence
import analysis
from infralab.core.environment import BridgeEnv, _calculate_cost
from infralab.agents.sandbox import SandboxViolation, policy_env, sandbox_enabled, sanitized_observation
from infralab.settings.config import SimParameters, MaintOpsConfig, CdaConfig, DEFAULT_COMPONENTS, AGENT_TYPES, API_PROVIDERS, \
    LLM_PROFILES, UNINA_PALETTE, HF_DEFAULT_MODELS
from infralab.agents.legacy import agent_factory, LLMAgent, HeuristicAgent
from infralab.calibration_profiles import list_calibration_profiles, get_calibration_profile, apply_calibration_profile

mplstyle.use('seaborn-v0_8-whitegrid')


# ---------------------------------------------------------------------------
# Design system InfraLab — "strumento / dossier tecnico" per l'affidabilità
# Graphite + ottone brunito · IBM Plex Serif/Sans/Mono · griglia blueprint + grana
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400;1,500&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root{
      --il-bg:#f5f8fc; --il-bg2:#ffffff; --il-panel:#ffffff; --il-panel2:#eef4fb;
      --il-ink:#132638; --il-dim:#4a5c6e; --il-faint:#8496a8;
      --il-line:rgba(20,60,110,.10); --il-line2:rgba(20,60,110,.16);
      --il-blue:#1f6fb2; --il-blue-deep:#0b4f86; --il-celeste:#4aa3df; --il-celeste-soft:#7cc4e8;
      --il-tint:#eaf3fb; --il-green:#2e9e6b; --il-amber:#e0902f; --il-red:#d1544a;
    }

    /* ---------- Superficie principale: chiara, griglia blueprint tenue + alone celeste --------- */
    .stApp{
        background:var(--il-bg);
        color:var(--il-ink);
        background-image:
            radial-gradient(1100px 560px at 88% -14%, rgba(74,163,223,.08), transparent 60%),
            radial-gradient(760px 520px at -4% 112%, rgba(31,111,178,.045), transparent 55%);
        background-attachment:fixed;
    }
    .block-container{position:relative; z-index:1;}
    html, body, [class*="css"]{ font-family:'IBM Plex Sans', sans-serif; }

    /* ---------- Sidebar: pannello bianco, bordo blu tenue ---------- */
    [data-testid="stSidebar"]{
        background:var(--il-bg2);
        border-right:1px solid var(--il-line2);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3{
        font-family:'IBM Plex Mono', monospace; color:var(--il-blue) !important;
        text-transform:uppercase; letter-spacing:.16em; font-size:.78rem !important; font-weight:600 !important;
    }

    /* ---------- Tipografia ---------- */
    h1{ font-family:'IBM Plex Serif', serif !important; font-weight:600 !important;
        color:var(--il-blue-deep) !important; letter-spacing:-.01em; }
    h2{ font-family:'IBM Plex Serif', serif !important; font-weight:500 !important; color:var(--il-ink) !important; }
    h3{ font-family:'IBM Plex Sans', sans-serif !important; font-weight:600 !important; color:var(--il-blue) !important;
        letter-spacing:.01em; }
    .block-container p, .block-container li, .block-container label{ color:var(--il-dim); }

    /* ---------- Metriche = schede-strumento chiare ---------- */
    [data-testid="stMetric"]{
        background:var(--il-panel);
        border:1px solid var(--il-line2); border-radius:6px; padding:16px 18px;
        position:relative; overflow:hidden; box-shadow:0 1px 3px rgba(20,60,110,.05);
    }
    [data-testid="stMetric"]::after{
        content:""; position:absolute; left:0; top:0; width:3px; height:100%; background:var(--il-celeste);
    }
    [data-testid="stMetricValue"]{
        font-family:'IBM Plex Mono', monospace !important; font-weight:600 !important;
        font-size:1.7rem !important; color:var(--il-blue-deep) !important;
        font-variant-numeric:tabular-nums; letter-spacing:-.01em;
        -webkit-text-fill-color:var(--il-blue-deep);
    }
    [data-testid="stMetricLabel"]{
        font-family:'IBM Plex Mono', monospace !important; color:var(--il-faint) !important;
        text-transform:uppercase; letter-spacing:.14em; font-size:.68rem !important;
    }
    [data-testid="stMetricDelta"]{ font-family:'IBM Plex Mono', monospace !important; font-size:.74rem !important; }

    /* ---------- Container / schede ---------- */
    [data-testid="stVerticalBlockBorderWrapper"]{
        background:var(--il-panel) !important;
        border:1px solid var(--il-line2) !important; border-radius:6px !important;
        box-shadow:0 1px 3px rgba(20,60,110,.05);
        transition:border-color .25s, box-shadow .25s;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover{
        border-color:var(--il-celeste) !important; box-shadow:0 4px 14px rgba(31,111,178,.10);
    }

    /* ---------- Pulsanti ---------- */
    .stButton > button{
        font-family:'IBM Plex Mono', monospace !important; letter-spacing:.08em;
        text-transform:uppercase; font-size:.78rem !important; font-weight:500 !important;
        border-radius:4px !important; transition:all .18s;
    }
    .stButton > button[kind="primary"]{
        background:var(--il-blue) !important; color:#ffffff !important; border:1px solid var(--il-blue) !important;
        font-weight:600 !important; box-shadow:0 2px 6px rgba(31,111,178,.25);
    }
    .stButton > button[kind="primary"]:hover{ background:var(--il-blue-deep) !important; transform:translateY(-1px); }
    .stButton > button:not([kind="primary"]){
        background:var(--il-bg2) !important; color:var(--il-blue) !important;
        border:1px solid var(--il-line2) !important;
    }
    .stButton > button:not([kind="primary"]):hover{
        border-color:var(--il-blue) !important; color:var(--il-blue-deep) !important; background:var(--il-tint) !important;
    }
    [data-testid="stSidebar"] .stButton > button{ text-align:left; }

    /* ---------- Progress ---------- */
    .stProgress > div > div > div{
        border-radius:3px; background:linear-gradient(90deg, var(--il-red), var(--il-amber), var(--il-green));
    }

    /* ---------- Expander ---------- */
    details, .streamlit-expanderHeader{
        background:var(--il-panel2) !important; border:1px solid var(--il-line) !important; border-radius:5px !important;
    }
    .streamlit-expanderHeader{
        font-family:'IBM Plex Mono', monospace !important; color:var(--il-blue) !important;
        text-transform:uppercase; letter-spacing:.1em; font-size:.78rem !important;
    }

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrameContainer"], [data-testid="stTable"]{
        border-radius:6px; border:1px solid var(--il-line2);
    }

    /* ---------- Input ---------- */
    [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div{
        background:var(--il-bg2) !important; border-color:var(--il-line2) !important; border-radius:4px !important;
    }
    .stSelectbox > div > div,
    .stNumberInput [data-baseweb="input"],
    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea{
        background:var(--il-bg2) !important; color:var(--il-ink) !important;
        border-color:var(--il-line2) !important; border-radius:4px !important;
        font-family:'IBM Plex Mono', monospace !important;
    }
    .stNumberInput button{
        background:var(--il-panel2) !important; color:var(--il-blue) !important;
        border-color:var(--il-line2) !important;
    }
    .stNumberInput button:hover{ color:var(--il-blue-deep) !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{ background:var(--il-blue) !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div{ background:var(--il-celeste) !important; }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"]{ background:transparent; gap:2px; border-bottom:1px solid var(--il-line2); }
    .stTabs [data-baseweb="tab"]{
        font-family:'IBM Plex Mono', monospace !important; color:var(--il-faint) !important;
        text-transform:uppercase; letter-spacing:.1em; font-size:.76rem !important; font-weight:500;
        border-radius:0 !important; border-bottom:2px solid transparent;
    }
    .stTabs [aria-selected="true"]{
        background:transparent !important; color:var(--il-blue) !important;
        border-bottom:2px solid var(--il-blue) !important;
    }

    /* ---------- Alert ---------- */
    [data-testid="stAlert"]{ border-radius:6px !important; border-left:3px solid var(--il-celeste) !important; }

    /* ---------- Blocchi identità (hero, chip, righelli) ---------- */
    .il-eyebrow{ font-family:'IBM Plex Mono', monospace; font-size:.72rem; letter-spacing:.32em;
        text-transform:uppercase; color:var(--il-blue); display:flex; align-items:center; gap:14px;
        opacity:0; animation:ilrise .7s .05s forwards; }
    .il-eyebrow::before{ content:""; width:36px; height:1px; background:var(--il-celeste); display:inline-block; }
    .il-hero-title{ font-family:'IBM Plex Serif', serif; font-weight:600; font-size:clamp(38px,6vw,66px);
        line-height:1.02; letter-spacing:-.015em; margin:20px 0 0; color:var(--il-blue-deep);
        opacity:0; animation:ilrise .7s .12s forwards; }
    .il-hero-title em{ font-style:italic; font-weight:500; color:var(--il-celeste); }
    .il-lede{ max-width:660px; margin:20px 0 0; font-size:16px; line-height:1.62; color:var(--il-dim);
        font-weight:400; opacity:0; animation:ilrise .7s .2s forwards; }
    .il-lede b{ color:var(--il-ink); font-weight:600; }
    .il-topmeta{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.14em;
        text-transform:uppercase; color:var(--il-faint); }
    .il-topmeta b{ color:var(--il-blue); font-weight:500; }
    .il-rule{ height:1px; background:var(--il-line2); margin:34px 0; }
    .il-cardmeta{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:.1em;
        text-transform:uppercase; color:var(--il-faint); }
    .il-cardmeta .arw{ color:var(--il-blue); }
    .il-foot{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.1em;
        text-transform:uppercase; color:var(--il-faint); }

    @keyframes ilrise{ from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }

    /* Aspetto da prodotto: header/toolbar trasparenti, niente barra bianca */
    #MainMenu{ visibility:hidden; }
    footer{ visibility:hidden; }
    [data-testid="stHeader"]{ background:transparent !important; }
    [data-testid="stToolbar"]{ right:8px; }
    [data-testid="stDecoration"]{ display:none !important; }
    .block-container{ padding-top:2.4rem !important; }
    </style>
    """, unsafe_allow_html=True)


def render_brand_sidebar(subtitle: str = ""):
    """Marchio InfraLab per la sidebar (sostituisce il logo generico)."""
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:11px; padding:6px 2px 14px;
                    border-bottom:1px solid var(--il-line2); margin-bottom:14px;">
          <svg width="30" height="30" viewBox="0 0 34 34" fill="none" style="flex:none;">
            <rect x="1" y="1" width="32" height="32" rx="3" stroke="#1f6fb2" stroke-width="1.3"/>
            <path d="M6 24 L14 10 L20 20 L24 14 L28 24" stroke="#0b4f86" stroke-width="1.4"
                  fill="none" stroke-linejoin="round"/>
            <line x1="6" y1="24" x2="28" y2="24" stroke="#4aa3df" stroke-width="1.2"/>
            <circle cx="14" cy="10" r="1.7" fill="#4aa3df"/>
          </svg>
          <div style="font-family:'IBM Plex Serif',serif; font-weight:600; font-size:19px;
                      letter-spacing:.02em; color:#0b4f86;">Infra<span style="color:#1f6fb2;">Lab</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.sidebar.markdown(
            f"<div class='il-topmeta' style='margin:-6px 2px 12px;'>{subtitle}</div>",
            unsafe_allow_html=True,
        )


def run_turn(env: BridgeEnv, session_id: str):
    history_dfs = persistence.load_history(session_id)
    plans = {}
    for name in env.active_agents:
        agent_config = env.agent_config.get(name, {})
        agent = agent_factory(name, agent_config)

        decide_kwargs = {"session_id": session_id}
        sandboxed = sandbox_enabled(env)
        if isinstance(agent, LLMAgent):
            # policy_sandbox: la memoria KPI (salute, beta, Pf di rete) e' calcolata sullo
            # stato vero; resta solo quella contabile (budget, ricavi, costi).
            decide_kwargs["history_dfs"] = ({"budget": history_dfs.get("budget")} if sandboxed
                                            else history_dfs)

        try:
            plan, explanation_data = agent.decide(policy_env(env, name), **decide_kwargs)
        except SandboxViolation as e:
            plan, explanation_data = {}, ("sandbox_violation", str(e))

        if isinstance(agent, LLMAgent) and explanation_data:
            prompt, response_text = explanation_data
            log_llm_file = os.path.join(persistence.get_session_path(session_id), "llm_interactions.log")
            try:
                log_entry = {
                    "session_id": session_id, "timestamp": datetime.now().isoformat(), "agent_name": name,
                    "profile": getattr(agent, 'profile', 'N/A'), "quarter": env.quarter,
                    "prompt": prompt, "response": response_text
                }
                with open(log_llm_file, "a", encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, cls=persistence.NumpyJSONEncoder) + "\n")
            except Exception as e:
                print(f"Errore logging LLM: {e}")

        plans[name] = plan

    action_costs, revenues = env.step(plans)
    persistence.log_history(env, session_id, action_costs, revenues)
    persistence.save_game_state(env, session_id)

    player_actions_file = os.path.join(persistence.get_session_path(session_id), persistence.PLAYER_ACTIONS_FILENAME)
    if os.path.exists(player_actions_file):
        os.remove(player_actions_file)


def render_bridge_card(bridge_data: dict, maint_ops: dict, env, show_debug: bool = True):
    b = bridge_data
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"🌉 {b['name']} (ID: {b['id']})")
            status_color = "green" if b['status'] == 'Operativo' else "orange"
            last_insp_q = b.get('last_inspection_quarter', 0)
            info_cols = st.columns(3)
            info_cols[0].markdown(f"**Stato:** :{status_color}[{b['status']}]")
            info_cols[1].markdown(f"**CdA:** {b['cda']}")
            if last_insp_q > 0:
                info_cols[2].markdown(f"**Ult. Ispezione:** T{last_insp_q}")
            if b.get('reliability_index') is not None:
                beta_post = b.get('posterior_reliability_index') if b.get('posterior_reliability_index') is not None else b['reliability_index']
                pf_post = b.get('posterior_failure_probability') if b.get('posterior_failure_probability') is not None else b.get('failure_probability', 0.0)
                beta_ann = b.get('annual_reliability_index')
                beta_ann_post = b.get('posterior_annual_reliability_index') if b.get('posterior_annual_reliability_index') is not None else beta_ann
                rel_details = b.get('posterior_reliability_details') or b.get('reliability_details') or {}
                target_beta_ref = rel_details.get('target_beta_reference')
                target_beta_ann = rel_details.get('target_beta_annual')
                cc = rel_details.get('consequence_class')
                st.caption(
                    f"Beta prior: {b['reliability_index']:.2f} | "
                    f"Beta post: {beta_post:.2f} | "
                    f"Beta ann post: {beta_ann_post:.2f} | "
                    f"Pf post: {pf_post:.2e}"
                )
                if target_beta_ref is not None and target_beta_ann is not None:
                    st.caption(
                        f"Target {cc or 'CC'}: beta_ref={target_beta_ref:.2f}, "
                        f"beta_ann={target_beta_ann:.2f}"
                    )
            if b.get('traffic_fragility'):
                tf = b['traffic_fragility']
                st.caption(
                    f"Traffic fragility ({tf.get('traffic_model')}): "
                    f"eta={tf.get('eta', 0):.2f}, beta={tf.get('beta', 0):.2f}, "
                    f"lambda_ann={tf.get('annual_failure_rate', 0):.2e}"
                )
            if b.get('seismic_fragility'):
                sf = b['seismic_fragility']
                st.caption(
                    f"Seismic fragility ({sf.get('bridge_class')}, {sf.get('limit_state')}): "
                    f"P={sf.get('failure_probability', 0):.2e} @ PGA={sf.get('controlling_pga', 0):.3f}g"
                )
            if b.get('last_proof_load_report'):
                report = b['last_proof_load_report']
                outcome = "superata" if report.get('passed') else "non superata"
                st.caption(f"Proof load T{report.get('quarter')}: α={report.get('alpha', 0):.2f}, {outcome}.")
            if b.get('last_management_feedback'):
                feedback = b['last_management_feedback']
                st.caption(f"Feedback gestione T{feedback.get('quarter')}: {feedback.get('source')}.")
            for _d in b.get('known_progressive_defects') or []:
                st.warning(f"Difetto in sviluppo noto ({_d.get('type')}), scoperto al T{_d.get('known_since')}: "
                           "arrestabile con l'azione 'arresto_difetto'.")
            if b.get('network_importance_factor') is not None:
                st.caption(
                    f"Importanza rete: {b.get('network_importance_factor', 0):.2f} | "
                    f"Centralità: {b.get('network_centrality', 0):.2f} | "
                    f"Risk priority: {b.get('risk_priority_index', 0):.3e}"
                )
        with col2:
            is_under_maintenance = b['status'] != 'Operativo'
            feasibility = b.get('action_feasibility', {})
            _TYPE_ICON = {'analysis': '🔍', 'monitoring': '📡', 'proof_load': '⚖️', 'intervention': '🔧'}

            # Build labeled options dict
            action_display: dict = {"None": "— Nessuna azione"}
            for action_name, op_cfg in maint_ops.items():
                meta = feasibility.get(action_name, {})
                cost = meta.get('cost', 0)
                feasible = meta.get('feasible', False)
                reason = meta.get('reason', '')
                icon = _TYPE_ICON.get(op_cfg.get('type', ''), '📌')
                if feasible:
                    action_display[action_name] = f"{icon} {action_name}  (€ {cost:,.0f})"
                elif reason == 'insufficient_budget' and op_cfg.get('type') == 'intervention':
                    # Always show interventions so player knows they exist
                    action_display[action_name] = f"{icon} {action_name}  (€ {cost:,.0f}) ⚠️ budget"
                # Other blocks (bridge_busy, cooldown…) stay hidden

            action = st.selectbox(
                "Azione",
                list(action_display.keys()),
                key=f"action_{b['id']}",
                disabled=is_under_maintenance,
                format_func=lambda x: action_display.get(x, x),
                label_visibility="collapsed",
            )

            # Warn if selected action is budget-blocked
            sel_meta = feasibility.get(action, {})
            if action != "None" and not sel_meta.get('feasible') and sel_meta.get('reason') == 'insufficient_budget':
                st.caption("⚠️ Budget insufficiente — verrà bloccato al submit.")

            # Show non-budget procedural blocks
            proc_blocked = {n: m.get('reason') for n, m in feasibility.items()
                            if not m.get('feasible') and m.get('reason') not in ('insufficient_budget', None)}
            if proc_blocked:
                st.caption("Blocchi: " + ", ".join(f"{k}={v}" for k, v in proc_blocked.items()))

            st.divider()
            hi = b.get('health_index')
            nav = b.get('net_asset_value')
            if hi is not None:
                st.metric("Indice di Salute (HI)", f"{hi:.1%}")
            if nav is not None:
                st.metric("Valore Netto Asset (NAV)", f"€ {nav:,.0f}")

        with st.expander("🔬 Dettaglio Componenti e Previsioni"):
            for comp_name, comp_data in b['components'].items():
                st.markdown(f"**Componente: {comp_name.capitalize()}**")
                obs_min, obs_max = comp_data.get('observed_condition_min'), comp_data.get('observed_condition_max')
                if obs_min is not None and obs_max is not None:
                    st.progress((obs_min + obs_max) / 2,
                                text=f"Affidabilità Osservata: [{obs_min:.1%} - {obs_max:.1%}]")
                else:
                    st.progress(0, text="Affidabilità Osservata: Sconosciuta")

            if b.get('future_curve_range'):
                st.markdown("📈 **Andamento previsto con monitoraggio attivo:**")
                reshaped_data = {}
                for component_name, values in b['future_curve_range'].items():
                    for quarter, c_min, c_max in values:
                        if quarter not in reshaped_data:
                            reshaped_data[quarter] = {}
                        reshaped_data[quarter][f'{component_name}_min'] = c_min
                        reshaped_data[quarter][f'{component_name}_max'] = c_max

                if reshaped_data:
                    chart_df = pd.DataFrame.from_dict(reshaped_data, orient='index').sort_index()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    component_names = env.component_weights.keys()
                    palette = {name: color for name, color in zip(component_names, UNINA_PALETTE)}

                    for comp_name in component_names:
                        min_col, max_col = f'{comp_name}_min', f'{comp_name}_max'
                        if min_col in chart_df.columns and max_col in chart_df.columns:
                            ax.fill_between(chart_df.index, chart_df[min_col], chart_df[max_col],
                                            color=palette.get(comp_name, 'gray'), alpha=0.2)
                            ax.plot(chart_df.index, (chart_df[min_col] + chart_df[max_col]) / 2,
                                    color=palette.get(comp_name, 'gray'), linestyle='--', label=f'Media - {comp_name}')

                    ax.set_ylim(0, 1.05)
                    ax.set_xlabel("Trimestre Futuro")
                    ax.set_ylabel("Affidabilità Prevista")
                    ax.set_title("Previsione di Degrado con Monitoraggio")
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    fig.tight_layout(rect=[0, 0, 0.8, 1])
                    st.pyplot(fig)

        if show_debug:
            with st.expander("🕵️ Informazioni di Debug (Stato Interno)"):
                st.json(b)
    return action


# ============================================================
# Experiment support — tutorial, game, questionnaire, debrief
# ============================================================

_TUTORIAL_STEPS = [
    {
        "title": "Benvenuto in BridgeManager",
        "icon": "🌉",
        "items": [
            ("Il tuo ruolo", "Sei il **Responsabile della Manutenzione** di una rete di ponti. Ogni trimestre decidi quali interventi eseguire su ciascun ponte, gestendo un budget limitato."),
            ("L'obiettivo", "Massimizzare la **salute della rete** (media ponderata della condizione dei ponti) e il **patrimonio netto** al termine della simulazione."),
        ],
    },
    {
        "title": "I tuoi ponti",
        "icon": "📊",
        "items": [
            ("Condizione (0–1)", "Ogni componente ha una condizione da 0 (collasso) a 1 (perfetto). **Non la conosci** finché non la misuri con un'ispezione."),
            ("Asimmetria informativa", "⚠️ La condizione **osservata** (post-ispezione) può differire da quella reale — l'ispezione ha incertezza. Solo le verifiche più accurate (L4) si avvicinano alla realtà."),
            ("Classe di Attenzione (CdA)", "Priorità assegnata automaticamente: da **Bassa** (buono) ad **Alta** (critico). Influenza la frequenza obbligatoria di ispezione."),
            ("Indice di Affidabilità β", "Misura il margine strutturale di sicurezza. Se β scende sotto il target → il ponte richiede attenzione urgente."),
        ],
    },
    {
        "title": "Le azioni disponibili",
        "icon": "🔧",
        "items": [
            ("Nessuna azione", "Il ponte degrada naturalmente. Nessun costo."),
            ("L1 Ispezione  ≈ 1 €/m²", "Ispezione visiva rapida. Rivela la condizione con incertezza moderata. Obbligatoria per rispettare le linee guida."),
            ("L4 Verifica  ≈ 100 €/m²", "Verifica strutturale completa. Alta precisione, ma costosa."),
            ("Proof Load Test  ≈ 25 €/m²", "Test di carico diretto. Aggiorna l'affidabilità strutturale. Rischio minimo di danno se il ponte è in buone condizioni."),
            ("Monitoraggio  ≈ 70 €/m²", "Installa sensori per tracciare il degrado nel tempo e proiettare il futuro."),
            ("Interventi strutturali  — migliaia €/m²", "Riparazioni o sostituzioni di componenti (impalcato, pile, spalle, appoggi). Migliorano la condizione ma richiedono più trimestri e riducono i ricavi."),
        ],
    },
    {
        "title": "Regole economiche",
        "icon": "💰",
        "items": [
            ("Budget", "Ogni trimestre ricevi ricavi dai ponti operativi. I costi di manutenzione vengono detratti. Il saldo si accumula nei trimestri successivi."),
            ("Eliminazione", "Se il budget scende sotto zero per **due trimestri consecutivi**, sei eliminato dalla simulazione."),
            ("Ricavi ridotti", "Un ponte in manutenzione attiva genera ricavi ridotti durante i lavori."),
            ("Backlog ispezioni", "Con le linee guida attive, ogni ponte ha una scadenza di ispezione. Il ritardo accumula **backlog** che penalizza il punteggio finale."),
        ],
    },
    {
        "title": "Consigli pratici",
        "icon": "💡",
        "items": [
            ("Ponti critici", "I ponti più vecchi o ad alto traffico degradano più rapidamente — non trascurarli."),
            ("Ispezioni ≠ miglioramento", "Ispezionare ti dice *dove sei*, non migliora la condizione. Bilancia conoscenza e intervento."),
            ("Pianifica in anticipo", "Gli interventi pesanti durano più trimestri. Avviali prima che la situazione diventi critica."),
            ("Budget residuo", "Il budget inutilizzato matura interessi, ma i ponti degradati perdono valore. Scegli con cura."),
        ],
    },
]


def _set_exp_phase(session_id: str, phase: str):
    st.session_state.exp_phase = phase
    persistence.save_experiment_metadata(session_id, experiment_phase=phase)


def _display_tutorial(session_id: str, agent_name: str, max_quarters: int):
    step = st.session_state.get('exp_tutorial_step', 0)
    total = len(_TUTORIAL_STEPS)
    data = _TUTORIAL_STEPS[step]

    st.title(f"{data['icon']} {data['title']}")
    st.progress((step + 1) / total, text=f"Passo {step + 1} di {total}")
    st.divider()

    for label, text in data['items']:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(text)

    st.divider()
    col_back, _, col_next = st.columns([1, 3, 1])
    with col_back:
        if step > 0 and st.button("← Indietro", use_container_width=True):
            st.session_state.exp_tutorial_step = step - 1
            persistence.save_experiment_metadata(session_id, tutorial_step=step - 1)
            st.rerun()
    with col_next:
        if step < total - 1:
            if st.button("Avanti →", type="primary", use_container_width=True):
                st.session_state.exp_tutorial_step = step + 1
                persistence.save_experiment_metadata(session_id, tutorial_step=step + 1)
                st.rerun()
        else:
            if st.button(f"Inizia la partita ({max_quarters} trimestri) 🚀", type="primary", use_container_width=True):
                st.session_state.exp_tutorial_step = 0
                _set_exp_phase(session_id, 'playing')
                st.rerun()


def _display_game(session_id: str, agent_name: str, env, is_research: bool):
    max_q = env.params.max_quarters
    current_q = env.quarter

    if is_research and max_q and current_q > max_q:
        meta = persistence.load_experiment_metadata(session_id)
        next_phase = 'questionnaire' if meta.get('show_questionnaire') else 'done'
        _set_exp_phase(session_id, next_phase)
        st.rerun()
        return

    # Turn-timing: reset tracker when quarter advances
    if st.session_state.get('exp_tracked_quarter') != current_q:
        st.session_state.exp_tracked_quarter = current_q
        st.session_state.exp_turn_start = datetime.now()

    # policy_sandbox: anche il giocatore umano vede solo l'osservazione sanitizzata, e
    # lo storico KPI (salute e valore calcolati sullo stato vero) resta nascosto.
    sandboxed = sandbox_enabled(env)
    obs = sanitized_observation(env, agent_name) if sandboxed else env.get_agent_observations(agent_name)
    history_dfs = persistence.load_history(session_id)
    if sandboxed:
        history_dfs = {**history_dfs, "kpi": pd.DataFrame()}
    budget = obs['game_rules']['my_budget']

    st.title(f"🕹️ Dashboard di Gioco: {agent_name}")
    if is_research and max_q:
        pct = min(current_q / max_q, 1.0)
        st.progress(pct, text=f"Trimestre {current_q} / {max_q}  —  {max_q - current_q + 1} rimanenti")
    else:
        st.markdown(f"**Trimestre {current_q}** di {env.params.years * 4}")

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="💰 Budget Disponibile", value=f"€ {budget:,.0f}")
    m2.metric(label="🌉 Ponti Gestiti", value=len(obs['my_bridges']))
    observed_values = [(b.get('min_observed_condition'), b.get('initial_asset_value', 0))
                       for b in obs['my_bridges'] if b.get('min_observed_condition') is not None]
    if observed_values:
        total_w = sum(c * v for c, v in observed_values)
        total_v = sum(v for _, v in observed_values)
        m3.metric(label="❤️ Salute Osservata Rete", value=f"{total_w / total_v:.1%}" if total_v > 0 else "N/A")
    else:
        m3.metric(label="❤️ Salute Osservata Rete", value="Sconosciuta")
    total_nav = sum(b.get('net_asset_value', 0) for b in obs['my_bridges'])
    m4.metric(label="🏛️ NAV Portfolio", value=f"€ {total_nav:,.0f}")

    with st.container(border=True):
        st.subheader("Il Tuo Andamento Storico")
        df_kpi_h = history_dfs.get('kpi', pd.DataFrame())
        df_bud_h = history_dfs.get('budget', pd.DataFrame())
        if not df_kpi_h.empty and not df_bud_h.empty:
            df_kpi_h = df_kpi_h[df_kpi_h['agent'] == agent_name]
            df_bud_h = df_bud_h[df_bud_h['agent'] == agent_name]
            if not df_kpi_h.empty and not df_bud_h.empty:
                df_merged = pd.merge(df_bud_h, df_kpi_h, on=["quarter", "agent"])
                df_merged["net_worth"] = df_merged["budget"] + df_merged["total_asset_value"]
                g1, g2 = st.columns(2)
                with g1:
                    st.line_chart(df_kpi_h, x='quarter', y='health_index', color="#ff4b4b")
                with g2:
                    st.line_chart(df_merged, x='quarter', y='net_worth')
            else:
                st.info("Nessun dato storico ancora disponibile.")
        else:
            st.info("Nessun dato storico ancora disponibile.")

    st.divider()
    st.header("Pianifica i tuoi interventi")
    maintenance_plan = {}
    for b in obs['my_bridges']:
        action = render_bridge_card(b, env.maint_ops, env, show_debug=False)
        if action != "None":
            maintenance_plan[str(b['id'])] = action

    with st.sidebar:
        st.header(f"Scenario: {session_id.split('_')[0]}")
        with st.container(border=True):
            st.subheader("Riepilogo Finanziario")
            maint_cost = sum(
                _calculate_cost(env.maint_ops[a], next(b for b in obs['my_bridges'] if b['id'] == int(bid))['area'])
                for bid, a in maintenance_plan.items()
                if env.maint_ops.get(a) and any(b['id'] == int(bid) for b in obs['my_bridges'])
            )
            expected_revenue = env.get_agent_revenue_preview(agent_name)
            final_budget = budget - maint_cost + expected_revenue
            st.metric("Budget Corrente", f"€ {budget:,.0f}")
            st.metric("Costi Stimati", f"€ {-maint_cost:,.0f}")
            st.metric("Incasso Previsto", f"€ {expected_revenue:,.0f}")
            st.metric("Budget Finale Previsto", f"€ {final_budget:,.0f}",
                      delta=f"{final_budget - budget:,.0f} €", delta_color="normal")

        is_game_over = env.quarter > env.params.years * 4
        btn_label = "✅ Conferma Piano e Avanza" if is_research else "✅ Salva e Avanza al Prossimo Trimestre"
        if st.button(btn_label, type="primary", use_container_width=True, disabled=is_game_over):
            elapsed = (datetime.now() - st.session_state.get('exp_turn_start', datetime.now())).total_seconds()
            with st.spinner("Avanzamento simulazione..."):
                persistence.save_player_action(session_id, agent_name, {"maintenance": maintenance_plan})
                if is_research:
                    persistence.log_turn_time(session_id, current_q, agent_name, elapsed)
                run_turn(env, session_id)
                st.toast(f"Turno {env.quarter - 1} completato!", icon="✅")
            st.rerun()


def _display_questionnaire(session_id: str, agent_name: str):
    st.title("📋 Questionario di Fine Simulazione")
    st.info("Prima di vedere i risultati, rispondi a queste domande (circa 2 minuti). I dati sono anonimi e usati solo a fini di ricerca.")

    with st.form("questionnaire_form", clear_on_submit=False):
        st.subheader("1. Profilo")
        c1, c2 = st.columns(2)
        with c1:
            anni_esp = st.number_input("Anni di esperienza in ing. civile/strutturale", 0, 50, 0, step=1)
        with c2:
            ruolo = st.selectbox("Ruolo attuale", [
                "Studente magistrale",
                "Dottorando / Assegnista",
                "Ricercatore",
                "Ingegnere professionista",
                "Altro",
            ])
        ha_gestito = st.radio("Hai mai gestito o ispezionato strutture da ponte in contesti reali?",
                              ["Sì", "No"], horizontal=True)

        st.subheader("2. Esperienza nella simulazione")
        difficolta = st.slider("Difficoltà percepita dello scenario", 1, 5, 3,
                               help="1 = Molto facile  |  5 = Molto difficile")
        info_suff = st.slider("Le informazioni disponibili erano sufficienti per decidere?", 1, 5, 3,
                              help="1 = Assolutamente insufficienti  |  5 = Più che sufficienti")
        fam_affid = st.slider("Familiarità con l'analisi di affidabilità strutturale", 1, 5, 3,
                              help="1 = Nessuna  |  5 = Esperto")

        st.subheader("3. Strategia decisionale")
        strategia = st.selectbox("Quale criterio hai usato principalmente?", [
            "Priorità alla sicurezza (ponti più rischiosi prima)",
            "Priorità al budget (azioni più economiche)",
            "Bilanciamento sicurezza / costo",
            "Strategia preventiva (ispezionare sistematicamente tutto)",
            "Reattivo (intervenire solo all'emergenza)",
            "Altro",
        ])
        sistematica = st.radio("Hai pianificato una strategia sistematica in anticipo?", [
            "Sì, ho pianificato i turni in anticipo",
            "In parte",
            "No, ho deciso turno per turno",
        ])
        asimmetria = st.radio("Eri consapevole che la condizione reale può differire da quella osservata?", [
            "Sì, l'ho tenuto in considerazione nelle mie decisioni",
            "Sì, ma non ha cambiato il mio approccio",
            "No, non ne ero consapevole",
        ])
        uso_beta = st.radio("Hai usato l'indice di affidabilità (β) per prioritizzare?", [
            "Sì, sistematicamente",
            "Solo in alcuni casi",
            "No, mi sono basato sulla condizione visiva / CdA",
        ])
        testo = st.text_area("Note libere: descrivi brevemente la tua strategia (opzionale)", "", max_chars=500)

        submitted = st.form_submit_button(
            "Invia Questionario e Vedi i Risultati →", type="primary", use_container_width=True)

        if submitted:
            persistence.save_questionnaire(session_id, {
                "session_id": session_id,
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "anni_esperienza": int(anni_esp),
                "ruolo": ruolo,
                "ha_gestito_ponti": ha_gestito,
                "difficolta_percepita": difficolta,
                "info_sufficienti": info_suff,
                "familiarita_affidabilita": fam_affid,
                "strategia_principale": strategia,
                "approccio_sistematico": sistematica,
                "consapevole_asimmetria": asimmetria,
                "uso_beta": uso_beta,
                "note_libere": testo,
            })
            _set_exp_phase(session_id, 'done')
            st.rerun()


def _run_heuristic_baseline_for_debrief(scenario_id: str):
    try:
        scenarios_list = persistence.load_scenarios()
        scenario = next((s for s in scenarios_list if s['scenario_id'] == scenario_id), None)
        if not scenario:
            return None

        from infralab.calibration_profiles import apply_calibration_profile as _apply_cal
        from infralab.core.environment import BridgeEnv as _Env
        from infralab.settings.config import (SimParameters as _SP, MaintOpsConfig as _MOC,
                            CdaConfig as _CDA, DEFAULT_COMPONENTS as _DC)

        base = asdict(_SP())
        base.update(scenario.get('params', {}))
        if scenario.get('max_quarters'):
            base['max_quarters'] = scenario['max_quarters']
        params = _SP(**base)
        max_q = params.max_quarters or 20
        params.agent_names = ['Euristico']
        params.num_agents = 1

        mops = asdict(_MOC())
        params, mops = _apply_cal(params, mops, params.calibration_profile)

        env = _Env(params, mops, {'Euristico': {'type': 'Euristico'}}, _CDA(), _DC)
        bridges = scenario.get('bridges', [
            {"id": 1, "name": "Ponte", "length": 100, "width": 15, "adt": 5000,
             "eta_iniziale_anni": 20, "vita_utile_anni": 80,
             "incasso_trimestrale": 25000, "valore_asset": 5000000}
        ])
        env.initialize_bridges(bridges)

        rows = []
        agent = HeuristicAgent('Euristico')
        for _ in range(max_q):
            plan, _ = agent.decide(env)
            env.step({'Euristico': plan})
            my_bridges = list(env.bridges.values())
            if my_bridges:
                total_adt = sum(b.adt for b in my_bridges) or 1
                rows.append({
                    'quarter': env.quarter - 1,
                    'health_index': sum(b.min_condition * b.adt for b in my_bridges) / total_adt,
                    'backlog_count': sum(1 for b in my_bridges
                                        if env.params.enforce_guidelines and b.next_inspection_quarter < env.quarter),
                    'knowledge_gap_count': sum(1 for b in my_bridges if b.min_observed_condition() is None),
                    'net_worth': env.budgets.get('Euristico', 0) + sum(b.current_asset_value for b in my_bridges),
                })
        return pd.DataFrame(rows)
    except Exception:
        return None


def _display_debrief(session_id: str, agent_name: str):
    st.title("🏆 Risultati Finali")
    st.balloons()

    history_dfs = persistence.load_history(session_id)
    df_kpi = history_dfs.get('kpi', pd.DataFrame())
    df_bud = history_dfs.get('budget', pd.DataFrame())
    player_kpi = df_kpi[df_kpi['agent'] == agent_name].copy() if not df_kpi.empty else pd.DataFrame()
    player_bud = df_bud[df_bud['agent'] == agent_name].copy() if not df_bud.empty else pd.DataFrame()

    st.subheader("Il tuo risultato finale")
    if not player_kpi.empty and not player_bud.empty:
        last_k = player_kpi.iloc[-1]
        last_b = player_bud.iloc[-1]
        net_worth = last_b['budget'] + last_k.get('total_asset_value', 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Indice di Salute Finale", f"{last_k['health_index']:.1%}")
        c2.metric("Patrimonio Netto Finale", f"€ {net_worth:,.0f}")
        c3.metric("Backlog Ispezioni", int(last_k['backlog_count']))
        c4.metric("Knowledge Gap", int(last_k['knowledge_gap_count']))

    st.divider()
    st.subheader("Confronto con l'Agente Euristico (stesso scenario, stesso seed)")

    meta = persistence.load_experiment_metadata(session_id)
    scenario_id = meta.get('scenario_id')

    if scenario_id and 'debrief_baseline' not in st.session_state:
        with st.spinner("Calcolo benchmark euristico..."):
            st.session_state.debrief_baseline = _run_heuristic_baseline_for_debrief(scenario_id)

    baseline = st.session_state.get('debrief_baseline')

    if baseline is not None and not player_kpi.empty:
        p = player_kpi[['quarter', 'health_index', 'backlog_count']].copy()
        p['Agente'] = f'Tu ({agent_name})'
        b = baseline[['quarter', 'health_index', 'backlog_count']].copy()
        b['Agente'] = 'Euristico (AI)'
        combined = pd.concat([p, b], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for label, grp in combined.groupby('Agente'):
            axes[0].plot(grp['quarter'], grp['health_index'], label=label, linewidth=2, marker='o', markersize=3)
        axes[0].set_title('Indice di Salute Rete nel Tempo')
        axes[0].set_xlabel('Trimestre')
        axes[0].set_ylabel('Health Index')
        axes[0].set_ylim(0, 1)
        axes[0].legend()

        for label, grp in combined.groupby('Agente'):
            axes[1].plot(grp['quarter'], grp['backlog_count'], label=label, linewidth=2, marker='o', markersize=3)
        axes[1].set_title('Backlog Ispezioni nel Tempo')
        axes[1].set_xlabel('Trimestre')
        axes[1].set_ylabel('N. ponti in ritardo')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Benchmark non disponibile.")

    st.divider()
    st.success("Grazie per la partecipazione! I tuoi dati sono stati registrati con successo.")
    st.info("Puoi chiudere questa pagina.")


def display_player_page():
    if 'session_id' not in st.session_state or 'agent_name' not in st.session_state:
        st.error("Errore: Sessione non valida. Avvia una nuova partita dalla pagina principale.")
        st.link_button("Torna alla pagina di avvio", "/")
        return

    session_id = st.session_state.session_id
    agent_name = st.session_state.agent_name
    is_research = st.session_state.get('research_role') == 'player_playing'

    # Sync experiment phase from disk on first load
    if 'exp_phase' not in st.session_state:
        meta = persistence.load_experiment_metadata(session_id)
        st.session_state.exp_phase = meta.get('experiment_phase', 'tutorial')
        st.session_state.exp_tutorial_step = meta.get('tutorial_step', 0)

    phase = st.session_state.exp_phase if is_research else 'playing'

    # Load env (needed for tutorial max_quarters and game logic)
    env = persistence.load_game_state(session_id)
    if not env:
        st.error(f"Impossibile caricare lo stato della partita per la sessione {session_id}.")
        return

    # Handle natural game-over (budget eliminated / years exhausted)
    if phase == 'playing' and agent_name not in env.active_agents:
        if is_research:
            meta = persistence.load_experiment_metadata(session_id)
            next_phase = 'questionnaire' if meta.get('show_questionnaire') else 'done'
            _set_exp_phase(session_id, next_phase)
            st.rerun()
            return
        # Free-sim mode fallback
        st.balloons()
        st.success("Partita Terminata!")
        history_dfs = persistence.load_history(session_id)
        final_df = analysis.create_summary_df(env, history_dfs)
        player_row = final_df[final_df['Agente'] == agent_name]
        if not player_row.empty:
            r = player_row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Stato Finale", r['Stato'])
            c2.metric("Patrimonio Netto", f"€ {r['Patrimonio Netto Complessivo']:,.0f}")
            c3.metric("Indice di Salute", f"{r['Indice di Salute']:.2%}")
        return

    if phase == 'tutorial':
        max_q = env.params.max_quarters or (env.params.years * 4)
        _display_tutorial(session_id, agent_name, max_q)
    elif phase == 'questionnaire':
        _display_questionnaire(session_id, agent_name)
    elif phase == 'done':
        _display_debrief(session_id, agent_name)
    else:
        _display_game(session_id, agent_name, env, is_research)


def display_experiment_dashboard():
    st.title("Dashboard Esperimento — Admin")
    render_brand_sidebar("Protocol · Console amministratore")
    if st.sidebar.button("← Torna alla selezione modalità"):
        keys_to_clear = ['research_role', 'app_mode', 'participants', 'aggregated_results', 'session_to_analyze']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["📊 Monitoraggio Partecipanti", "⚙️ Gestione Scenari", "📈 Risultati Aggregati"])

    try:
        scenarios_data = persistence.load_scenarios(validate=False)
    except Exception as e:
        st.error(f"Impossibile leggere '{persistence.SCENARIOS_FILE}': {e}")
        return
    for scenario in scenarios_data:
        try:
            for problem in persistence.validate_scenario(scenario):
                st.warning(problem)
        except persistence.ScenarioValidationError as e:
            st.error(f"Scenario non valido (correggere scenarios.json): {e}")

    if 'participants' not in st.session_state:
        st.session_state.participants = persistence.load_participants()

    with tab1:
        st.header("Monitoraggio e Gestione Partecipanti")

        with st.form("participants_form"):
            st.subheader("Aggiungi o Aggiorna Lista Partecipanti")
            participants_str = st.text_area("Un nome per riga", value="\n".join(st.session_state.participants),
                                            height=200)
            base_url = st.text_input("URL Base dell'Applicazione", value="http://localhost:8501")

            if st.form_submit_button("Salva Lista e Genera/Aggiorna Tabella"):
                st.session_state.participants = [name.strip() for name in participants_str.split('\n') if name.strip()]
                persistence.save_participants(st.session_state.participants)
                st.success("Lista partecipanti salvata!")

        st.divider()
        st.header("Tabella di Monitoraggio")

        participants = st.session_state.get('participants', [])
        if participants and scenarios_data:
            monitoring_data = []
            for scenario in scenarios_data:
                for participant in participants:
                    session_id = persistence.find_session(scenario['scenario_id'], participant)
                    status = "Non Iniziata"
                    if session_id:
                        if session_id.endswith("_ARCHIVED"):
                            status = "Archiviata"
                        else:
                            env = persistence.load_game_state(session_id)
                            if env:
                                is_failed = env.last_failure_info and env.last_failure_info.get(
                                    'agent_name') == participant
                                exp_meta = persistence.load_experiment_metadata(session_id)
                                is_finished = (
                                    env.quarter > env.params.years * 4
                                    or participant not in env.active_agents
                                    or (env.params.max_quarters and env.quarter > env.params.max_quarters)
                                    or exp_meta.get('experiment_phase') in ('questionnaire', 'done')
                                )
                                exp_phase = exp_meta.get('experiment_phase', '')
                                if is_failed:
                                    status = "Fallito"
                                elif exp_phase == 'done':
                                    status = "Completata (con questionario)"
                                elif exp_phase == 'questionnaire':
                                    status = "In Corso — Questionario"
                                elif is_finished:
                                    status = "Completata"
                                else:
                                    status = f"In Corso (T {env.quarter})"
                            else:
                                status = "Errore Caricamento"

                    from infralab.ui.auth import issue_player_token
                    token = issue_player_token(scenario['scenario_id'], participant)
                    link = f"{base_url}/?token={token}"
                    monitoring_data.append({
                        "Partecipante": participant, "Scenario": scenario['scenario_name'], "Stato": status,
                        "Link di Gioco": link, "Link Testo": link, "Session ID": session_id if session_id else ""
                    })

            if monitoring_data:
                df_monitor = pd.DataFrame(monitoring_data)
                st.dataframe(df_monitor,
                             column_config={
                                 "Link di Gioco": st.column_config.LinkColumn("Link Partita", display_text="Avvia 🚀"),
                                 "Link Testo": st.column_config.TextColumn("URL (copia e incolla)"),
                             },
                             use_container_width=True, hide_index=True
                             )

                st.subheader("Gestione Sessioni")
                sessions_to_manage = df_monitor[df_monitor['Stato'].isin(["Completata", "Fallito"])]
                if not sessions_to_manage.empty:
                    valid_sessions = [sid for sid in sessions_to_manage['Session ID'] if sid]
                    if valid_sessions:
                        selected_session_id = st.selectbox("Seleziona una sessione da archiviare:",
                                                           options=valid_sessions)
                        if st.button("Archivia Sessione Selezionata"):
                            if selected_session_id and isinstance(selected_session_id, str):
                                persistence.archive_session(selected_session_id)
                                st.success(f"Sessione {selected_session_id} archiviata.")
                                st.rerun()
                    else:
                        st.info("Nessuna sessione completata o fallita con un ID valido da gestire.")
                else:
                    st.info("Nessuna sessione completata o fallita da gestire.")

        elif participants:
            st.warning("Nessuno scenario trovato. Aggiungi degli scenari nella scheda 'Gestione Scenari'.")
        else:
            st.info("Inserisci i nomi dei partecipanti nel modulo qui sopra e clicca 'Salva' per generare la tabella.")

    with tab2:
        st.header("Gestione Scenari di Gioco")

        _ALL_ACTIONS = [
            "L1_inspect", "L4_verify", "monitoring", "proof_load_test",
            "replace_appoggi", "replace_spalle", "replace_pile", "replace_impalcato",
            "ripristino_strutturale", "ripristino_completo",
        ]

        updated_scenarios = list(scenarios_data)
        any_changed = False

        for idx, scenario in enumerate(updated_scenarios):
            sid = scenario.get("scenario_id", f"Scenario {idx+1}")
            sname = scenario.get("scenario_name", sid)
            with st.expander(f"**{sid}** — {sname}", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_name = st.text_input("Nome scenario", value=sname, key=f"sname_{idx}")
                    new_budget = st.number_input("Budget iniziale (€)", value=int(scenario.get("params", {}).get("initial_budget", 1000000)), step=50000, key=f"budget_{idx}")
                    new_quarters = st.number_input("Trimestri massimi", value=int(scenario.get("max_quarters") or 20), min_value=1, max_value=80, key=f"quarters_{idx}")
                with col_b:
                    new_use_analysis = st.checkbox("Usa in analisi", value=bool(scenario.get("use_in_analysis", False)), key=f"analysis_{idx}")
                    new_show_q = st.checkbox("Mostra questionario", value=bool(scenario.get("show_questionnaire", False)), key=f"showq_{idx}")

                current_allowed = scenario.get("allowed_actions", _ALL_ACTIONS)
                new_allowed = st.multiselect(
                    "Azioni consentite",
                    options=_ALL_ACTIONS,
                    default=[a for a in current_allowed if a in _ALL_ACTIONS],
                    key=f"allowed_{idx}",
                    help="Seleziona le azioni disponibili per questo scenario. Deseleziona 'proof_load_test' per escluderla.",
                )

                if st.button(f"Salva '{sid}'", key=f"save_{idx}"):
                    updated_scenarios[idx] = dict(scenario)
                    updated_scenarios[idx]["scenario_name"] = new_name
                    updated_scenarios[idx]["max_quarters"] = new_quarters
                    updated_scenarios[idx]["use_in_analysis"] = new_use_analysis
                    updated_scenarios[idx]["show_questionnaire"] = new_show_q
                    updated_scenarios[idx]["allowed_actions"] = new_allowed
                    params = dict(scenario.get("params", {}))
                    params["initial_budget"] = new_budget
                    updated_scenarios[idx]["params"] = params
                    with open(persistence.SCENARIOS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(updated_scenarios, f, indent=2, ensure_ascii=False)
                    st.success(f"Scenario '{sid}' salvato.")
                    st.rerun()

        st.divider()
        st.caption("Per modificare ponti o parametri avanzati, edita direttamente `scenarios.json`.")

    with tab3:
        st.header("Analisi Risultati Aggregati")
        if st.button("Carica e Analizza Tutti i Risultati"):
            if os.path.exists(persistence.SESSIONS_DIR):
                all_sessions = [d for d in os.listdir(persistence.SESSIONS_DIR) if not d.endswith('_ARCHIVED')]
                all_summary_data = []
                with st.spinner("Aggregazione dati in corso..."):
                    for session_id in all_sessions:
                        env = persistence.load_game_state(session_id)
                        history_dfs = persistence.load_history(session_id)
                        if env:
                            summary_df = analysis.create_summary_df(env, history_dfs)
                            summary_df['session_id'] = session_id
                            all_summary_data.append(summary_df)
                if all_summary_data:
                    final_df = pd.concat(all_summary_data, ignore_index=True)
                    st.session_state.aggregated_results = final_df
                else:
                    st.warning("Nessun dato di riepilogo trovato nelle sessioni attive.")

        if 'aggregated_results' in st.session_state:
            st.dataframe(st.session_state.aggregated_results)
            st.download_button(
                label="📥 Scarica Risultati Aggregati (CSV)",
                data=st.session_state.aggregated_results.to_csv(index=False).encode('utf-8'),
                file_name="risultati_aggregati_esperimento.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Questionari dei Partecipanti")
        if st.button("Carica Tutti i Questionari"):
            if os.path.exists(persistence.SESSIONS_DIR):
                all_sessions = [d for d in os.listdir(persistence.SESSIONS_DIR) if not d.endswith('_ARCHIVED')]
                q_rows = []
                for sid in all_sessions:
                    q = persistence.load_questionnaire(sid)
                    if q:
                        meta = persistence.load_experiment_metadata(sid)
                        q['scenario_id'] = meta.get('scenario_id', '')
                        q_rows.append(q)
                if q_rows:
                    df_q = pd.DataFrame(q_rows)
                    st.session_state.aggregated_questionnaires = df_q
                    st.success(f"Caricati {len(df_q)} questionari.")
                else:
                    st.warning("Nessun questionario trovato.")

        if 'aggregated_questionnaires' in st.session_state:
            st.dataframe(st.session_state.aggregated_questionnaires)
            st.download_button(
                label="📥 Scarica Questionari (CSV)",
                data=st.session_state.aggregated_questionnaires.to_csv(index=False).encode('utf-8'),
                file_name="questionari_partecipanti.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Esporta Tutto (ZIP)")
        if st.button("Genera Archivio ZIP di tutte le sessioni"):
            if os.path.exists(persistence.SESSIONS_DIR):
                all_sessions = [d for d in os.listdir(persistence.SESSIONS_DIR)]
                zip_bytes = persistence.create_zip_archive(all_sessions)
                st.download_button(
                    label="📦 Scarica ZIP Sessioni",
                    data=zip_bytes,
                    file_name=f"sessioni_esperimento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )


def display_setup_page():
    st.header("⚙️ 1. Configurazione Simulazione")

    # Inizializza i parametri solo se non esistono
    if 'params' not in st.session_state:
        st.session_state.params = SimParameters()
        st.session_state.params.agent_names = [f"Manager {i + 1}" for i in range(st.session_state.params.num_agents)]

    params = st.session_state.params

    with st.form("setup_form"):
        st.subheader("Giocatori e Parametri di Gioco")

        c1, c2 = st.columns(2)
        with c1:
            num_agents = st.number_input("Numero di Giocatori", min_value=1, max_value=6, value=params.num_agents)

        # La modifica del numero di agenti richiede un refresh per aggiornare i campi dei nomi
        if num_agents != params.num_agents:
            params.agent_names = [f"Manager {i + 1}" for i in range(num_agents)]
            params.num_agents = num_agents
            st.rerun()

        new_agent_names = params.agent_names[:]
        cols = st.columns(num_agents)
        for i, col in enumerate(cols):
            if i < len(new_agent_names):
                new_agent_names[i] = col.text_input(f"Nome Giocatore {i + 1}", value=new_agent_names[i],
                                                    key=f"agent_name_setup_{i}")

        st.divider()

        profile_names = list_calibration_profiles()
        current_profile = params.calibration_profile if params.calibration_profile in profile_names else profile_names[0]
        selected_profile = st.selectbox(
            "Profilo di calibrazione",
            profile_names,
            index=profile_names.index(current_profile),
            help="Preset replicabili per gli esperimenti paper-oriented.",
        )
        profile_info = get_calibration_profile(selected_profile)
        st.caption(profile_info.get("description", ""))
        preview_params = deepcopy(params)
        preview_params, preview_ops = apply_calibration_profile(preview_params, asdict(MaintOpsConfig()), selected_profile)

        c1, c2, c3 = st.columns(3)
        initial_budget = c1.number_input("Budget Iniziale (€)", value=preview_params.initial_budget, step=100000)
        years = c1.number_input("Durata (Anni)", value=preview_params.years, min_value=1, max_value=100)
        seed = c1.number_input("Seed per la Casualità (0 per random)", value=preview_params.seed, min_value=0)

        unallocated_budget_return_annual = c2.slider("Rendimento annuo budget (%)", 0.0, 10.0,
                                                     preview_params.unallocated_budget_return_annual * 100, 0.5) / 100.0
        monitoring_duration_years = c2.number_input("Durata previsione monitoraggio (Anni)",
                                                    value=preview_params.monitoring_duration_years, min_value=1, max_value=20)
        failure_threshold = c2.slider("Soglia di Fallimento", 0.0, 0.5, preview_params.failure_threshold, 0.05)
        reliability_target_beta = c2.slider("Beta target affidabilità", 1.5, 6.0, float(preview_params.reliability_target_beta), 0.1)
        traffic_load_model = c2.selectbox("Traffic load model", ["NTC", "GL_heavy", "GL_medium", "GL_light"], index=["NTC", "GL_heavy", "GL_medium", "GL_light"].index(preview_params.traffic_load_model))

        deterioration_exponent = c3.slider("Esponente Curva di Degrado", 1.0, 5.0, preview_params.deterioration_exponent, 0.1)
        traffic_degradation_factor = c3.slider("Fattore Degrado Traffico", 0.0, 0.5, preview_params.traffic_degradation_factor,
                                               0.01)
        env_degradation_std = c3.slider("Dev. Std. Fattore Ambientale", 0.0, 0.5, preview_params.env_degradation_std, 0.01)
        deterioration_process_cov = c3.slider("CoV processo di degrado", 0.0, 0.5, float(preview_params.deterioration_process_cov), 0.01)
        traffic_lane_count = c3.selectbox("Numero corsie carico", [1, 2, 3], index=[1, 2, 3].index(preview_params.traffic_lane_count))
        traffic_first_lane_offset_m = c3.slider("Offset prima corsia (m)", 0.0, 2.0, float(preview_params.traffic_first_lane_offset_m), 0.5)

        enforce_guidelines = st.checkbox("Applica Linee Guida (Ispezioni Obbligatorie)",
                                         value=preview_params.enforce_guidelines)

        st.divider()
        st.subheader("🌍 Parametri Simulazione Sismica")
        eq_enabled = st.checkbox("Abilita simulazione sismica stocastica", value=preview_params.earthquake_enabled,
                                 help="Se attiva, ogni trimestre esiste una probabilità che si verifichi un terremoto.")
        eq_c1, eq_c2, eq_c3 = st.columns(3)
        earthquake_annual_prob = eq_c1.slider("Probabilità annua (%)", 1, 30,
                                              int(preview_params.earthquake_annual_prob * 100), 1) / 100.0
        earthquake_min_mag = eq_c2.slider("Magnitudo minima", 3.5, 6.0, float(preview_params.earthquake_min_mag), 0.1)
        earthquake_max_mag = eq_c3.slider("Magnitudo massima", 5.0, 8.0, float(preview_params.earthquake_max_mag), 0.1)
        seismic_bridge_class = eq_c1.selectbox("Classe sismica ponte", ["standard", "moderno", "TV", "SV", "TO", "IAB"], index=["standard", "moderno", "TV", "SV", "TO", "IAB"].index(preview_params.seismic_bridge_class))
        seismic_limit_state = eq_c2.selectbox("Limit state", ["UPD", "SD"], index=["UPD", "SD"].index(preview_params.seismic_limit_state))
        geo_c1, geo_c2, geo_c3 = st.columns(3)
        bridge_region_lat = geo_c1.number_input("Lat. regione (°N)", value=float(preview_params.bridge_region_lat), step=0.1, format="%.4f")
        bridge_region_lon = geo_c2.number_input("Lon. regione (°E)", value=float(preview_params.bridge_region_lon), step=0.1, format="%.4f")
        bridge_region_radius = geo_c3.number_input("Raggio regione (km)", value=float(preview_params.bridge_region_radius_km), step=5.0)

        # Danno locale improvviso: stessi campi di /api/config (metadati nel modello).
        sudden_values = _sudden_damage_form(preview_params)
        sudden_values.update(_sudden_damage_form(preview_params, regional=True))
        sudden_values.update(_sudden_damage_form(preview_params, progressive=True))
        st.divider()
        st.subheader("🔒 Informazione e integrita' del benchmark")
        bm_c1, bm_c2 = st.columns(2)
        _obs_modes = ["full", "inspected"]
        observation_mode = bm_c1.selectbox(
            "Modalita' di osservazione", _obs_modes, index=_obs_modes.index(preview_params.observation_mode),
            help="full = gli agenti vedono lo stato vero; inspected = solo cio' che ispezioni e prove hanno accertato.")
        _planner_modes = ["truth", "belief"]
        planner_information = bm_c2.selectbox(
            "Mondo dei pianificatori (GA/MCTS/memetico)", _planner_modes,
            index=_planner_modes.index(preview_params.planner_information),
            help="truth = copia dello stato vero (storico); belief = copia costruita sulla conoscenza del gestore.")
        oracle_observation = bm_c1.checkbox(
            "Oracolo (concede lo stato vero agli agenti)", value=bool(preview_params.oracle_observation))
        policy_sandbox = bm_c2.checkbox(
            "Sandbox delle policy", value=bool(preview_params.policy_sandbox),
            help="Le policy ricevono solo l'osservazione sanitizzata e copie del mondo del gestore, "
                 "mai il motore ne' il seme (vedi infralab/agents/sandbox.py).")
        strict_information = bm_c1.checkbox(
            "Informazione stretta", value=bool(preview_params.strict_information),
            help="Nessun segnale derivato dal vero nella conoscenza del gestore: incertezza solo da "
                 "evidenza, prova di carico pianificata sulla stima, vita utile nominale per i componenti "
                 "mai osservati, classificazione del regime sulla stima. Raccomandata con il sandbox.")
        strict_uncertainty_drop = bm_c2.number_input(
            "Calo osservato letto come danno", min_value=0.0, max_value=1.0,
            value=float(preview_params.strict_uncertainty_drop), step=0.01,
            help="Solo con informazione stretta: calo del centro dell'intervallo osservato fra due "
                 "osservazioni oltre il quale l'incertezza epistemica cresce (x1.10).")
        _rev_sources = ["truth", "assessed", "none"]
        revenue_health_source = bm_c1.selectbox(
            "Salute che riduce i ricavi", _rev_sources,
            index=_rev_sources.index(getattr(preview_params, "revenue_health_source", "truth")),
            help="truth = stato vero (storico: il budget porta un segnale dal vero); assessed = stima del "
                 "gestore; none = ricavi indipendenti dalla condizione. Per il benchmark: assessed o none.")
        detour_km_cap = bm_c2.number_input(
            "Tetto ai km di deviazione (0 = nessuno)", min_value=0.0,
            value=float(getattr(preview_params, "detour_km_cap", 0.0)), step=5.0,
            help="Il codice NBI 199 dell'item 019 significa 'nessuna deviazione possibile', non 199 km.")
        _part = ["all", "trucks"]
        detour_partial_diversion = bm_c1.selectbox(
            "Limitazione parziale: chi devia", _part,
            index=_part.index(getattr(preview_params, "detour_partial_diversion", "all")),
            help="all = devia la quota limitata di tutto il traffico (storico); trucks = solo i mezzi pesanti.")
        detour_works_diversion = bm_c2.checkbox(
            "I lavori deviano il traffico", value=bool(getattr(preview_params, "detour_works_diversion", True)))

        submitted = st.form_submit_button("Salva Parametri e Vai all'Editor Agenti")
        if submitted:
            if len(set(new_agent_names)) != len(new_agent_names):
                st.error("I nomi dei giocatori devono essere unici.")
            else:
                # Salva tutti i parametri nello stato della sessione
                params.agent_names = new_agent_names
                params.num_agents = num_agents
                params.calibration_profile = selected_profile
                params, calibrated_ops = apply_calibration_profile(params, asdict(MaintOpsConfig()), selected_profile)
                params.initial_budget = initial_budget
                params.years = years
                params.seed = seed
                params.unallocated_budget_return_annual = unallocated_budget_return_annual
                params.monitoring_duration_years = monitoring_duration_years
                params.failure_threshold = failure_threshold
                params.reliability_target_beta = reliability_target_beta
                params.traffic_load_model = traffic_load_model
                params.deterioration_exponent = deterioration_exponent
                params.traffic_degradation_factor = traffic_degradation_factor
                params.env_degradation_std = env_degradation_std
                params.deterioration_process_cov = deterioration_process_cov
                params.traffic_lane_count = traffic_lane_count
                params.traffic_first_lane_offset_m = traffic_first_lane_offset_m
                params.enforce_guidelines = enforce_guidelines
                params.earthquake_enabled = eq_enabled
                params.earthquake_annual_prob = earthquake_annual_prob
                params.earthquake_min_mag = earthquake_min_mag
                params.earthquake_max_mag = earthquake_max_mag
                params.seismic_bridge_class = seismic_bridge_class
                params.seismic_limit_state = seismic_limit_state
                params.bridge_region_lat = bridge_region_lat
                params.bridge_region_lon = bridge_region_lon
                params.bridge_region_radius_km = bridge_region_radius
                for _k, _v in sudden_values.items():
                    setattr(params, _k, _v)
                params.observation_mode = observation_mode
                params.planner_information = planner_information
                params.oracle_observation = bool(oracle_observation)
                params.policy_sandbox = bool(policy_sandbox)
                params.strict_information = bool(strict_information)
                params.strict_uncertainty_drop = float(strict_uncertainty_drop)
                params.revenue_health_source = revenue_health_source
                params.detour_km_cap = float(detour_km_cap)
                params.detour_partial_diversion = detour_partial_diversion
                params.detour_works_diversion = bool(detour_works_diversion)

                st.session_state.params = params
                st.session_state.maint_ops_override = calibrated_ops
                st.session_state.free_sim_page = "Editor"
                st.rerun()


def _sudden_damage_form(preview_params, regional: bool = False, progressive: bool = False) -> dict:
    """Parametri del danno locale improvviso (o, con regional=True, degli eventi estremi
    regionali; con progressive=True, degli eventi progressivi e della regola di fine
    partita al collasso) in un riquadro avanzato del form di setup.
    Restituisce {campo: valore}; un JSON non valido lascia il valore corrente."""
    if progressive:
        from infralab.core.hazards.progressive_events import UI_FIELDS
        title = "⏳ Eventi progressivi (scalzamento, versante), collassi e fine partita — avanzato"
    elif regional:
        from infralab.core.hazards.regional_events import UI_FIELDS
        title = "🌊 Eventi estremi regionali (piena di bacino, tempesta) — avanzato"
    else:
        from infralab.core.hazards.sudden_damage import UI_FIELDS
        title = "⚡ Danno locale improvviso (idraulico, urto, fatica) — avanzato"
    values = {}
    with st.expander(title, expanded=False):
        group = None
        for f in UI_FIELDS:
            if f["group"] != group:
                group = f["group"]
                st.markdown(f"**{group}**")
            key, cur = f["key"], getattr(preview_params, f["key"])
            wkey = f"sudden_{key}"
            if f["type"] == "bool":
                values[key] = st.checkbox(f["label"], value=bool(cur), key=wkey)
            elif f["type"] == "select":
                opts = list(f["options"])
                values[key] = st.selectbox(f["label"], opts, index=opts.index(cur) if cur in opts else 0, key=wkey)
            elif f["type"] == "text":
                values[key] = st.text_input(f["label"], value=str(cur), key=wkey)
            elif f["type"] == "json":
                raw = st.text_area(f["label"], value=json.dumps(cur), key=wkey)
                try:
                    values[key] = json.loads(raw) if raw.strip() else type(cur)()
                except ValueError:
                    st.warning(f"{f['label']}: JSON non valido, valore invariato.")
                    values[key] = cur
            else:
                values[key] = st.number_input(f["label"], value=float(cur), min_value=float(f.get("min", 0.0)),
                                              max_value=float(f["max"]) if "max" in f else None,
                                              step=float(f.get("step", 0.01)), format="%.6g", key=wkey)
                if isinstance(cur, int) and not isinstance(cur, bool):
                    values[key] = int(round(values[key]))    # campi interi (es. durate in trimestri)
    return values


def display_editor_page():
    st.header("🛠️ 2. Configurazione Agenti e Avvio")

    if 'params' not in st.session_state or not st.session_state.params.agent_names:
        st.error("Errore: Parametri di gioco non trovati. Configurali prima nella pagina 'Setup'.")
        if st.button("Torna a Setup"):
            st.session_state.free_sim_page = "Setup"
            st.rerun()
        return

    st.subheader("Configurazione Tipi di Agente")
    agent_names = st.session_state.params.agent_names

    if 'agent_config' not in st.session_state or set(st.session_state.agent_config.keys()) != set(agent_names):
        st.session_state.agent_config = {name: {"type": "Umano"} for name in agent_names}

    for agent_name in agent_names:
        with st.container(border=True):
            st.markdown(f"**Configurazione per {agent_name}**")
            cols = st.columns(4)
            current_type = st.session_state.agent_config.get(agent_name, {}).get('type', 'Umano')
            try:
                agent_type_index = AGENT_TYPES.index(current_type)
            except ValueError:
                agent_type_index = 0

            agent_type = cols[0].selectbox("Tipo Agente", AGENT_TYPES, key=f"type_{agent_name}", index=agent_type_index)

            # Inizializza/aggiorna la configurazione per questo agente
            if agent_name not in st.session_state.agent_config:
                st.session_state.agent_config[agent_name] = {}
            st.session_state.agent_config[agent_name]['type'] = agent_type

            if agent_type in ["LLM", "LLM Agentico"]:
                provider = cols[1].selectbox("Provider API", API_PROVIDERS, key=f"provider_{agent_name}")
                st.session_state.agent_config[agent_name]['provider'] = provider
                st.session_state.agent_config[agent_name]['profile'] = cols[2].selectbox("Profilo", LLM_PROFILES,
                                                                                         key=f"profile_{agent_name}")
                if provider == "HuggingFace":
                    hf_col1, hf_col2 = st.columns(2)
                    model_choice = hf_col1.selectbox("Modello Hugging Face", HF_DEFAULT_MODELS + ["Altro / Custom"],
                                                     key=f"hf_model_sel_{agent_name}")
                    if model_choice == "Altro / Custom":
                        custom_model = hf_col1.text_input("ID Modello (es. org/model-name)", key=f"hf_model_txt_{agent_name}")
                        st.session_state.agent_config[agent_name]['model'] = custom_model.strip() if custom_model.strip() else HF_DEFAULT_MODELS[0]
                    else:
                        st.session_state.agent_config[agent_name]['model'] = model_choice
                    hf_token_input = hf_col2.text_input("Token Hugging Face (opzionale se impostato in HF_TOKEN)",
                                                        type="password", key=f"hf_tok_{agent_name}")
                    if hf_token_input.strip():
                        st.session_state.agent_config[agent_name]['hf_token'] = hf_token_input.strip()
            if agent_type in ["GA Orizzonte Variabile", "Algoritmo Memetico"]:
                ga_params = st.session_state.agent_config[agent_name].setdefault('ga_params', {})
                ga_params['horizon'] = cols[1].number_input('Orizzonte', value=ga_params.get('horizon', 4),
                                                            key=f"ga_h_{agent_name}")
                ga_params['pop_size'] = cols[2].number_input('Popolazione', value=ga_params.get('pop_size', 50),
                                                             key=f"ga_p_{agent_name}")
                ga_params['generations'] = cols[3].number_input('Generazioni', value=ga_params.get('generations', 20),
                                                                key=f"ga_g_{agent_name}")
            if agent_type == "MCTS":
                mcts_params = st.session_state.agent_config[agent_name].setdefault('mcts_params', {})
                mcts_params['n_rollouts'] = cols[1].number_input('Rollout', min_value=5, value=mcts_params.get('n_rollouts', 50),
                                                                 key=f"mcts_r_{agent_name}")
                mcts_params['rollout_depth'] = cols[2].number_input('Profondità', min_value=1, value=mcts_params.get('rollout_depth', 4),
                                                                     key=f"mcts_d_{agent_name}")
                mcts_params['n_candidates'] = cols[3].number_input('Candidati', min_value=2, value=mcts_params.get('n_candidates', 12),
                                                                    key=f"mcts_c_{agent_name}")
            if agent_type == "RL Q-Learning":
                rl_params = st.session_state.agent_config[agent_name].setdefault('rl_params', {})
                rl_params['learning_rate'] = cols[1].number_input('α (LR)', value=float(rl_params.get('learning_rate', 0.04)),
                                                                   min_value=0.001, max_value=1.0, step=0.01, format="%.3f",
                                                                   key=f"rl_lr_{agent_name}")
                rl_params['epsilon_start'] = cols[2].number_input('ε iniziale', value=float(rl_params.get('epsilon_start', 0.30)),
                                                                   min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                                                                   key=f"rl_eps_{agent_name}")
                rl_params['discount'] = cols[3].number_input('γ (sconto)', value=float(rl_params.get('discount', 0.95)),
                                                             min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                                                             key=f"rl_disc_{agent_name}")

    st.divider()

    if st.button("🚀 Inizia Nuova Simulazione", type="primary", use_container_width=True):
        with st.spinner("Inizializzazione della partita..."):
            session_id = st.session_state.session_id
            persistence.clear_and_initialize_log_files(session_id)

            bridges_data = [
                # Set di ponti per il Manager 1
                {"id": 1, "name": "Viadotto Capodichino", "length": 1130, "width": 24, "adt": 65000,
                 "eta_iniziale_anni": 50, "vita_utile_anni": 80, "incasso_trimestrale": 250000,
                 "valore_asset": 25000000},
                {"id": 2, "name": "Viadotto Arena S. Antonio", "length": 420, "width": 22, "adt": 55000,
                 "eta_iniziale_anni": 48, "vita_utile_anni": 75, "incasso_trimestrale": 95000, "valore_asset": 9000000},
                {"id": 3, "name": "Viadotto Calata Capodichino", "length": 650, "width": 24, "adt": 60000,
                 "eta_iniziale_anni": 52, "vita_utile_anni": 70, "incasso_trimestrale": 120000,
                 "valore_asset": 14000000},
                {"id": 4, "name": "Viadotto Scudillo", "length": 310, "width": 22, "adt": 45000,
                 "eta_iniziale_anni": 45, "vita_utile_anni": 85, "incasso_trimestrale": 70000, "valore_asset": 7500000},
                {"id": 5, "name": "Viadotto Gerolomini", "length": 280, "width": 22, "adt": 40000,
                 "eta_iniziale_anni": 49, "vita_utile_anni": 75, "incasso_trimestrale": 60000, "valore_asset": 6000000},
                {"id": 6, "name": "Viadotto San Rocco", "length": 550, "width": 24, "adt": 58000,
                 "eta_iniziale_anni": 51, "vita_utile_anni": 70, "incasso_trimestrale": 110000,
                 "valore_asset": 12000000},
                {"id": 7, "name": "Viadotto Capodivilla", "length": 380, "width": 22, "adt": 50000,
                 "eta_iniziale_anni": 47, "vita_utile_anni": 80, "incasso_trimestrale": 85000, "valore_asset": 8000000},

                # Set di ponti duplicato per il Manager 2 (con ID univoci)
                {"id": 8, "name": "Viadotto Capodichino", "length": 1130, "width": 24, "adt": 65000,
                 "eta_iniziale_anni": 50, "vita_utile_anni": 80, "incasso_trimestrale": 250000,
                 "valore_asset": 25000000},
                {"id": 9, "name": "Viadotto Arena S. Antonio", "length": 420, "width": 22, "adt": 55000,
                 "eta_iniziale_anni": 48, "vita_utile_anni": 75, "incasso_trimestrale": 95000, "valore_asset": 9000000},
                {"id": 10, "name": "Viadotto Calata Capodichino", "length": 650, "width": 24, "adt": 60000,
                 "eta_iniziale_anni": 52, "vita_utile_anni": 70, "incasso_trimestrale": 120000,
                 "valore_asset": 14000000},
                {"id": 11, "name": "Viadotto Scudillo", "length": 310, "width": 22, "adt": 45000,
                 "eta_iniziale_anni": 45, "vita_utile_anni": 85, "incasso_trimestrale": 70000, "valore_asset": 7500000},
                {"id": 12, "name": "Viadotto Gerolomini", "length": 280, "width": 22, "adt": 40000,
                 "eta_iniziale_anni": 49, "vita_utile_anni": 75, "incasso_trimestrale": 60000, "valore_asset": 6000000},
                {"id": 13, "name": "Viadotto San Rocco", "length": 550, "width": 24, "adt": 58000,
                 "eta_iniziale_anni": 51, "vita_utile_anni": 70, "incasso_trimestrale": 110000,
                 "valore_asset": 12000000},
                {"id": 14, "name": "Viadotto Capodivilla", "length": 380, "width": 22, "adt": 50000,
                 "eta_iniziale_anni": 47, "vita_utile_anni": 80, "incasso_trimestrale": 85000, "valore_asset": 8000000},
            ]
            bridges_df = pd.DataFrame(bridges_data)
            maint_ops = deepcopy(st.session_state.get("maint_ops_override", asdict(MaintOpsConfig())))
            cda_config = CdaConfig()
            comp_weights = DEFAULT_COMPONENTS

            env = BridgeEnv(st.session_state.params, maint_ops, st.session_state.agent_config, cda_config, comp_weights)

            env.initialize_bridges(bridges_df.to_dict(orient='records'))
            persistence.save_game_state(env, session_id)
            st.session_state.free_sim_page = "Dashboard"
            st.rerun()


def display_dashboard_page():
    inject_custom_css()
    st.header("📊 Dashboard di Controllo")
    session_id = st.session_state.session_id
    env = persistence.load_game_state(session_id)
    if not env:
        st.error("Nessuno stato di gioco trovato. Inizia una nuova partita dalla pagina 'Setup'.")
        return

    history_dfs = persistence.load_history(session_id)
    if env.last_failure_info and env.last_failure_info.get("agent_name") not in env.active_agents:
        display_failure_summary(env, history_dfs)
        return

    has_human_player = any(conf['type'] == 'Umano' for conf in env.agent_config.values())
    is_finished = (env.quarter > env.params.years * 4 or not env.active_agents)

    col1, col2 = st.columns(2)
    with col1:
        if not has_human_player:
            if st.button("Avanza Trimestre", disabled=is_finished, type="primary"):
                with st.spinner("Calcolo del prossimo trimestre..."):
                    run_turn(env, session_id)
                st.rerun()
        else:
            st.info("Avanza il turno dalle interfacce dei giocatori umani.")

    with col2:
        if not has_human_player:
            if st.button("Esegui Tutta la Simulazione", disabled=is_finished):
                total_quarters = env.params.years * 4
                start_quarter = env.quarter
                progress_bar = st.progress(0.0, text=f"Simulazione in corso... T {env.quarter}")
                while env.quarter <= total_quarters and env.active_agents:
                    run_turn(env, session_id)
                    progress = (env.quarter - start_quarter) / (total_quarters - start_quarter + 1)
                    progress_bar.progress(progress, text=f"Simulazione in corso... T {env.quarter}")
                st.rerun()

    if is_finished:
        st.success("Simulazione Conclusa!")
        final_report = analysis.create_final_report_excel(env, history_dfs)
        sim_id_for_download = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("Scarica Report Finale (Excel)", final_report, f"report_finale_{sim_id_for_download}.xlsx")

    with st.expander("Interfaccia Giocatori Umani"):
        human_agents = [name for name, conf in env.agent_config.items() if conf.get('type') == 'Umano']
        if not human_agents:
            st.info("Nessun giocatore umano configurato per questa simulazione.")
        else:
            for agent_name in human_agents:
                if st.button(f"Apri Interfaccia per {agent_name}", key=f"player_btn_{agent_name}",
                             use_container_width=True):
                    st.session_state.agent_name = agent_name
                    st.session_state.free_sim_page = "Dashboard Giocatore"
                    st.rerun()

    st.subheader("Indicatori di Performance (KPI)")
    df_kpi = history_dfs.get('kpi', pd.DataFrame())
    df_budget = history_dfs.get('budget', pd.DataFrame())

    if df_kpi.empty:
        st.info("Nessun dato KPI disponibile. Avanza il primo trimestre per generare i grafici.")
    else:
        from infralab.ui.gis_bim import render_kpi_timeline, render_budget_timeline
        c1, c2 = st.columns(2)
        with c1:
            fig_kpi = render_kpi_timeline(df_kpi, env.params.agent_names)
            st.plotly_chart(fig_kpi, use_container_width=True)
        with c2:
            if not df_budget.empty:
                fig_bud = render_budget_timeline(df_budget, env.params.agent_names)
                st.plotly_chart(fig_bud, use_container_width=True)

    # Storico eventi sismici nel dashboard
    eq_events = getattr(env, 'earthquake_events', [])
    if eq_events:
        with st.expander(f"🌍 Eventi Sismici Registrati ({len(eq_events)})", expanded=False):
            for ev in eq_events:
                n_dam = len(ev.get('damage_reports', []))
                severity = "🔴" if ev.get('magnitude', 0) >= 6.0 else "🟠" if ev.get('magnitude', 0) >= 5.0 else "🟡"
                st.markdown(
                    f"{severity} **{ev.get('name', 'Terremoto')}** — "
                    f"M{ev.get('magnitude', '?')} | T{ev.get('quarter', '?')} | "
                    f"{n_dam} ponti danneggiati"
                )
                for dr in ev.get('damage_reports', []):
                    st.caption(
                        f"  → {dr['bridge_name']}: distanza {dr['distance_km']} km | "
                        f"PGA {dr['pga_g']:.3f}g | Danno: {dr.get('damage_level', 'N/D')}"
                    )

    st.subheader("Stato Ponti")
    records = [b.to_dict() for b in env.bridges.values()]
    df_bridges_display = pd.DataFrame.from_records(records)
    if not df_bridges_display.empty:
        df_bridges_display['min_observed_condition'] = df_bridges_display['min_observed_condition'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/D")
        st.dataframe(df_bridges_display[
                         ['id', 'name', 'manager', 'status', 'min_condition', 'min_observed_condition', 'cda',
                          'next_inspection_quarter']].style.format({"min_condition": "{:.3f}"}),
                     use_container_width=True)

    st.subheader("Cruscotto Finanziario")
    if df_budget.empty:
        st.info("Dati finanziari non ancora disponibili.")
    else:
        fig_budget, fig_net_worth = analysis.create_financial_charts(df_budget, df_kpi, env.params.agent_names)
        c1, c2 = st.columns(2)
        if fig_budget:
            with c1:
                st.pyplot(fig_budget)
        if fig_net_worth:
            with c2:
                st.pyplot(fig_net_worth)

    st.subheader("Riepilogo Economico Complessivo")
    summary_df = analysis.create_summary_df(env, history_dfs)
    st.dataframe(summary_df.style.format(formatter="{:,.2f}", subset=["Patrimonio Netto Complessivo", "Budget Liquido",
                                                                      "Valore Asset Ponti"], na_rep="N/D"),
                 use_container_width=True)

def display_failure_summary(env: BridgeEnv, history_dfs: dict):
    failure_info = env.last_failure_info
    if not failure_info: return

    with st.container(border=True):
        st.error(f"🚨 **FALLIMENTO AGENTE: {failure_info.get('agent_name')}** 🚨")
        st.subheader(failure_info.get("message"))
        if failure_info.get("reason") == "BRIDGE_COLLAPSE":
            st.subheader("Dettaglio Crollo Ponte")
            bridge_id = failure_info.get("bridge_id")
            df_actions = history_dfs.get('action', pd.DataFrame())
            if not df_actions.empty:
                df_bridge = df_actions[df_actions['bridge_id'] == bridge_id].set_index('quarter')
                comp_cols = [c for c in df_bridge.columns if c.endswith('_cond')]
                if not df_bridge.empty and comp_cols:
                    st.line_chart(df_bridge[comp_cols])
                else:
                    st.warning("Nessun dato storico disponibile per il ponte crollato.")
        if st.button("Continua Simulazione"):
            env.last_failure_info = None
            persistence.save_game_state(env, st.session_state.session_id)
            st.rerun()


def display_analytics_page():
    st.header("🔬 Analisi Dettagliata")
    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    session_id = st.session_state.session_id
    history_dfs = persistence.load_history(session_id)
    df_actions = history_dfs.get('action', pd.DataFrame())

    if df_actions.empty:
        st.info("Nessun dato di azione disponibile.")
        return

    st.subheader("Andamento e Ispezioni dei Componenti")

    show_inspections = st.checkbox("Mostra barre di errore delle ispezioni", value=True)

    bridge_ids = sorted([int(i) for i in df_actions['bridge_id'].unique() if i != -1])
    if not bridge_ids:
        st.info("Nessuna azione registrata sui ponti.")
        return

    env = persistence.load_game_state(session_id)
    if not env:
        st.warning("Stato del gioco non trovato.")
        return

    bridge_id = st.selectbox("Seleziona Ponte", options=bridge_ids)
    if bridge_id:
        df_bridge = df_actions[df_actions['bridge_id'] == bridge_id].set_index('quarter')
        comp_names = env.component_weights.keys()
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = {name: color for name, color in zip(comp_names, UNINA_PALETTE)}

        for comp_name in comp_names:
            cond_col, obs_min_col, obs_max_col = f'{comp_name}_cond', f'{comp_name}_obs_min', f'{comp_name}_obs_max'
            color = palette.get(comp_name, 'gray')
            if cond_col in df_bridge.columns:
                ax.plot(df_bridge.index, df_bridge[cond_col], label=f'Reale - {comp_name}', color=color)
                if show_inspections and obs_min_col in df_bridge.columns and obs_max_col in df_bridge.columns:
                    inspections = df_bridge[[obs_min_col, obs_max_col]].dropna()
                    if not inspections.empty:
                        y1, y2 = inspections[obs_min_col], inspections[obs_max_col]
                        ax.errorbar(inspections.index, (y1 + y2) / 2, yerr=(y2 - y1) / 2, fmt='o', color=color,
                                    alpha=0.7, capsize=5, label=f'Ispezione - {comp_name}')

        ax.set_title(f"Andamento Affidabilità Ponte ID: {bridge_id}")
        ax.set_xlabel("Trimestre");
        ax.set_ylabel("Affidabilità");
        ax.legend();
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)


def display_llm_analysis_page():
    st.header("🕵️ Analisi Agenti LLM")
    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    session_id = st.session_state.session_id
    llm_log_file = os.path.join(persistence.get_session_path(session_id), "llm_interactions.log")

    if not os.path.exists(llm_log_file):
        st.warning("File di log LLM non trovato per questa sessione.")
        return
    try:
        with open(llm_log_file, "r", encoding='utf-8') as f:
            log_lines = [json.loads(line) for line in f if line.strip()]
        if not log_lines:
            st.info("Nessun log valido trovato.")
            return
        df_logs = pd.DataFrame(log_lines).sort_values(by="timestamp", ascending=False)
        for _, row in df_logs.iterrows():
            with st.expander(f"**{row['agent_name']}** - Trimestre {row.get('quarter', 'N/A')} ({row['timestamp']})"):
                st.text_area("Spiegazione e Risposta", value=row.get('response', 'N/A'), height=200,
                             key=f"resp_{row['timestamp']}")
                st.text_area("Prompt Inviato", value=row.get('prompt', 'N/A'), height=300,
                             key=f"prompt_{row['timestamp']}")
    except Exception as e:
        st.error(f"Impossibile leggere il file di log: {e}")


def display_plotting_engine_page():
    st.header("📈 Motore di Plotting")
    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    session_id = st.session_state.session_id
    history_dfs = persistence.load_history(session_id)
    if not any(not df.empty for df in history_dfs.values()):
        st.warning("Nessun dato di simulazione da plottare.")
        return

    source_map = {"Log Azioni": "action", "Log Budget": "budget", "Log KPI": "kpi"}
    source_name = st.selectbox("1. Fonte Dati", list(source_map.keys()))
    df = history_dfs.get(source_map[source_name])

    if df is not None and not df.empty:
        time_col = 'quarter'
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        x_axis = c1.selectbox("2. Asse X", cols, cols.index(time_col) if time_col in cols else 0)
        y_axis = c2.selectbox("3. Asse Y", cols, 1 if len(cols) > 1 else 0)
        group_by = c3.selectbox("4. Raggruppa per", [None] + cols, (cols.index('agent') + 1) if 'agent' in cols else 0)

        if st.button("Genera Grafico", type="primary"):
            fig, ax = plt.subplots(figsize=(12, 7))

            # --- NUOVA LOGICA DI CONTROLLO TIPO DATI ---
            if pd.api.types.is_numeric_dtype(df[y_axis]):
                # Se Y è numerico, crea un grafico a linee
                plot_df = df.copy()
                plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors='coerce')
                plot_df.dropna(subset=[x_axis], inplace=True)

                ax.set_title(f"Andamento di '{y_axis}' vs '{x_axis}'")
                ax.set_xlabel(str(x_axis).replace('_', ' ').title())
                ax.set_ylabel(str(y_axis).replace('_', ' ').title())

                if group_by and group_by in plot_df.columns:
                    palette = {name: color for name, color in zip(plot_df[group_by].unique(), UNINA_PALETTE)}
                    for name, data in plot_df.groupby(group_by):
                        ax.plot(data[x_axis], data[y_axis], marker='o', linestyle='-', label=name,
                                color=palette.get(name))
                    ax.legend(title=str(group_by).replace('_', ' ').title(), loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax.plot(plot_df[x_axis], plot_df[y_axis], marker='o', color=UNINA_PALETTE[0])

            else:
                # Se Y è categorico (testo), crea un grafico a barre (conteggio)
                ax.set_title(f"Conteggio di '{y_axis}' raggruppato per '{group_by if group_by else x_axis}'")
                if group_by and group_by in df.columns:
                    counts = df.groupby([x_axis, y_axis]).size().reset_index(name='count')
                    unique_vals = counts[y_axis].unique()
                    for idx, val in enumerate(unique_vals):
                        sub = counts[counts[y_axis] == val]
                        ax.bar(sub[x_axis].astype(str), sub['count'], label=str(val),
                               alpha=0.8, color=UNINA_PALETTE[idx % len(UNINA_PALETTE)])
                    ax.legend(title=str(y_axis).replace('_', ' ').title(), loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    val_counts = df[y_axis].value_counts()
                    colors = [UNINA_PALETTE[i % len(UNINA_PALETTE)] for i in range(len(val_counts))]
                    ax.bar(val_counts.index.astype(str), val_counts.values, color=colors, alpha=0.85)

                ax.set_xlabel(str(y_axis).replace('_', ' ').title())
                ax.set_ylabel("Conteggio")
                plt.xticks(rotation=45, ha='right')

            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# Pagina GIS
# ---------------------------------------------------------------------------

def display_gis_page():
    inject_custom_css()
    st.header("🗺️ Mappa GIS — Rete Infrastrutturale")

    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    env = persistence.load_game_state(st.session_state.session_id)
    if not env:
        st.error("Nessuno stato di gioco trovato. Avvia prima una simulazione.")
        return

    from infralab.ui.gis_bim import render_gis_map, render_asset_value_bar
    from infralab.core.hazards.earthquake import EarthquakeEvent

    eq_raw = getattr(env, 'earthquake_events', [])
    eq_events_objs = [EarthquakeEvent.from_dict(e) for e in eq_raw]

    # --- Mappa principale ---
    fig_map = render_gis_map(env.bridges, earthquake_events=eq_events_objs if eq_events_objs else None)
    st.plotly_chart(fig_map, use_container_width=True)

    # --- KPI rete ---
    bridges = list(env.bridges.values())
    bridges_with_coords = [b for b in bridges if getattr(b, 'lat', None) is not None]
    avg_cond = float(np.mean([b.min_condition for b in bridges])) if bridges else 0.0
    critical_count = sum(1 for b in bridges if b.min_condition < 0.30)
    total_value = sum(b.current_asset_value for b in bridges)
    in_maintenance = sum(1 for b in bridges if b.active_maintenance)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌉 Ponti", len(bridges))
    c2.metric("❤️ Cond. Media", f"{avg_cond:.1%}")
    c3.metric("🚨 Critici (<30%)", critical_count)
    c4.metric("🔧 In Manutenzione", in_maintenance)
    c5.metric("💎 Valore Rete", f"€{total_value / 1e6:.1f}M")

    # --- Grafico valore asset ---
    st.plotly_chart(render_asset_value_bar(env.bridges), use_container_width=True)

    # --- Tabella dettaglio ---
    st.subheader("📋 Dettaglio Ponti — Rete Completa")
    bridge_rows = []
    for b in sorted(bridges, key=lambda x: x.min_condition):
        bridge_rows.append({
            "ID": b.id,
            "Nome": b.name,
            "Manager": b.manager,
            "Condizione": b.min_condition,
            "CdA": b.cda,
            "Stato": b.status,
            "Valore (€M)": b.current_asset_value / 1_000_000,
            "ADT": b.adt,
            "Lat (°N)": getattr(b, 'lat', None),
            "Lon (°E)": getattr(b, 'lon', None),
        })
    if bridge_rows:
        df_br = pd.DataFrame(bridge_rows)
        fmt = {"Condizione": "{:.1%}", "Valore (€M)": "{:.2f}",
               "Lat (°N)": "{:.4f}", "Lon (°E)": "{:.4f}"}
        st.dataframe(
            df_br.style.format(fmt, na_rep="N/D")
                  .background_gradient(subset=["Condizione"], cmap="RdYlGn"),
            use_container_width=True,
        )

    # --- Pannello terremoto manuale ---
    st.divider()
    with st.expander("💥 Simula Evento Sismico Manuale", expanded=False):
        st.caption("Applica un terremoto manuale alla rete in qualsiasi momento della simulazione.")
        eq_c1, eq_c2, eq_c3 = st.columns(3)
        mag = eq_c1.slider("Magnitudine", 3.5, 8.0, 5.5, 0.1)
        ep_lat = eq_c2.number_input("Lat. Epicentro (°N)", value=float(env.params.bridge_region_lat), step=0.01, format="%.4f")
        ep_lon = eq_c3.number_input("Lon. Epicentro (°E)", value=float(env.params.bridge_region_lon), step=0.01, format="%.4f")

        if st.button("💥 Applica Terremoto alla Rete", type="primary"):
            report = env.apply_manual_earthquake(mag, ep_lat, ep_lon)
            persistence.save_game_state(env, st.session_state.session_id)
            damage_reports = report.get('damage_reports', [])
            if damage_reports:
                st.error(f"M{mag:.1f} applicato — {len(damage_reports)} ponti danneggiati!")
                for dr in damage_reports:
                    with st.container(border=True):
                        st.markdown(
                            f"**🌉 {dr['bridge_name']}** | "
                            f"Distanza: {dr['distance_km']} km | PGA: {dr['pga_g']:.3f}g | "
                            f"**{dr.get('damage_level', 'N/D')}**"
                        )
                        for cn, cd in dr.get('components', {}).items():
                            st.progress(
                                cd['condition_after'],
                                text=f"{cn.capitalize()}: {cd['condition_before']:.1%} → "
                                     f"{cd['condition_after']:.1%} (−{cd['damage_factor']:.1%})"
                            )
            else:
                st.success(f"M{mag:.1f} applicato — nessun danno significativo registrato.")
            st.rerun()

    # --- Storico terremoti ---
    if eq_events_objs:
        st.subheader(f"🌍 Storico Eventi Sismici ({len(eq_events_objs)})")
        eq_rows = []
        for ev in eq_events_objs:
            eq_rows.append({
                "Trimestre": ev.quarter,
                "Nome": ev.name,
                "Magnitudine": ev.magnitude,
                "Lat Epicentro": ev.epicenter_lat,
                "Lon Epicentro": ev.epicenter_lon,
                "Ponti Danneggiati": len(ev.damage_reports),
            })
        st.dataframe(pd.DataFrame(eq_rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina BIM 3D
# ---------------------------------------------------------------------------

def display_bim_page():
    inject_custom_css()
    st.header("🏗️ Viewer BIM 3D — Analisi Strutturale")

    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    env = persistence.load_game_state(st.session_state.session_id)
    if not env:
        st.error("Nessuno stato di gioco trovato. Avvia prima una simulazione.")
        return

    from infralab.ui.gis_bim import render_bim_3d, render_asset_value_bar

    bridge_options = {b.id: f"[{b.id}] {b.name} — {b.manager}" for b in sorted(env.bridges.values(), key=lambda x: x.id)}
    if not bridge_options:
        st.info("Nessun ponte disponibile.")
        return

    selected_id = st.selectbox(
        "🌉 Seleziona Ponte da Visualizzare",
        options=list(bridge_options.keys()),
        format_func=lambda x: bridge_options[x],
    )

    if selected_id:
        bridge = env.bridges[selected_id]
        bridge_dict = bridge.to_dict()
        cond = bridge.min_condition

        # KPI ponte
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Condizione Min.", f"{cond:.1%}")
        c2.metric("Valore Asset", f"€{bridge.current_asset_value:,.0f}")
        c3.metric("CdA", bridge.cda)
        c4.metric("Stato", bridge.status)
        if bridge.reliability_index is not None:
            beta_post = bridge.posterior_reliability_index if bridge.posterior_reliability_index is not None else bridge.reliability_index
            pf_post = bridge.posterior_failure_probability if bridge.posterior_failure_probability is not None else (bridge.failure_probability or 0.0)
            beta_ann_post = bridge.posterior_annual_reliability_index if bridge.posterior_annual_reliability_index is not None else bridge.annual_reliability_index
            rel_details = bridge.posterior_reliability_details or bridge.reliability_details or {}
            target_beta_ref = rel_details.get("target_beta_reference")
            target_beta_ann = rel_details.get("target_beta_annual")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Beta Prior", f"{bridge.reliability_index:.2f}")
            r2.metric("Beta Post", f"{beta_post:.2f}")
            r3.metric("Beta Ann. Post", f"{beta_ann_post:.2f}")
            r4.metric("Pf Post", f"{pf_post:.2e}")
            if target_beta_ref is not None and target_beta_ann is not None:
                st.caption(
                    f"Target affidabilità: beta_ref={target_beta_ref:.2f} | "
                    f"beta_ann={target_beta_ann:.2f}"
                )
            life_cycle = rel_details.get("life_cycle_profile") or {}
            if life_cycle:
                st.caption(
                    f"Min beta annuo residuo ({life_cycle.get('projection_years')} anni): "
                    f"{life_cycle.get('min_remaining_structural_annual_beta', 0):.2f} "
                    f"al +{life_cycle.get('year_of_min_structural_beta', 0)} anno"
                )
        st.caption(
            f"Importanza rete={bridge.network_importance_factor:.2f} | "
            f"Centralità={bridge.network_centrality:.2f} | "
            f"Risk priority={bridge.risk_priority_index:.3e}"
        )

        # Visualizzazione 3D
        fig_bim = render_bim_3d(bridge_dict)
        st.plotly_chart(fig_bim, use_container_width=True)

        # Dettaglio componenti
        st.subheader("📋 Dettaglio Componenti Strutturali")
        comp_rows = []
        for comp_name, comp in bridge.components.items():
            obs_min = comp.observed_condition_min
            obs_max = comp.observed_condition_max
            comp_rows.append({
                "Componente": comp_name.capitalize(),
                "Condizione Reale": comp.current_condition,
                "Età Effettiva (anni)": round(comp.effective_age_years, 1),
                "Vita Utile (anni)": comp.vita_utile_anni,
                "Obs. Min": obs_min if obs_min is not None else float('nan'),
                "Obs. Max": obs_max if obs_max is not None else float('nan'),
            })

        if comp_rows:
            df_comp = pd.DataFrame(comp_rows)
            st.dataframe(
                df_comp.style.format({
                    "Condizione Reale": "{:.1%}",
                    "Obs. Min": "{:.1%}",
                    "Obs. Max": "{:.1%}",
                }, na_rep="N/D").background_gradient(subset=["Condizione Reale"], cmap="RdYlGn"),
                use_container_width=True,
            )

        # Barre progresso condizione visiva
        st.subheader("📊 Stato Visivo Componenti")
        for comp_name, comp in bridge.components.items():
            obs_min = comp.observed_condition_min
            obs_max = comp.observed_condition_max
            if obs_min is not None and obs_max is not None:
                mid = (obs_min + obs_max) / 2.0
                st.progress(mid, text=f"**{comp_name.capitalize()}** — Osservato: [{obs_min:.1%} – {obs_max:.1%}]")
            else:
                st.progress(comp.current_condition,
                            text=f"**{comp_name.capitalize()}** — Reale: {comp.current_condition:.1%} (non ispezionato)")

    # --- Confronto valore rete ---
    st.divider()
    st.subheader("🏛️ Confronto Valore Asset — Intera Rete")
    fig_bar = render_asset_value_bar(env.bridges)
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina Analisi Sismica con Curve di Fragilità
# ---------------------------------------------------------------------------

def display_seismic_analytics_page():
    inject_custom_css()
    st.header("🌍 Analisi Sismica & Curve di Fragilità")

    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    env = persistence.load_game_state(st.session_state.session_id)
    if not env:
        st.error("Nessuno stato di gioco trovato.")
        return

    from infralab.ui.gis_bim import render_seismic_timeline, render_fragility_curves
    from infralab.core.hazards.earthquake import (
        EarthquakeEvent, calculate_pga, haversine_distance,
        get_fragility_profile, FRAGILITY_PARAMS, pga_to_mcs,
    )

    eq_raw = getattr(env, 'earthquake_events', [])

    # ── KPI sismici ──────────────────────────────────────────────────────────
    n_events = len(eq_raw)
    if eq_raw:
        mags = [ev.get('magnitude', 0) for ev in eq_raw]
        total_damaged = sum(len(ev.get('damage_reports', [])) for ev in eq_raw)
        max_mag = max(mags)
        avg_mag = sum(mags) / len(mags)
    else:
        total_damaged, max_mag, avg_mag = 0, 0.0, 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 N. Terremoti", n_events)
    c2.metric("📊 Magnitudo Media", f"M{avg_mag:.1f}" if n_events else "—")
    c3.metric("🔴 Magnitudo Max", f"M{max_mag:.1f}" if n_events else "—")
    c4.metric("🏗️ Ponti Danneggiati (tot.)", total_damaged)

    st.divider()

    # ── Timeline sismicità ────────────────────────────────────────────────────
    st.subheader("📅 Timeline Sismicità")
    fig_timeline = render_seismic_timeline(eq_raw)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Curve di fragilità ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📉 Curve di Fragilità Strutturale (Lognormale HAZUS)")
    col_ctrl1, col_ctrl2 = st.columns(2)
    bridge_type = col_ctrl1.selectbox(
        "Tipo di Ponte",
        ["standard", "vecchio", "moderno"],
        format_func=lambda x: {"standard": "Standard", "vecchio": "Pre-normativa (Vecchio)", "moderno": "Post-2003 (Moderno)"}[x],
    )
    highlight_pga = col_ctrl2.slider(
        "PGA di riferimento (g)", min_value=0.0, max_value=1.2, value=0.15, step=0.01,
        help="Valore di PGA da evidenziare sulle curve"
    )

    fig_frag = render_fragility_curves(bridge_type, highlight_pga=highlight_pga)
    st.plotly_chart(fig_frag, use_container_width=True)

    # ── Tabella probabilità al PGA selezionato ────────────────────────────────
    if highlight_pga > 0:
        st.subheader(f"📋 Probabilità di Danno a PGA = {highlight_pga:.3f}g")
        profs = get_fragility_profile(highlight_pga, bridge_type)
        cols = st.columns(4)
        damage_icons = {"Lieve": "🟡", "Moderato": "🟠", "Esteso": "🔴", "Completo": "🟣"}
        for col, (state, prob) in zip(cols, profs.items()):
            col.metric(f"{damage_icons.get(state, '')} {state}", f"{prob:.1%}")

    # ── Analisi danno per singolo evento ─────────────────────────────────────
    if eq_raw:
        st.divider()
        st.subheader("🔬 Dettaglio per Evento Sismico")
        eq_objs = [EarthquakeEvent.from_dict(ev) for ev in eq_raw]
        ev_options = {i: f"T{ev.quarter} — {ev.name} (M{ev.magnitude})" for i, ev in enumerate(eq_objs)}
        sel_idx = st.selectbox("Seleziona Evento", list(ev_options.keys()), format_func=lambda x: ev_options[x])

        sel_ev = eq_objs[sel_idx]
        dr = sel_ev.damage_reports

        col_a, col_b = st.columns(2)
        col_a.markdown(f"""
        **Nome:** {sel_ev.name}
        **Magnitudine:** M{sel_ev.magnitude}
        **Epicentro:** {sel_ev.epicenter_lat:.4f}°N, {sel_ev.epicenter_lon:.4f}°E
        **Trimestre:** {sel_ev.quarter}
        **Ponti colpiti:** {len(dr)}
        **MCS:** {pga_to_mcs(calculate_pga(sel_ev.magnitude, 10.0))} (a 10 km)
        """)

        if dr:
            df_dr = pd.DataFrame([
                {
                    "Ponte": r['bridge_name'],
                    "Distanza (km)": r['distance_km'],
                    "PGA (g)": r['pga_g'],
                    "Livello Danno": r['damage_level'],
                    "N. Comp. Danneggiati": len(r.get('components', {})),
                }
                for r in dr
            ])
            with col_b:
                st.dataframe(
                    df_dr.style.background_gradient(subset=["PGA (g)"], cmap="Reds"),
                    use_container_width=True,
                )

            # Barras componenti danneggiati
            st.subheader("🏗️ Stato Componenti dopo il Terremoto")
            for rep in dr:
                if not rep.get('components'):
                    continue
                with st.expander(f"🌉 {rep['bridge_name']} — PGA {rep['pga_g']:.3f}g", expanded=False):
                    for comp_name, cd in rep['components'].items():
                        delta = cd['condition_after'] - cd['condition_before']
                        st.progress(
                            cd['condition_after'],
                            text=f"**{comp_name.capitalize()}**: {cd['condition_before']:.1%} → "
                                 f"{cd['condition_after']:.1%} (Δ {delta:+.1%}) — {cd['damage_level']}"
                        )
        else:
            col_b.info("Nessun danno registrato per questo evento.")


# ---------------------------------------------------------------------------
# Pagina Report Finale / Dashboard Conclusivo
# ---------------------------------------------------------------------------

def display_final_report_page():
    inject_custom_css()
    st.header("📋 Report Finale & Dashboard Conclusivo")

    if 'session_id' not in st.session_state:
        st.warning("Nessuna sessione attiva.")
        return

    env = persistence.load_game_state(st.session_state.session_id)
    if not env:
        st.error("Nessuno stato di gioco trovato.")
        return

    history_dfs = persistence.load_history(st.session_state.session_id)
    df_kpi = history_dfs.get('kpi', pd.DataFrame())
    df_budget = history_dfs.get('budget', pd.DataFrame())

    if df_kpi.empty and df_budget.empty:
        st.info("Avanza almeno un trimestre per generare il report.")
        return

    agent_names = env.params.agent_names

    # ── Riepilogo finale agenti ───────────────────────────────────────────────
    st.subheader("🏆 Classifica Finale Agenti")
    summary_df = analysis.create_summary_df(env, history_dfs)
    if not summary_df.empty:
        summary_sorted = summary_df.sort_values("Patrimonio Netto Complessivo", ascending=False).reset_index(drop=True)
        summary_sorted.index += 1  # Ranking da 1
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        summary_sorted.insert(0, "Rank", [medals.get(i, f"#{i}") for i in summary_sorted.index])
        st.dataframe(
            summary_sorted.style.format(
                {"Patrimonio Netto Complessivo": "€{:,.0f}", "Budget Liquido": "€{:,.0f}",
                 "Valore Asset Ponti": "€{:,.0f}", "Indice di Salute": "{:.1%}"},
                na_rep="N/D"
            ).background_gradient(subset=["Indice di Salute"], cmap="RdYlGn"),
            use_container_width=True,
        )

    st.subheader("🧪 Output Sperimentale per Paper")
    paper_df = analysis.create_paper_experiment_df(env)
    if not paper_df.empty:
        st.dataframe(
            paper_df.style.format(
                {
                    "Min Condition": "{:.1%}",
                    "Posterior Beta": "{:.2f}",
                    "Posterior Pf": "{:.2e}",
                    "Traffic Eta": "{:.2f}",
                    "Traffic Beta": "{:.2f}",
                    "Traffic Annual Failure Rate": "{:.2e}",
                    "Seismic Pf": "{:.2e}",
                    "Seismic Annual Exceedance Proxy": "{:.2e}",
                },
                na_rep="N/D",
            ),
            use_container_width=True,
        )
    management_diag_df = analysis.create_management_diagnostics_df(history_dfs, agent_names)
    if not management_diag_df.empty:
        st.subheader("🧭 Diagnostica Gestione")
        st.dataframe(
            management_diag_df.style.format(
                {
                    "Inspection Coverage Finale": "{:.1%}",
                    "Blocked Action Rate": "{:.1%}",
                },
                na_rep="N/D",
            ),
            use_container_width=True,
        )

    # ── KPI finali per colonne ────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Indicatori Finali")
    if not summary_df.empty:
        kpi_cols = st.columns(len(agent_names))
        for col, agent in zip(kpi_cols, agent_names):
            row = summary_df[summary_df['Agente'] == agent]
            if row.empty:
                continue
            r = row.iloc[0]
            col.metric(f"**{agent}**", f"€{r['Patrimonio Netto Complessivo']:,.0f}",
                       delta=f"HI: {r['Indice di Salute']:.1%}")
            col.caption(f"Stato: {r['Stato']}")

    # ── Analisi statistica ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Analisi Statistica")
    stat_results = analysis.create_statistical_analysis(history_dfs, agent_names)

    tab1, tab2, tab3, tab4 = st.tabs(["Health Index", "Budget", "Azioni & Costi", "Gestione"])

    with tab1:
        hi_stats = stat_results.get('health_stats')
        if hi_stats is not None and not hi_stats.empty:
            st.dataframe(
                hi_stats.style.format(
                    {"Media": "{:.1%}", "Std": "{:.3f}", "Min": "{:.1%}", "Max": "{:.1%}",
                     "Mediana": "{:.1%}", "CoV (%)": "{:.1f}"}
                ).background_gradient(subset=["Media"], cmap="RdYlGn"),
                use_container_width=True,
            )
            # Boxplot sintetico con matplotlib
            if not df_kpi.empty and 'health_index' in df_kpi.columns:
                fig_box, ax_box = plt.subplots(figsize=(8, 3))
                data_box = [df_kpi[df_kpi['agent'] == a]['health_index'].dropna().tolist() for a in agent_names]
                bplot = ax_box.boxplot(data_box, labels=agent_names, patch_artist=True, notch=True)
                for patch, color in zip(bplot['boxes'], UNINA_PALETTE):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax_box.set_ylabel("Health Index")
                ax_box.set_title("Distribuzione Health Index per Agente")
                ax_box.set_ylim(0, 1.05)
                fig_box.tight_layout()
                st.pyplot(fig_box)
        else:
            st.info("Dati Health Index non disponibili.")

    with tab2:
        bud_stats = stat_results.get('budget_stats')
        if bud_stats is not None and not bud_stats.empty:
            st.dataframe(
                bud_stats.style.format(
                    {"Budget Medio": "€{:,.0f}", "Budget Min": "€{:,.0f}",
                     "Budget Max": "€{:,.0f}", "Budget Finale": "€{:,.0f}"}
                ).background_gradient(subset=["Budget Medio"], cmap="Blues"),
                use_container_width=True,
            )
        corr = stat_results.get('correlations')
        if corr is not None and not corr.empty:
            st.markdown("**Correlazione Health Index ↔ Budget:**")
            st.dataframe(corr.style.background_gradient(subset=["Correlazione HI-Budget"], cmap="RdBu"), use_container_width=True)

    with tab3:
        cost_stats = stat_results.get('cost_stats')
        if cost_stats is not None and not cost_stats.empty:
            st.dataframe(
                cost_stats.style.format(
                    {"Costo Totale": "€{:,.0f}", "Costo Medio Azione": "€{:,.0f}"}
                ),
                use_container_width=True,
            )
        action_counts = stat_results.get('action_counts')
        if action_counts is not None and not action_counts.empty:
            st.markdown("**Distribuzione azioni per tipo:**")
            fig_ac, ax_ac = plt.subplots(figsize=(9, 3))
            if 'agent' in action_counts.columns:
                for i, agent in enumerate(agent_names):
                    sub = action_counts[action_counts['agent'] == agent]
                    if not sub.empty:
                        ax_ac.bar(sub['action'], sub['Conteggio'], label=agent,
                                  alpha=0.8, color=UNINA_PALETTE[i % len(UNINA_PALETTE)])
                ax_ac.legend(title="Agente")
            else:
                ax_ac.bar(action_counts['action'], action_counts['Conteggio'],
                          color=UNINA_PALETTE[0], alpha=0.85)
            ax_ac.set_xlabel("Azione")
            ax_ac.set_ylabel("Conteggio")
            ax_ac.set_title("Frequenza Azioni di Manutenzione")
            plt.xticks(rotation=35, ha='right')
            fig_ac.tight_layout()
            st.pyplot(fig_ac)

    with tab4:
        management_stats = stat_results.get('management_diagnostics')
        if management_stats is not None and not management_stats.empty:
            st.dataframe(
                management_stats.style.format(
                    {
                        "Inspection Coverage Finale": "{:.1%}",
                        "Blocked Action Rate": "{:.1%}",
                    },
                    na_rep="N/D",
                ),
                use_container_width=True,
            )
        else:
            st.info("Diagnostica di gestione non disponibile.")

    # ── Download Excel ────────────────────────────────────────────────────────
    st.divider()
    excel_bytes = analysis.create_final_report_excel(env, history_dfs)
    st.download_button(
        "📥 Scarica Report Completo (Excel)",
        data=excel_bytes,
        file_name=f"report_finale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Valutazione AI Lab ────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔬 Valutazione: Completezza come AI Laboratory", expanded=False):
        st.markdown("""
        ### Cosa è implementato ✅
        - Simulazione di degrado non lineare (Power Law) con incertezza ambientale
        - 7 tipi di agenti: Umano, Euristico, GA, Memetico, MCTS, RL Q-Learning, LLM
        - Sistema sismico con attenuazione PGA e danni per componente
        - Curve di fragilità lognormali (HAZUS MH MR5)
        - GIS interattivo + BIM 3D senza API key
        - Analisi statistica descrittiva per agente
        - Export Excel multi-foglio

        ### Sviluppi futuri 🚀
        - **ML avanzato**: Deep RL (PPO/SAC), Graph Neural Networks sulla rete
        - **Modelli predittivi**: ARIMA/Prophet per forecast degrado, anomaly detection
        - **Multi-obiettivo**: Ottimizzazione Pareto (costo vs sicurezza vs resilienza)
        - **Dati reali**: Import cataloghi sismici INGV, traffico da ANAS, meteo API
        - **Analisi di sensitività**: Indici di Sobol, Morris One-At-A-Time
        - **Governance multi-livello**: Coordinamento regionale, fondi ministeriali
        - **CO₂ tracking**: Impatto ambientale delle manutenzioni
        - **Digital Twin**: Collegamento con sensori IoT reali
        """)


# ---------------------------------------------------------------------------
# Pagina Analisi Curve di Degrado
# ---------------------------------------------------------------------------

def display_degradation_analysis_page():
    inject_custom_css()
    st.header("📉 Analisi Curve di Degrado Strutturale")

    st.markdown("""
    Confronto analitico dei modelli matematici di degrado implementabili.
    Permette di comprendere come diversi modelli influenzino le previsioni di vita utile.
    """)

    # ── Configurazione ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    vita_utile = col1.slider("Vita Utile (anni)", 30, 150, 80, 5)
    exponent = col2.slider("Esponente (per modelli Power/Weibull)", 1.0, 6.0, 3.0, 0.5)
    eta_corrente = col1.slider("Età corrente da evidenziare (anni)", 0, vita_utile, vita_utile // 2)

    df_deg = analysis.create_degradation_model_df(vita_utile, exponent)

    # ── Grafico con matplotlib ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    model_colors = {
        'Power Law (attuale)': '#0077c8',
        'Lineare':              '#f9a825',
        'Weibull':              '#d62728',
        'Bilineare':            '#2ca02c',
        'Esponenziale':         '#9467bd',
    }
    for model, color in model_colors.items():
        if model in df_deg.columns:
            ax.plot(df_deg['Età (anni)'], df_deg[model], label=model, color=color, linewidth=2.2)

    # Linea età corrente
    ax.axvline(x=eta_corrente, color='gray', linestyle='--', linewidth=1.2, label=f'Età attuale ({eta_corrente}a)')
    # Zona critica
    ax.axhspan(0.0, 0.1, alpha=0.12, color='red', label='Zona Collasso (<10%)')
    ax.axhspan(0.1, 0.3, alpha=0.07, color='orange', label='Zona Critica (10–30%)')

    ax.set_xlabel("Età del Componente (anni)", fontsize=11)
    ax.set_ylabel("Affidabilità / Condizione", fontsize=11)
    ax.set_title(f"Confronto Modelli di Degrado — Vita Utile {vita_utile}a, Esponente {exponent}", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, vita_utile)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)

    # ── Tabella valori all'età corrente ──────────────────────────────────────
    st.subheader(f"📊 Condizione Prevista a {eta_corrente} anni")
    row_at_age = df_deg[df_deg['Età (anni)'] >= eta_corrente].iloc[0] if eta_corrente < vita_utile else df_deg.iloc[-1]
    model_cols = [c for c in df_deg.columns if c != 'Età (anni)']
    kpi_cols = st.columns(len(model_cols))
    for col, model in zip(kpi_cols, model_cols):
        val = row_at_age[model]
        col.metric(model, f"{val:.1%}", delta=f"{val - 1.0:.1%}")

    # ── Vita residua per soglia ────────────────────────────────────────────────
    st.subheader("⏳ Vita Residua al 30% di Affidabilità (soglia di intervento)")
    residua_data = []
    for model in model_cols:
        below = df_deg[df_deg[model] <= 0.30]
        if not below.empty:
            age_threshold = below.iloc[0]['Età (anni)']
            residua_data.append({"Modello": model, "Età al 30% (anni)": round(age_threshold, 1),
                                  "Vita Residua (anni)": max(0.0, round(age_threshold - eta_corrente, 1))})
        else:
            residua_data.append({"Modello": model, "Età al 30% (anni)": ">Vita Utile",
                                  "Vita Residua (anni)": "Oltre soglia"})
    st.dataframe(pd.DataFrame(residua_data), use_container_width=True)

    # ── Spiegazione modelli ────────────────────────────────────────────────────
    with st.expander("📖 Descrizione dei Modelli di Degrado"):
        st.markdown(f"""
        | Modello | Formula | Caratteristica |
        |---|---|---|
        | **Power Law** *(attuale)* | `1 - (t/T)^{exponent}` | Degrado lento poi accelerato |
        | **Lineare** | `1 - t/T` | Perdita uniforme nel tempo |
        | **Weibull** | `exp(-(t/T)^{exponent})` | Tasso di guasto variabile |
        | **Bilineare** | Due segmenti lineari | Fase latente + fase critica |
        | **Esponenziale** | `exp(-λ·t/T)` | Decadimento esponenziale puro |

        Dove `t` = età effettiva, `T` = vita utile nominale.
        L'esponente controlla la convessità della curva per i modelli Power Law e Weibull.
        """)

