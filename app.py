# app.py
import os
import json
from datetime import datetime
from dataclasses import asdict

import streamlit as st
import google.generativeai as genai
import openai
import matplotlib.style as mplstyle
import pandas as pd

import infralab.ui.pages_legacy as ui_pages
import infralab.persistence as persistence
from infralab.ui import auth
from infralab.settings.config import SimParameters, MaintOpsConfig, CdaConfig, DEFAULT_COMPONENTS
from infralab.core.environment import BridgeEnv
from infralab.calibration_profiles import apply_calibration_profile


def setup_apis():
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=google_api_key)
    except (KeyError, FileNotFoundError):
        pass
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass


def run_free_simulation_app():
    ui_pages.render_brand_sidebar("Sandbox · Simulazione libera")

    if st.sidebar.button("Torna alla selezione modalità"):
        keys_to_clear = ['research_role', 'app_mode', 'participants', 'aggregated_results', 'session_to_analyze',
                         'free_sim_page', 'agent_name', 'session_id', 'params', 'agent_config']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    PAGES = {
        "Setup": ui_pages.display_setup_page,
        "Editor": ui_pages.display_editor_page,
        "Dashboard": ui_pages.display_dashboard_page,
        "Dashboard Giocatore": ui_pages.display_player_page,
        "Mappa GIS": ui_pages.display_gis_page,
        "Viewer BIM 3D": ui_pages.display_bim_page,
        "Analisi Dettagliata": ui_pages.display_analytics_page,
        "Analisi Sismica": ui_pages.display_seismic_analytics_page,
        "Report Finale": ui_pages.display_final_report_page,
        "Curve di Degrado": ui_pages.display_degradation_analysis_page,
        "Plotting Engine": ui_pages.display_plotting_engine_page,
    }

    if 'session_id' not in st.session_state:
        st.session_state.session_id = persistence.DEFAULT_SESSION_ID

    if 'free_sim_page' not in st.session_state:
        st.session_state.free_sim_page = "Setup"

    for name in PAGES.keys():
        if name == "Dashboard Giocatore":
            continue

        game_state_exists = os.path.exists(
            os.path.join(persistence.get_session_path(st.session_state.session_id), persistence.GAME_STATE_FILENAME))

        # L'editor e le pagine analitiche pure sono sempre accessibili
        # Le pagine che richiedono una sessione attiva sono disabilitate senza gioco
        ALWAYS_ON = {"Setup", "Editor", "Dashboard Giocatore", "Curve di Degrado"}
        is_post_game_page = name not in ALWAYS_ON

        if st.sidebar.button(name, use_container_width=True, disabled=is_post_game_page and not game_state_exists):
            st.session_state.free_sim_page = name
            if 'agent_name' in st.session_state:
                del st.session_state.agent_name
            st.rerun()

    page_function = PAGES.get(st.session_state.free_sim_page)
    if page_function:
        page_function()


# ... (il resto del file app.py rimane invariato)
def run_research_app():
    # --- NUOVA LOGICA DI AVVIO PER I GIOCATORI ---
    query_params = st.query_params.to_dict()

    # Accesso giocatore via token firmato dall'admin (?token=...).
    token_from_url = query_params.get("token")
    if token_from_url and 'research_role' not in st.session_state:
        token_payload = auth.resolve_player_token(token_from_url)
        if token_payload:
            st.session_state.research_role = 'player_init'
            st.session_state.scenario_id = token_payload["scenario_id"]
            st.session_state.player_name = token_payload["player_name"]
        else:
            st.error("Link di gioco non valido o revocato. Contatta lo sperimentatore.")

    # Percorso legacy (?scenario_id=&player_name=): mantenuto per i vecchi link.
    player_name_from_url = query_params.get("player_name")
    scenario_id_from_url = query_params.get("scenario_id")
    if player_name_from_url and scenario_id_from_url and 'research_role' not in st.session_state:
        st.session_state.research_role = 'player_init'
        st.session_state.scenario_id = scenario_id_from_url
        st.session_state.player_name = player_name_from_url

    if 'research_role' not in st.session_state:
        ui_pages.render_brand_sidebar("Protocol · Modalità ricerca")
        st.markdown(
            """
            <div class="il-eyebrow">Protocol · Esperimento controllato</div>
            <div class="il-hero-title" style="font-size:clamp(30px,4vw,44px);">Modalità Ricerca</div>
            <div class="il-lede" style="margin-top:14px;">Accedi come amministratore per gestire scenari e
            partecipanti, oppure avvia una sessione di test con il tuo codice partita.</div>
            <div class="il-rule"></div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("← Torna alla selezione modalità"):
            del st.session_state.app_mode
            st.rerun()

        # Pagina di accesso per Admin o per Giocatore (se non usa il link diretto)
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.header("👨‍💻 Accesso Amministratore")
                password = st.text_input("Inserisci la password", type="password", key="admin_password")
                if st.button("Accedi come Admin"):
                    if auth.verify_admin_password(password):
                        st.session_state.research_role = "admin"
                        st.rerun()
                    else:
                        st.error("Password errata.")
        with col2:
            with st.container(border=True):
                st.header("🎮 Avvia Partita per Test")
                scenario_code = st.text_input("Inserisci il tuo Codice Partita", key="scenario_code_input")
                player_name = st.text_input("Inserisci il tuo Nome", key="player_name_input")
                if st.button("Avvia Partita"):
                    if scenario_code and player_name:
                        st.session_state.research_role = 'player_init'
                        st.session_state.scenario_id = scenario_code
                        st.session_state.player_name = player_name
                        st.rerun()
                    else:
                        st.warning("Per favore, inserisci sia il codice partita che il tuo nome.")
        return

    # --- Gestione dei ruoli dopo l'accesso ---
    if st.session_state.research_role == "admin":
        ui_pages.display_experiment_dashboard()

    elif st.session_state.research_role == 'player_init':
        try:
            scenario_id = st.session_state.scenario_id
            player_name = st.session_state.player_name
            scenarios = persistence.load_scenarios()
            scenario = next((s for s in scenarios if s['scenario_id'] == scenario_id), None)
            
            if not scenario:
                st.error("Codice Partita non valido o scenario non trovato.")
                st.link_button("Torna alla pagina di avvio", "/")
                del st.session_state.research_role # Resetta per riprovare
                return

            # Cerca se esiste già una sessione per questo giocatore e scenario
            existing_session = persistence.find_session(scenario_id, player_name)
            if existing_session:
                st.session_state.session_id = existing_session
                st.session_state.agent_name = player_name
                st.session_state.research_role = 'player_playing'
                st.rerun()
            else:
                # Se non esiste, crea una nuova sessione
                timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                session_id = f"{scenario_id}_{player_name.replace(' ', '_')}_{timestamp}"
                
                persistence.initialize_session_storage(session_id)
                
                base_params = asdict(SimParameters())
                base_params.update(scenario.get('params', {}))
                if scenario.get('max_quarters'):
                    base_params['max_quarters'] = scenario['max_quarters']
                params = SimParameters(**base_params)
                params.agent_names = [player_name]
                params.num_agents = 1
                agent_config = {player_name: {"type": "Umano"}}

                bridges_list = scenario.get('bridges', [
                    {"id": 1, "name": "Ponte di Prova", "length": 100, "width": 20, "adt": 5000,
                     "eta_iniziale_anni": 20, "vita_utile_anni": 80,
                     "incasso_trimestrale": 25000, "valore_asset": 5000000}
                ])
                maint_ops = asdict(MaintOpsConfig())
                params, maint_ops = apply_calibration_profile(params, maint_ops, params.calibration_profile)
                from infralab.assets import get_asset_type
                asset_type = scenario.get('asset_type', 'bridge')
                spec = get_asset_type(asset_type)
                maint_ops = {**maint_ops, **spec.maintenance_ops}
                allowed = scenario.get('allowed_actions')
                if allowed:
                    maint_ops = {k: v for k, v in maint_ops.items() if k in allowed}
                cda_config = CdaConfig()
                comp_weights = dict(spec.default_component_weights)

                env = BridgeEnv(params, maint_ops, agent_config, cda_config, comp_weights,
                                asset_type=asset_type)
                env.initialize_bridges(bridges_list)

                persistence.clear_and_initialize_log_files(session_id, component_names=spec.component_names)
                persistence.save_game_state(env, session_id)
                persistence.save_experiment_metadata(
                    session_id,
                    experiment_phase='tutorial',
                    tutorial_step=0,
                    scenario_id=scenario_id,
                    show_questionnaire=scenario.get('use_in_analysis', False),
                )

                st.session_state.session_id = session_id
                st.session_state.agent_name = player_name
                st.session_state.research_role = 'player_playing'
                st.rerun()

        except Exception as e:
            st.error(f"Errore durante l'avvio della partita: {e}")
            st.link_button("Riprova", "/")
            if 'research_role' in st.session_state: del st.session_state.research_role

    elif st.session_state.research_role == 'player_playing':
        ui_pages.display_player_page()

def render_landing():
    """Landing page InfraLab — dossier/strumento tecnico per l'affidabilità strutturale."""
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between;
                    border-bottom:1px solid var(--il-line2); padding-bottom:16px; margin-bottom:44px;">
          <div style="display:flex; align-items:center; gap:12px;">
            <svg width="32" height="32" viewBox="0 0 34 34" fill="none">
              <rect x="1" y="1" width="32" height="32" rx="3" stroke="#1f6fb2" stroke-width="1.3"/>
              <path d="M6 24 L14 10 L20 20 L24 14 L28 24" stroke="#0b4f86" stroke-width="1.4"
                    fill="none" stroke-linejoin="round"/>
              <line x1="6" y1="24" x2="28" y2="24" stroke="#4aa3df" stroke-width="1.2"/>
              <circle cx="14" cy="10" r="1.7" fill="#4aa3df"/>
            </svg>
            <div style="font-family:'IBM Plex Serif',serif; font-weight:600; font-size:21px;
                        letter-spacing:.02em; color:#0b4f86;">Infra<span style="color:#1f6fb2;">Lab</span></div>
          </div>
          <div class="il-topmeta">REL · <b>&beta; / p<sub>f</sub></b> &nbsp;·&nbsp; ENGINE <b>v0.1.0</b> &nbsp;·&nbsp; <b>UNINA</b> · STRUCT-AI</div>
        </div>

        <div class="il-eyebrow">Laboratorio virtuale · Gestione di asset infrastrutturali</div>
        <div class="il-hero-title">Un banco di prova per<br>decisioni di sicurezza <em>affidabili</em>.</div>
        <div class="il-lede">InfraLab simula il ciclo di vita di reti infrastrutturali &mdash; ponti,
        muri di sostegno e oltre &mdash; accoppiando <b>affidabilit&agrave; strutturale FOSM/bayesiana</b>,
        degrado stocastico e agenti decisionali (euristici, ottimizzazione, LLM) in un unico banco di prova
        riproducibile per <b>nuovi paper e nuovi test</b>.</div>
        <div class="il-rule"></div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="il-cardmeta">01 / SANDBOX</div>
                <div style="font-family:'IBM Plex Serif',serif; font-weight:500; font-size:24px;
                            color:#0b4f86; margin:14px 0 10px;">Simulazione Libera</div>
                <div style="color:#4a5c6e; font-size:14px; line-height:1.55; font-weight:400; min-height:64px;">
                Configura scenari, calibra agenti e gioca partite singole. Mappa GIS, viewer BIM&nbsp;3D,
                curve di degrado e analisi sismica in tempo reale.</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Entra nel banco  →", use_container_width=True, key="mode_free"):
                st.session_state.app_mode = 'free_sim'
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="il-cardmeta">02 / PROTOCOL</div>
                <div style="font-family:'IBM Plex Serif',serif; font-weight:500; font-size:24px;
                            color:#0b4f86; margin:14px 0 10px;">Modalità Ricerca</div>
                <div style="color:#4a5c6e; font-size:14px; line-height:1.55; font-weight:400; min-height:64px;">
                Esperimenti controllati con soggetti umani e campagne computazionali. Token per-partecipante,
                scenari versionati, export dati per la pubblicazione.</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Avvia esperimento  →", use_container_width=True, type="primary", key="mode_research"):
                st.session_state.app_mode = 'research'
                st.rerun()

    st.markdown(
        """
        <div style="margin-top:46px; padding-top:18px; border-top:1px solid var(--il-line);
                    display:flex; justify-content:space-between;" class="il-foot">
          <span>InfraLab · Reliability-based AI for structural safety</span>
          <span>Build 0.1.0 — bridge · retaining wall</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        layout="wide",
        page_title="InfraLab · Reliability Lab",
        page_icon="◇",
        initial_sidebar_state="expanded",
    )
    mplstyle.use('seaborn-v0_8-whitegrid')
    ui_pages.inject_custom_css()
    setup_apis()

    if 'app_mode' not in st.session_state:
        # Gestisce l'accesso diretto via URL anche prima della selezione della modalità
        query_params = st.query_params.to_dict()
        if "token" in query_params or ("scenario_id" in query_params and "player_name" in query_params):
            st.session_state.app_mode = 'research'
            st.rerun()

        render_landing()

    elif st.session_state.app_mode == 'free_sim':
        run_free_simulation_app()
    
    elif st.session_state.app_mode == 'research':
        run_research_app()

if __name__ == "__main__":
    main()
