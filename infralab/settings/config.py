# config.py
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd

ADMIN_PASSWORD = "admin"

AGENT_TYPES = ["Umano", "Euristico", "Basic", "GA Orizzonte Variabile", "Algoritmo Memetico", "MCTS", "RL Q-Learning", "LLM", "LLM Agentico"]
API_PROVIDERS = ["Google", "OpenAI", "HuggingFace", "Locale"]
HF_DEFAULT_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]
LLM_PROFILES = ["Molto Prudente", "Prudente", "Bilanciato", "Dinamico", "Aggressivo"]
DEFAULT_COMPONENTS = {"impalcato": 0.5, "pile": 0.3, "spalle": 0.15, "appoggi": 0.05}
# Deve restare allineato alle chiavi di TRAFFIC_FRAGILITY_BASE in environment.py
# (environment importa config: la lista vive qui per evitare import circolari).
TRAFFIC_LOAD_MODELS = ("NTC", "GL_heavy", "GL_medium", "GL_light")
UNINA_PALETTE = ["#0077c8", "#f9a825", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]


@dataclass
class SimParameters:
    calibration_profile: str = "baseline_demo"
    num_agents: int = 2
    agent_names: List[str] = field(default_factory=list)
    initial_budget: int = 1_000_000
    years: int = 50
    unallocated_budget_return_annual: float = 0.02
    debt_interest_rate_annual: float = 0.05
    enforce_guidelines: bool = True
    monitoring_duration_years: int = 5
    deterioration_exponent: float = 3.0
    deterioration_process_cov: float = 0.12
    # Modello di degrado: "weibull" (eta' efficace, default storico) oppure
    # "gamma" (processo gamma alla van Noortwijk: stessa traiettoria media,
    # varianza dei percorsi controllata da gamma_process_theta).
    degradation_model: str = "weibull"
    gamma_process_theta: float = 0.05
    # Cadenza decisionale del giocatore umano: quanti trimestri avanza il motore
    # per ogni decisione. 1 = trimestrale (default); 4 = annuale (esperimenti
    # con orizzonti lunghi entro un budget di tempo).
    human_cadence_quarters: int = 1
    # --- Workflow Linee Guida Ponti 2020 (opt-in per non alterare gli snapshot) ---
    # CdA multi-rischio: combina condizione + rischio sismico/esposizione con
    # regola max() (un difetto strutturale grave porta comunque CdA Alta).
    cda_multi_hazard: bool = False
    # Escalation: dopo N trimestri in CdA Alta senza una valutazione L4 recente,
    # l'asset è "overdue" (segnalato all'agente, KPI di conformità, penalità sul
    # rischio). 0 = disabilitata.
    l4_escalation_quarters: int = 0
    l4_escalation_risk_penalty: float = 2.0
    # Ispezione L1 scaduta: dopo quanti trimestri oltre la scadenza è "overdue".
    inspection_overdue_grace_quarters: int = 2
    # Ispezioni: bias sistematico (>0 = ispettori ottimisti) e POD opzionale
    # (il danno puo' non essere rilevato; prob. di detection cresce col danno).
    inspection_bias: float = 0.0
    inspection_pod_enabled: bool = False
    inspection_pod_exponent: float = 8.0
    # Non-stazionarieta': crescita annua del traffico e trend climatico
    # (moltiplicatore annuo del tasso di invecchiamento).
    traffic_growth_annual: float = 0.0
    climate_degradation_trend_annual: float = 0.0
    # Normativa che detta i target di affidabilità (vedi infralab.codes).
    # 'custom' = usa i target manuali sottostanti (comportamento storico).
    code_framework: str = "custom"
    # Regime normativo (vedi infralab.regimes): 'none' = nessuna legge imposta
    # (comportamento storico). Un regime puo' fissare obblighi, enforcement,
    # tier di target (override di code_framework) e parametri.
    regulatory_regime: str = "none"
    # Costo atteso del rischio (metrica, nessun effetto sulla dinamica): il
    # collasso di un asset costa la sostituzione C0 (valore iniziale) piu' le
    # conseguenze dirette H = failure_consequence_ratio * C0. Default 3.0 = caso
    # di riferimento H/C0 = 3 in Steenbergen, Rozsas, Vrouwenvelder, HERON 63(3)
    # 2018, Tab. 5; classe di conseguenza "moderata" JCSS 2001: 1 <= H/C0 < 4.
    failure_consequence_ratio: float = 3.0
    # Cosa vede il gestore (agente e trigger del regime):
    #  "full"      = stato vero del simulatore (storico; per debug e UI didattica);
    #  "inspected" = stato STIMATO: condizione osservata dove ispezionato, altrimenti
    #                credenza nominale (eta' nominale + parametri medi del profilo);
    #                beta/Pf/rischio ricalcolati sullo stato stimato con parametri
    #                nominali. Lo stato vero resta solo nel punteggio.
    observation_mode: str = "full"
    # Modello di collasso:
    #  "condition"   = storico: collasso se la condizione strutturale minima scende
    #                  sotto failure_threshold; l'agente esce dal gioco;
    #  "reliability" = collasso estratto ogni trimestre con prob. 1-(1-Pf_annua)^(1/4)
    #                  (Pf vera, posterior se c'e' prova di carico); l'evento e' registrato
    #                  come perdita realizzata C0*(1+H/C0), l'opera e' ricostruita (nuova)
    #                  e la partita continua. Coerente con la perdita attesa.
    failure_model: str = "condition"
    # Affidabilita' dello stato VERO (collassi, punteggio a posteriori):
    #  "knowledge" = storico: l'incertezza epistemica del gestore entra nella resistenza
    #                e negli scarti (ispezionare alza la resistenza calcolata);
    #  "physical"  = la Pf vera usa un'incertezza di riferimento FISSA
    #                (physical_epistemic_reference): ispezionare non cambia la fisica.
    #                La stima del gestore (observation_mode="inspected") usa comunque la
    #                sua incertezza reale, che le ispezioni riducono.
    true_reliability_mode: str = "knowledge"
    # Incertezza epistemica usata nella STIMA del gestore (observation_mode="inspected"):
    #  "manager"   = storico: quella del gestore, che le ispezioni riducono (la stima sale
    #                a ogni ispezione; con true_reliability_mode="physical" chi ispeziona
    #                sovrastima il beta vero, chi non ispeziona lo sottostima);
    #  "reference" = la stessa incertezza di riferimento del vero
    #                (physical_epistemic_reference): la conoscenza del gestore entra solo
    #                dalla condizione osservata, e stima e vero sono sulla stessa scala.
    assessment_epistemic_mode: str = "manager"
    # Valutatore prudente (opt-in, default invariato): nella STIMA del gestore la resistenza
    # e' divisa per un fattore di confidenza legato alla conoscenza del componente, come i
    # fattori di confidenza delle norme sugli esistenti (NTC 2018, Circolare 2019 C8.5.4).
    #  "none"            = storico, nessun fattore;
    #  "knowledge_level" = FC del primo gradino di assessment_confidence_levels la cui soglia
    #                      e' <= incertezza epistemica del gestore sul componente
    #                      (gradini [soglia, FC] in ordine di soglia decrescente).
    # Non tocca lo stato vero ne' la Pf fisica.
    assessment_confidence_mode: str = "none"
    assessment_confidence_levels: tuple = ((0.20, 1.35), (0.12, 1.20), (0.0, 1.00))
    # Scomposizione dello scarto stima - vero (opt-in, default invariati; diagnostica):
    #  assessment_parameter_mode "mean"    = storico: la stima usa i parametri MEDI del profilo
    #                                        (vita, scala, forma, scarti di resistenza e domanda);
    #                            "sampled" = la stima usa i parametri campionati del ponte vero.
    #  assessment_state_mode "last_observation" = storico: componente osservato fermo al centro
    #                                        dell'ultima osservazione;
    #                        "projected"  = l'eta' efficace invertita dall'osservazione avanza del
    #                                        tempo trascorso dall'ultima ispezione (curva della stima).
    #  assessment_seed_mode "distinct" = storico: campioni Monte Carlo della stima distinti da quelli
    #                                    del vero; "shared" = stessi campioni del vero.
    assessment_parameter_mode: str = "mean"
    assessment_state_mode: str = "last_observation"
    assessment_seed_mode: str = "distinct"
    # Moltiplicatore della deviazione standard delle osservazioni d'ispezione (1.0 = storico).
    inspection_sigma_scale: float = 1.0
    # Casualita' della simulazione:
    #  "shared" = storico: un solo generatore per degrado, ispezioni, esiti degli
    #             interventi, prove di carico e sismi (un'azione in piu' sposta tutto
    #             il futuro casuale);
    #  "keyed"  = un flusso per fonte, legato a (seme, asset, componente, trimestre):
    #             due gestori sullo stesso seme vivono lo stesso futuro fisico.
    rng_streams: str = "shared"
    # Quali azioni sospendono il degrado dell'asset nel trimestre:
    #  "any_action"    = storico: qualunque azione (anche un'ispezione) congela il degrado;
    #  "interventions" = solo interventi e manutenzioni in corso (tipi intervention,
    #                    preventive): ispezioni, monitoraggio, misure operative no.
    degradation_pause_on: str = "any_action"
    # Un gestore pubblico non fallisce: con False il budget negativo diventa debito
    # (interessi a debt_interest_rate_annual) e la partita continua.
    bankruptcy_enabled: bool = True
    # Chi paga e con quale capienza le azioni imposte dall'autorita':
    #  "owner_unlimited" = storico: eseguite comunque, a debito se serve;
    #  "owner_budget"    = eseguite solo se il gestore ha capienza; altrimenti restano
    #                      pendenti (arretrato) e scattano le eventuali sanzioni;
    #  "state"           = eseguite e pagate dallo Stato (mandato finanziato);
    #  "matched"         = contributo in quota parte: lo Stato paga
    #                      forced_matching_share, il gestore il resto e solo se ha
    #                      capienza (23 U.S.C. 120: quota federale 80-90%).
    forced_funding: str = "owner_unlimited"
    # Quota a carico dello Stato quando forced_funding = "matched".
    forced_matching_share: float = 0.8
    # Solo per riprodurre campagne precedenti (comportamento storico, difettoso): True =
    # un'azione imposta che chi paga non puo' finanziare annulla comunque l'azione
    # programmata dal gestore e blocca gli obblighi successivi dell'asset nel trimestre.
    # False (default) = l'obbligo non finanziabile resta pendente, l'azione del gestore
    # resta, e si passa all'obbligo dovuto successivo.
    regime_override_unfunded: bool = False
    # Trasferimento pubblico annuo, come frazione del valore di ricostruzione del parco
    # gestito (0 = nessun trasferimento). Accreditato per quarti ogni trimestre.
    annual_budget_grant_fraction: float = 0.0
    # Ogni quanti trimestri si rivaluta l'affidabilita' dell'intero parco (1 = storico).
    # Gli asset su cui si e' agito nel trimestre sono sempre rivalutati. Con 4 la
    # rivalutazione e' annuale: il degrado e' lento, il costo di calcolo si divide per 4.
    reliability_refresh_every_quarters: int = 1
    # Taratura assoluta dell'affidabilita' (default = modello storico):
    #  capacity_condition_sensitivity k: fattore di capacita' = 1 - k (1 - condizione);
    #      k = 1 -> capacita' proporzionale alla condizione (storico);
    #  reliability_resistance_scale: moltiplica la resistenza media di tutti gli stati limite.
    capacity_condition_sensitivity: float = 1.0
    # Traffico nella domanda e nell'invecchiamento:
    #  "network_relative" = storico: ADT del ponte / ADT di riferimento della rete;
    #  "self_designed"    = ogni ponte e' progettato per il proprio traffico: conta solo
    #                       la crescita del traffico dopo la costruzione e le restrizioni.
    traffic_demand_mode: str = "network_relative"
    reliability_resistance_scale: float = 1.0
    physical_epistemic_reference: float = 0.22
    reliability_consequence_class: str = "CC2"
    reliability_reference_period_years: int = 50
    reliability_target_beta: float = 3.8
    reliability_target_beta_annual: float | None = None
    reliability_sls_target_beta: float = 1.5
    reliability_brittle_failure_beta_increment: float = 0.3
    reliability_mc_samples: int = 256
    reliability_model_uncertainty_cov: float = 0.10
    reliability_geometry_cov: float = 0.05
    reliability_projection_years: int = 10
    failure_threshold: float = 0.1
    traffic_degradation_factor: float = 0.05
    traffic_reference_adt: int = 10000
    env_degradation_mean: float = 1.0
    env_degradation_std: float = 0.1
    demand_cov: float = 0.12
    resistance_cov: float = 0.15
    proof_load_base_alpha: float = 0.95
    proof_load_damage_limit: float = 0.05
    proof_load_cost_factor: float = 250.0
    proof_load_step_count: int = 3
    post_maintenance_observation_sigma: float = 0.12
    post_maintenance_knowledge_gain: float = 0.10
    traffic_load_model: str = "GL_heavy"
    traffic_lane_count: int = 3
    traffic_first_lane_offset_m: float = 0.0
    traffic_remaining_area_active: bool = True
    traffic_congestion_probability: float = 0.01
    traffic_monthly_overload_probability: float = 0.09
    seismic_bridge_class: str = "standard"
    seismic_limit_state: str = "SD"
    management_objective_mode: str = "safety_driven"
    # Seed della simulazione: qualunque intero (0 compreso) => run deterministico;
    # None => entropia di sistema (run non riproducibile).
    seed: int | None = 0
    # Simulazione sismica (master toggle: se False la sismica non incide sull'affidabilità)
    earthquake_enabled: bool = False
    earthquake_annual_prob: float = 0.05
    earthquake_min_mag: float = 4.5
    earthquake_max_mag: float = 6.5
    # Mappa di pericolosità sismica: la PGA di sito dipende dalla distanza dal
    # centro-sorgente (zona ad alta pericolosità). Default: Appennino campano.
    seismic_hazard_center_lat: float = 40.80
    seismic_hazard_center_lon: float = 15.30
    seismic_hazard_peak_pga: float = 0.35
    seismic_hazard_far_pga: float = 0.05
    seismic_hazard_decay_km: float = 45.0
    # Regione geografica di default per i ponti (Campania / UNINA)
    bridge_region_lat: float = 40.85
    bridge_region_lon: float = 14.25
    bridge_region_radius_km: float = 60.0
    network_importance_adt_weight: float = 0.65
    # Come si misura la rilevanza di un ponte nella rete:
    #  "proximity"  = storico: inverso della distanza media dai tre ponti piu' vicini,
    #                 mescolato al traffico normalizzato;
    #  "nbi_detour" = percentile dell'esposizione da deviazione (ADT x km di deviazione,
    #                 item NBI 019), se lo scenario porta il dato.
    network_relevance_mode: str = "proximity"
    # Costo per gli utenti della deviazione, contabilizzato ma NON incluso nel punteggio:
    # EUR per veicolo-km deviato, auto e mezzi pesanti. Derivati dal Handbook on the
    # external costs of transport (CE Delft per DG MOVE, 2019): valore del tempo
    # diviso per 60 km/h piu' costo operativo del veicolo.
    detour_cost_car_eur_km: float = 0.41
    detour_cost_truck_eur_km: float = 1.15
    detour_accounting: bool = False          # True: calcola il costo utenti per trimestre
    # Modello del costo utenti (default = storico). detour_km_cap: tetto ai km di
    # deviazione, 0 = nessuno (il codice NBI 199 = "nessuna deviazione possibile").
    # detour_partial_diversion: "all" (storico) o "trucks" (una limitazione parziale
    # devia solo i mezzi pesanti). detour_works_diversion: i lavori deviano la quota
    # revenue_impact del traffico (True, storico) o solo la chiusura vera (False).
    detour_km_cap: float = 0.0
    detour_partial_diversion: str = "all"
    detour_works_diversion: bool = True

    # --- PERICOLO IDRAULICO (piena ed erosione al fondo) -------------------------
    # E' la causa che domina il registro reale dei crolli (piene 33% ed erosione 16%
    # su 503 crolli americani 1989-2000, Wardhana e Hadipriono 2003), mentre il sisma
    # ne spiega il 2%. Colpisce solo i ponti su corso d'acqua (item NBI 071), con un
    # tasso piu' alto per quelli con fondazione a rischio erosione (item 113 <= 3).
    # Tassi per ponte per TRIMESTRE, calibrati dall'alto sul tasso osservato di crollo
    # (2.13e-4 per ponte-anno, di cui 49% idraulico), non dal basso su HEC-18.
    hydraulic_hazard_enabled: bool = False
    hydraulic_collapse_rate_scour_q: float = 2.7e-4    # fondazione a rischio erosione
    hydraulic_collapse_rate_water_q: float = 2.7e-5    # altri attraversamenti d'acqua
    hydraulic_damage_rate_multiplier: float = 10.0     # eventi di danno non distruttivo
    hydraulic_damage_condition_drop: float = 0.20      # perdita di condizione alle fondazioni
    # --- LIMITAZIONE DI PORTATA INDIPENDENTE DALLA SANZIONE ----------------------
    # Nella realta' il declassamento di portata nasce dalla verifica di capacita', non
    # dall'inadempienza: il 2.52% dei ponti del Delaware e il 10.1% di quelli americani
    # risulta limitato. Tasso di ingresso ricavato da quella quota con durata media di
    # cinque anni (legge di Little).
    posting_entry_rate_q: float = 0.0                  # 1.26e-3 = quota osservata Delaware
    posting_mean_duration_quarters: int = 20
    posting_restriction_factor: float = 0.6            # come limitazione_carico
    # Due opzioni (default = comportamento storico). La probabilita' di ingresso e'
    # tasso x severita', severita' = max(pavimento, 2 x (1 - condizione vera minima)):
    # `posting_severity_floor` e' quel pavimento (storico 0.25; 0 = un ponte in ottimo
    # stato non viene quasi mai declassato). Con `posting_lift_on_intervention` un
    # intervento strutturale revoca il declassamento in corso e, finche' durano i
    # lavori, non ne nasce uno nuovo (un ponte appena rifatto non resta limitato).
    posting_severity_floor: float = 0.25
    posting_lift_on_intervention: bool = False
    # --- CHIUSURA TEMPORANEA INDIPENDENTE DALLO STATO (opt-in; 0 = spento) ---------
    # Chiusure che non nascono dalla condizione dell'opera (lavori sulla viabilita',
    # eventi, verifiche): nell'inventario reale anche i ponti in buono stato risultano
    # chiusi in piccola quota. Stesso schema del declassamento: tasso di ingresso per
    # ponte e trimestre, durata geometrica; quota attesa = tasso x durata media.
    # Flusso casuale dedicato. Valori PROVISIONAL (docs/CHANGELOG_MODEL.md).
    temporary_closure_rate_q: float = 0.0
    temporary_closure_mean_duration_quarters: int = 4
    # Parte legata allo stato (opt-in; 0 = spento): la probabilita' di ingresso diventa
    # temporary_closure_rate_q + temporary_closure_condition_rate_q x 2 x (1 - condizione
    # vera minima), cioe' piu' alta sui ponti peggiori (stessa forma del declassamento).
    temporary_closure_condition_rate_q: float = 0.0

    # --- DANNO LOCALE IMPROVVISO (infralab/core/hazards/sudden_damage.py) ----------
    # Tutto opt-in: con i default nessun evento e comportamento storico identico.
    # Ogni evento e' sorteggiato per ponte e trimestre da un flusso a chiave dedicato
    # (numeri casuali comuni; non sposta gli altri flussi), riduce la condizione VERA
    # dei componenti bersaglio ed e' invisibile a gestore e legge finche' un'ispezione
    # (o il monitoraggio) non lo rileva, salvo la frazione autodenunciante.
    # Distribuzione dell'intensita' (perdita di condizione): "fixed" (= media),
    # "uniform" (media +- sqrt(3) CoV media), "lognormal" (media e CoV); troncata a [0,1].
    # Bersagli: nomi di componente separati da virgole; hits "one" = un bersaglio a caso,
    # "all" = tutti. Le fonti dei valori (o PROVISIONAL) sono in docs/CHANGELOG_MODEL.md.
    # Idraulico: True = canale storico invariato (RNG del motore con rng_streams="shared",
    # perdita fissa su pile e spalle, evento scritto subito nello storico azioni).
    # False = l'idraulico diventa un tipo del canale improvviso (stessa esposizione e
    # stessi tassi, flusso dedicato, danno nascosto fino all'ispezione).
    hydraulic_legacy_mode: bool = True
    hydraulic_damage_targets: str = "pile,spalle"
    hydraulic_damage_hits: str = "all"
    hydraulic_damage_drop_dist: str = "fixed"
    hydraulic_damage_drop_cov: float = 0.5             # PROVISIONAL
    hydraulic_damage_announce_prob: float = 0.0        # PROVISIONAL: erosione sott'acqua, non visibile
    # Urto di veicolo. Tasso di danno per ponte e trimestre all'ADT di riferimento:
    # quota urti 12.9% dei crolli (campione nazionale USA) x 2.13e-4 crolli/ponte-anno
    # (Cook, Barr e Halling 2015) / 4 = 6.9e-6 crolli/trimestre, per il rapporto danno/
    # crollo 10 del canale idraulico (PROVISIONAL) = 6.9e-5.
    # Esposizione: x (ADT/traffic_reference_adt)^impact_adt_exponent
    #              x (1 + impact_truck_weight x quota mezzi pesanti, item NBI 109)
    #              x impact_road_under_multiplier se sotto passa una strada (item 042B).
    impact_hazard_enabled: bool = False
    impact_damage_rate_q: float = 6.9e-5
    impact_adt_exponent: float = 1.0                   # PROVISIONAL (frequenza ~ traffico)
    impact_truck_weight: float = 1.0                   # PROVISIONAL
    impact_road_under_multiplier: float = 1.0          # PROVISIONAL (neutro)
    impact_damage_targets: str = "impalcato,pile"
    impact_damage_hits: str = "one"
    impact_damage_drop_dist: str = "lognormal"
    impact_damage_drop_mean: float = 0.15              # PROVISIONAL
    impact_damage_drop_cov: float = 0.5                # PROVISIONAL
    impact_damage_announce_prob: float = 0.5           # PROVISIONAL (urto visibile)
    # Urto di natante: solo ponti su via navigabile (attributo navigable_waterway,
    # item NBI 038 = 1). Tasso PROVISIONAL (nessuna fonte nei documenti di validazione).
    vessel_impact_enabled: bool = False
    vessel_impact_rate_q: float = 1.0e-4               # PROVISIONAL
    vessel_damage_targets: str = "pile"
    vessel_damage_hits: str = "one"
    vessel_damage_drop_dist: str = "lognormal"
    vessel_damage_drop_mean: float = 0.25              # PROVISIONAL
    vessel_damage_drop_cov: float = 0.5                # PROVISIONAL
    vessel_damage_announce_prob: float = 0.8           # PROVISIONAL
    # Danno locale tipo fatica/frattura. Tasso PROVISIONAL: limite superiore = intera
    # quota "degrado" dei crolli (8.8% x 2.13e-4 / 4) x rapporto danno/crollo 10.
    fatigue_hazard_enabled: bool = False
    fatigue_damage_rate_q: float = 4.7e-5              # PROVISIONAL
    fatigue_adt_exponent: float = 0.0                  # PROVISIONAL (nessuna dipendenza)
    fatigue_truck_weight: float = 2.0                  # PROVISIONAL
    fatigue_damage_targets: str = "impalcato,appoggi"
    fatigue_damage_hits: str = "one"
    fatigue_damage_drop_dist: str = "lognormal"
    fatigue_damage_drop_mean: float = 0.10             # PROVISIONAL
    fatigue_damage_drop_cov: float = 0.5               # PROVISIONAL
    fatigue_damage_announce_prob: float = 0.0          # PROVISIONAL (cricca nascosta)
    # Tipi definiti dall'utente: lista di dizionari con chiavi
    #  name, rate_q (per ponte e trimestre), targets, hits, drop_dist, drop_mean, drop_cov,
    #  announce_prob, requires (attributi del ponte che devono essere veri, es.
    #  "crosses_water" o "road_under"), multipliers ({attributo: fattore}),
    #  adt_exponent, truck_weight,
    #  condition_max (l'evento avviene solo se la condizione VERA minima dei bersagli e'
    #  <= valore), condition_exponent (tasso x (1 - condizione minima)^esponente):
    #  cedimento locale di un elemento gia' degradato. hits "worst" = il bersaglio in
    #  condizione peggiore.
    sudden_custom_events: List[Dict[str, Any]] = field(default_factory=list)
    # Probabilita' per trimestre che un monitoraggio attivo riveli un danno nascosto
    # (0 = il monitoraggio non vede i danni improvvisi, come prima).
    sudden_damage_monitoring_detect_prob: float = 0.0
    # Rumore dell'osservazione del danno reso noto (autodenuncia): come un'ispezione L1.
    sudden_damage_reveal_sigma: float = 0.10

    # --- EVENTI ESTREMI REGIONALI (infralab/core/hazards/regional_events.py) -------
    # Opt-in: con regional_events_enabled=False nessun effetto (snapshot invariato).
    # Eventi rari che colpiscono insieme molti ponti (piena di bacino, tempesta): numero
    # per trimestre ~ Poisson(regional_event_rate_y / 4) da un flusso a chiave dedicato;
    # centro su un ponte a caso ("random_bridge") o fisso ("fixed"); area = ponti esposti
    # entro il raggio; intensita' locale decrescente con la distanza. L'evento (quando,
    # dove, quali ponti esposti nell'area) e' pubblico; il danno per ponte e' nascosto
    # fino all'ispezione, salvo la quota autodenunciante. Default "piena di bacino":
    # solo ponti su corso d'acqua, fondazioni a rischio erosione x3, pile e spalle.
    # Tutti i valori numerici sono PROVISIONAL (docs/CHANGELOG_MODEL.md).
    regional_events_enabled: bool = False
    regional_event_rate_y: float = 0.05                # PROVISIONAL: uno ogni 20 anni sul parco
    regional_event_center_mode: str = "random_bridge"  # "random_bridge" | "fixed"
    regional_event_center_lat: float = 0.0             # solo con center_mode="fixed"
    regional_event_center_lon: float = 0.0
    regional_event_radius_km: float = 25.0             # PROVISIONAL
    regional_event_decay_model: str = "exponential"    # "exponential" | "gaussian" | "uniform"
    regional_event_decay_km: float = 15.0              # PROVISIONAL
    regional_event_requires: str = "crosses_water"     # attributi richiesti (vuoto = tutti i ponti)
    regional_event_multipliers: Dict[str, float] = field(default_factory=lambda: {"scour_critical": 3.0})  # PROVISIONAL
    regional_event_hit_prob: float = 0.3               # PROVISIONAL: prob. di danno al centro
    regional_event_targets: str = "pile,spalle"
    regional_event_hits: str = "all"
    regional_event_drop_dist: str = "lognormal"
    regional_event_drop_mean: float = 0.15             # PROVISIONAL (al centro, poi x intensita')
    regional_event_drop_cov: float = 0.5               # PROVISIONAL
    regional_event_scale_damage: bool = True
    regional_event_announce_prob: float = 0.2          # PROVISIONAL (danno visibile dopo la piena)

    # --- EVENTI PROGRESSIVI / LATENTI (infralab/core/hazards/progressive_events.py) ---
    # Opt-in: con progressive_events_enabled=False nessun effetto (snapshot invariato).
    # Modello delay-time: un difetto si innesca (flusso a chiave dedicato), si sviluppa per
    # una finestra di durata casuale, e' nascosto finche' un'ispezione non lo rileva, puo'
    # essere arrestato; se non arrestato, a fine finestra perdita di condizione e, con
    # probabilita' p_collapse_if_untreated, crollo. Fonti e intervalli plausibili: nota
    # letteratura-eventi-progressivi.md (vedi docs/CHANGELOG_MODEL.md). Flag come nella
    # nota: FROM SOURCE = letto o derivato da una fonte; PROVISIONAL = scelta di modello.
    progressive_events_enabled: bool = False
    # Quota dei crolli idraulici DIRETTI riassegnata al canale progressivo: il tasso
    # diretto e' moltiplicato per (1 - quota). Vale solo con il tipo idraulico acceso.
    hydraulic_collapse_progressive_share: float = 0.0          # PROVISIONAL (0 = storico)
    # Idraulico (scalzamento): ponti su acqua, stessa esposizione del crollo diretto.
    progressive_hydraulic_enabled: bool = True
    # "calibrated": innesco = quota x tasso diretto storico del ponte / p_collapse_if_untreated
    # (tasso atteso di crollo idraulico invariato per chi non ispeziona); "fixed": tasso dato.
    progressive_hydraulic_rate_mode: str = "calibrated"
    progressive_hydraulic_rate_y: float = 1e-3                 # PROVISIONAL (solo "fixed"; 3e-4 - 3e-3)
    progressive_hydraulic_scour_multiplier: float = 3.0        # FROM SOURCE (derivato, odds ratio 3.2; solo "fixed")
    progressive_hydraulic_duration_dist: str = "geometric"     # PROVISIONAL ("fino alla prossima piena sopra soglia")
    progressive_hydraulic_duration_mean_q: float = 20.0        # FROM SOURCE il campo (T = 2-25 anni), PROVISIONAL il default
    progressive_hydraulic_duration_cov: float = 1.0            # PROVISIONAL (solo "lognormal")
    progressive_hydraulic_detect_factor_l1: float = 0.5        # PROVISIONAL (direzione FROM SOURCE; 0.2 - 0.8)
    progressive_hydraulic_detect_factor_detailed: float = 1.0  # PROVISIONAL
    progressive_hydraulic_self_evident_prob: float = 0.1       # PROVISIONAL (0 - 0.3)
    progressive_hydraulic_p_collapse_if_untreated: float = 0.05  # FROM SOURCE il campo (0.01 - 0.5), PROVISIONAL il default
    progressive_hydraulic_arrest_cost_ratio: float = 0.3       # PROVISIONAL (0.1 - 0.5)
    progressive_hydraulic_regional_trigger_prob: float = 0.0   # PROVISIONAL (inneschi in piu' rispetto alla taratura)
    # Geotecnico (versante instabile): ponti con l'attributo on_unstable_slope (dallo
    # scenario) o, se assente, una quota assegnata con un flusso a chiave (0 = nessuno).
    progressive_geotechnical_enabled: bool = True
    progressive_geotechnical_exposed_fraction: float = 0.0     # PROVISIONAL (nessun dato: variabile utente)
    progressive_geotechnical_rate_y: float = 1e-3              # PROVISIONAL (1e-4 - 5e-3)
    progressive_geotechnical_rate_other_y: float = 0.0         # PROVISIONAL
    progressive_geotechnical_duration_dist: str = "lognormal"  # PROVISIONAL (coda lunga)
    progressive_geotechnical_duration_mean_q: float = 4.0      # FROM SOURCE il campo (1 - 12 trimestri), PROVISIONAL il default
    progressive_geotechnical_duration_cov: float = 1.0         # PROVISIONAL
    progressive_geotechnical_detect_factor_l1: float = 1.0     # PROVISIONAL (segni in superficie)
    progressive_geotechnical_detect_factor_detailed: float = 1.0  # PROVISIONAL
    progressive_geotechnical_self_evident_prob: float = 0.3    # PROVISIONAL (0.1 - 0.6)
    progressive_geotechnical_p_collapse_if_untreated: float = 0.10  # PROVISIONAL (0.02 - 0.5)
    progressive_geotechnical_arrest_cost_ratio: float = 0.3    # PROVISIONAL (0.1 - 0.7)
    progressive_geotechnical_targets: str = "pile,spalle"
    progressive_geotechnical_hits: str = "all"
    progressive_geotechnical_drop_dist: str = "lognormal"
    progressive_geotechnical_drop_mean: float = 0.35           # FROM SOURCE qualitativo ("grave"), PROVISIONAL il valore
    progressive_geotechnical_drop_cov: float = 0.5             # PROVISIONAL
    # Tipi utente: lista di dizionari (chiavi in progressive_events.CUSTOM_EVENT_KEYS).
    progressive_custom_events: List[Dict[str, Any]] = field(default_factory=list)
    # Arresto: azione dedicata "arresto_difetto" (aggiunta al catalogo con il canale
    # acceso), costo = arrest_cost_ratio del tipo x costo dell'intervento di riferimento.
    progressive_arrest_reference_intervention: str = "ripristino_strutturale"   # PROVISIONAL
    progressive_arrest_duration_quarters: int = 1              # PROVISIONAL
    # Un intervento arresta il difetto se il suo guadagno previsto su ogni componente
    # bersaglio e' almeno questo (il bonus 0.15 delle sostituzioni altrui non basta).
    progressive_arrest_min_gain: float = 0.3                   # PROVISIONAL
    # Livello d'ispezione (inspection_level del catalogo) da cui vale il fattore "detailed".
    progressive_detailed_inspection_min_level: int = 2
    # "inspection": con il modello POD acceso il difetto si vede solo se l'ispezione ha
    # rilevato un componente bersaglio; "independent": la POD del motore non conta.
    progressive_detection_pod_mode: str = "inspection"
    progressive_monitoring_detect_prob: float = 0.0            # PROVISIONAL
    # Con la soglia etica accesa, un difetto NOTO e' un trigger: arresto d'ufficio o chiusura.
    progressive_floor_trigger: bool = True

    # --- REGISTRO DEI COLLASSI E FINE PARTITA (opt-in; default = storico) ---
    # env.collapse_log registra sempre ogni collasso (causa, stato noto al gestore il
    # trimestre prima). collapse_game_over: "none" = la partita continua (storico con
    # failure_model="reliability"); "any" = ogni collasso elimina il gestore;
    # "known_critical" = solo se il ponte era NOTO come critico all'inizio del trimestre
    # (stato noto sotto ethical_floor_condition / ethical_floor_beta o difetto in sviluppo noto).
    collapse_game_over: str = "none"
    collapse_game_over_exempt_closed: bool = True   # un ponte chiuso al traffico che crolla non elimina
    collapse_log_metrics: bool = False              # contatori per causa nelle uscite anche a canale spento

    # --- CAPACITA' OPERATIVA -----------------------------------------------------
    # Ispettori e squadre di lavoro sono persone diverse, quindi hanno capacita'
    # separate. Quella ispettiva e' il vincolo che morde davvero quando una legge
    # impone cadenze fitte su molti ponti: le ispezioni obbligatorie sui ponti peggiori
    # consumano gli ispettori che servirebbero agli altri, e li' l'informazione arriva
    # in ritardo. 0 = illimitata (comportamento storico).
    inspection_capacity: float = 0.0     # ispezioni per trimestre per gestore
    works_capacity: float = 0.0          # interventi contemporanei per gestore
    # Con crew_capacity > 0 le due attingono invece a un'unica riserva condivisa
    # (variante da testare, non predefinita).
    crew_capacity: float = 0.0
    # Carico di squadra per tipo di azione, in squadre-trimestre. Un intervento occupa
    # la sua squadra per tutta la durata dei lavori; un'analisi per un solo trimestre.
    crew_load_intervention: float = 1.0
    crew_load_preventive: float = 0.5
    crew_load_analysis: float = 0.25
    crew_load_monitoring: float = 0.1

    # --- CONOSCENZA INIZIALE, SOGLIA ETICA, ORACOLO (opt-in; default = storico) ---
    # L'inventario nazionale E' un'ispezione: con True, al reset ogni ponte riceve una
    # condizione osservata come dopo un'ispezione L1 al trimestre 0 (stesso rumore e
    # stessa POD), a costo zero, non contata come azione ne' come spesa del gestore, con
    # un flusso casuale dedicato che non sposta quelli degli altri processi.
    initial_inventory_known: bool = False
    # Soglia etica: non si puo' lasciar cadere consapevolmente un ponte. Ogni trimestre,
    # dopo il piano del gestore e le azioni imposte, un ponte il cui stato NOTO scende
    # sotto soglia riceve l'intervento strutturale piu' economico adeguato, a carico del
    # gestore, se la cassa e le squadre lo consentono; altrimenti viene chiuso al
    # traffico, e riaperto quando lo stato noto torna sopra soglia.
    #  ethical_floor_condition: soglia sulla condizione minima OSSERVATA (mai quella vera);
    #  ethical_floor_beta: soglia opzionale sul beta annuo STIMATO dal gestore.
    ethical_floor_enabled: bool = False
    ethical_floor_condition: float = 0.2
    ethical_floor_beta: float | None = None
    # Risposta della soglia etica a un problema noto (opt-in, default invariato):
    #  "repair_or_close": ripara se cassa e squadre lo consentono, altrimenti chiude;
    #  "close_only": non ripara mai, chiude il ponte finche' lo stato noto resta sotto soglia;
    #  "restrict_only": non ripara ne' chiude, limita il carico (posting_restriction_factor).
    # Serve a separare il valore della rilevazione da quello dell'intervento.
    ethical_floor_response: str = "repair_or_close"
    # Oracolo: gli AGENTI ricevono lo stato vero (condizione minima, osservazioni per
    # componente, beta e Pf). Trigger di legge e punteggio restano su observation_mode.
    # Serve come limite superiore per misurare il margine di un benchmark.
    oracle_observation: bool = False

    # --- PIANIFICATORI (GA, memetico, MCTS; opt-in; default = storico) ---
    # Su quale mondo simulano i pianificatori (BridgeEnv.planning_clone):
    #  "truth"  = storico: copia dello stato VERO del motore (pianificatori sempre oracolo);
    #  "belief" = copia costruita sulla conoscenza del gestore: con observation_mode=
    #             "inspected" e senza oracolo lo stato nascosto (condizione, danno, eta'
    #             efficace, parametri latenti, ultimo danno idraulico) e' riestratto dalla
    #             credenza a ogni copia; con oracle_observation=True lo stato resta vero.
    #             Il futuro casuale della copia e' comunque indipendente da quello reale.
    planner_information: str = "truth"
    # Obiettivo dei rollout:
    #  "asset_value"   = storico (budget finale + valore patrimoniale; memetico: punteggio
    #                    composito storico a un passo);
    #  "risk_adjusted" = minimizza sull'orizzonte spesa del gestore + perdita attesa da
    #                    collasso (expected_quarterly_failure_loss, = metric_risk_cost_quarter)
    #                    + planner_safety_weight x deficit di sicurezza, con deficit =
    #                    somma su trimestri e ponti di max(0, beta_target_annuo - beta_annuo).
    planner_objective: str = "asset_value"
    # lambda del deficit di sicurezza (EUR per unita' di beta x ponte x trimestre).
    planner_safety_weight: float = 0.0
    # Valore terminale dell'obiettivo "risk_adjusted": a fine rollout si aggiungono N
    # trimestri di perdita attesa (+ lambda x deficit) sullo stato finale tenuto fermo.
    # 0 = nessun termine (storico).
    planner_terminal_quarters: int = 0

    # --- SANDBOX DELLE POLICY (integrita' del benchmark; opt-in; default = storico) ---
    # True: le policy non ricevono mai il motore. Ricevono un PlanningHandle
    # (infralab.agents.sandbox) con l'osservazione SANITIZZATA del proprietario (copia
    # JSON, senza seme; in "inspected" solo stime da evidenza, anche per danno
    # idraulico, prova di carico e monitoraggio), il catalogo, le regole e copie del
    # mondo del gestore (belief_clone) con futuro casuale proprio. Lo stato vero
    # arriva solo per dichiarazione (observation_mode="full") o concessione
    # (oracle_observation=True). Accessi vietati -> SandboxViolation (registrata).
    # Per i benchmark usarlo con strict_information=True (vedi sotto): il sandbox
    # filtra cio' che la policy legge, non i segnali dal vero calcolati dal motore.
    policy_sandbox: bool = False
    # --- INFORMAZIONE STRETTA (integrita' del benchmark; opt-in; default = storico) ---
    # True: nessun segnale derivato dallo stato vero entra nella conoscenza del gestore.
    # (1) l'incertezza epistemica non cresce a un danno vero non osservato (idraulico,
    #     sismico): cresce (x1.10) quando un'osservazione mostra un calo del centro
    #     dell'intervallo osservato maggiore di `strict_uncertainty_drop` rispetto
    #     all'osservazione precedente, e il guadagno di conoscenza di un'ispezione non
    #     dipende dall'esito (vero) della POD; la prova di carico fallita (esito
    #     osservato) la aumenta come prima;
    # (2) la prova di carico sceglie alpha, passi e componenti da provare sullo stato
    #     STIMATO dal gestore (salvo "full"/oracolo), la fisica della prova resta sul vero;
    # (3) componenti mai osservati: stima (e copia "belief") con la vita utile NOMINALE
    #     dell'inventario/profilo, non quella campionata;
    # (4) con observation_mode="inspected" la classificazione dichiarativa del regime
    #     legge beta ed eta' dello stato stimato (anche con l'oracolo: la legge non e'
    #     l'oracolo). Raccomandato con policy_sandbox=True nei benchmark.
    strict_information: bool = False
    # Calo del centro dell'intervallo osservato fra due osservazioni successive letto
    # come evidenza di danno (solo con strict_information=True).
    strict_uncertainty_drop: float = 0.10
    # Fonte della salute che riduce i ricavi trimestrali (enforce_guidelines e adattatori
    # di produzione per tipo di asset): "truth" = stato vero (storico; il budget porta un
    # segnale dal vero), "assessed" = stima del gestore (con observation_mode="inspected"),
    # "none" = ricavi indipendenti dalla condizione. Per il benchmark: "assessed" o "none".
    revenue_health_source: str = "truth"
    max_quarters: int | None = None

    def __post_init__(self):
        problems: List[str] = []
        if self.observation_mode not in ("full", "inspected"):
            problems.append(f"observation_mode deve essere 'full' o 'inspected' (trovato {self.observation_mode!r})")
        if self.true_reliability_mode not in ("knowledge", "physical"):
            problems.append(f"true_reliability_mode deve essere 'knowledge' o 'physical' (trovato {self.true_reliability_mode!r})")
        if self.assessment_epistemic_mode not in ("manager", "reference"):
            problems.append(f"assessment_epistemic_mode deve essere 'manager' o 'reference' (trovato {self.assessment_epistemic_mode!r})")
        if self.assessment_confidence_mode not in ("none", "knowledge_level"):
            problems.append(f"assessment_confidence_mode deve essere 'none' o 'knowledge_level' (trovato {self.assessment_confidence_mode!r})")
        try:
            _lv = [(float(s), float(f)) for s, f in self.assessment_confidence_levels]
            if not _lv or any(f < 1.0 for _, f in _lv) or [s for s, _ in _lv] != sorted((s for s, _ in _lv), reverse=True):
                problems.append("assessment_confidence_levels: gradini [soglia, FC] con FC >= 1 e soglie decrescenti")
        except (TypeError, ValueError):
            problems.append(f"assessment_confidence_levels non valido (trovato {self.assessment_confidence_levels!r})")
        if self.assessment_parameter_mode not in ("mean", "sampled"):
            problems.append(f"assessment_parameter_mode deve essere 'mean' o 'sampled' (trovato {self.assessment_parameter_mode!r})")
        if self.assessment_state_mode not in ("last_observation", "projected"):
            problems.append(f"assessment_state_mode deve essere 'last_observation' o 'projected' (trovato {self.assessment_state_mode!r})")
        if self.assessment_seed_mode not in ("distinct", "shared"):
            problems.append(f"assessment_seed_mode deve essere 'distinct' o 'shared' (trovato {self.assessment_seed_mode!r})")
        if not float(self.inspection_sigma_scale) > 0.0:
            problems.append(f"inspection_sigma_scale deve essere > 0 (trovato {self.inspection_sigma_scale})")
        if not 0.0 <= float(self.temporary_closure_rate_q) <= 1.0:
            problems.append(f"temporary_closure_rate_q deve stare in [0,1] (trovato {self.temporary_closure_rate_q})")
        if not 0.0 <= float(self.temporary_closure_condition_rate_q) <= 0.5:
            problems.append(f"temporary_closure_condition_rate_q deve stare in [0,0.5] (trovato {self.temporary_closure_condition_rate_q})")
        if not 0.0 <= float(self.posting_severity_floor) <= 2.0:
            problems.append(f"posting_severity_floor deve stare in [0,2] (trovato {self.posting_severity_floor})")
        if int(self.temporary_closure_mean_duration_quarters) < 1:
            problems.append(f"temporary_closure_mean_duration_quarters deve essere >= 1 (trovato {self.temporary_closure_mean_duration_quarters})")
        if self.rng_streams not in ("shared", "keyed"):
            problems.append(f"rng_streams deve essere 'shared' o 'keyed' (trovato {self.rng_streams!r})")
        if self.degradation_pause_on not in ("any_action", "interventions"):
            problems.append(f"degradation_pause_on deve essere 'any_action' o 'interventions' (trovato {self.degradation_pause_on!r})")
        if self.failure_model not in ("condition", "reliability"):
            problems.append(f"failure_model deve essere 'condition' o 'reliability' (trovato {self.failure_model!r})")
        if self.years <= 0:
            problems.append(f"years deve essere > 0 (trovato {self.years})")
        if self.deterioration_exponent <= 0:
            problems.append(f"deterioration_exponent deve essere > 0 (trovato {self.deterioration_exponent})")
        if self.deterioration_process_cov < 0:
            problems.append(f"deterioration_process_cov non puo' essere negativo (trovato {self.deterioration_process_cov})")
        if self.reliability_consequence_class.upper() not in ("CC1", "CC2", "CC3"):
            problems.append(f"reliability_consequence_class deve essere CC1/CC2/CC3 (trovato {self.reliability_consequence_class!r})")
        if self.reliability_reference_period_years < 1:
            problems.append(f"reliability_reference_period_years deve essere >= 1 (trovato {self.reliability_reference_period_years})")
        if self.reliability_mc_samples < 8:
            problems.append(f"reliability_mc_samples deve essere >= 8 (trovato {self.reliability_mc_samples})")
        if self.reliability_projection_years < 1:
            problems.append(f"reliability_projection_years deve essere >= 1 (trovato {self.reliability_projection_years})")
        if not 0.0 <= self.failure_threshold < 1.0:
            problems.append(f"failure_threshold deve stare in [0,1) (trovato {self.failure_threshold})")
        if self.traffic_load_model not in TRAFFIC_LOAD_MODELS:
            problems.append(f"traffic_load_model deve essere uno di {TRAFFIC_LOAD_MODELS} (trovato {self.traffic_load_model!r})")
        for prob_field in ("traffic_congestion_probability", "traffic_monthly_overload_probability", "earthquake_annual_prob"):
            value = getattr(self, prob_field)
            if not 0.0 <= value <= 1.0:
                problems.append(f"{prob_field} deve stare in [0,1] (trovato {value})")
        if self.seismic_hazard_decay_km <= 0:
            problems.append(f"seismic_hazard_decay_km deve essere > 0 (trovato {self.seismic_hazard_decay_km})")
        if not 0.0 <= self.seismic_hazard_far_pga <= self.seismic_hazard_peak_pga:
            problems.append("seismic_hazard: serve 0 <= far_pga <= peak_pga")
        if self.seed is not None and not isinstance(self.seed, int):
            problems.append(f"seed deve essere un intero o None (trovato {self.seed!r})")
        # 'custom:<nome>' seleziona un modello di degrado utente (plugin);
        # l'esistenza del plugin è verificata al caricamento nel motore.
        if self.degradation_model not in ("weibull", "gamma") and not self.degradation_model.startswith("custom:"):
            problems.append(
                f"degradation_model deve essere 'weibull', 'gamma' o 'custom:<nome>' (trovato {self.degradation_model!r})"
            )
        if self.gamma_process_theta <= 0:
            problems.append(f"gamma_process_theta deve essere > 0 (trovato {self.gamma_process_theta})")
        for trend_field in ("traffic_growth_annual", "climate_degradation_trend_annual"):
            trend_value = getattr(self, trend_field)
            if not -0.5 <= trend_value <= 0.5:
                problems.append(f"{trend_field} deve stare in [-0.5, 0.5] (trovato {trend_value})")
        for flag_field in ("initial_inventory_known", "ethical_floor_enabled", "oracle_observation",
                           "regime_override_unfunded", "policy_sandbox", "strict_information"):
            if not isinstance(getattr(self, flag_field), (bool, np.bool_)):
                problems.append(f"{flag_field} deve essere un booleano (trovato {getattr(self, flag_field)!r})")
        if not 0.0 <= float(self.ethical_floor_condition) <= 1.0:
            problems.append(f"ethical_floor_condition deve stare in [0,1] (trovato {self.ethical_floor_condition})")
        if self.ethical_floor_response not in ("repair_or_close", "close_only", "restrict_only"):
            problems.append("ethical_floor_response deve essere 'repair_or_close', 'close_only' o 'restrict_only' "
                            f"(trovato {self.ethical_floor_response!r})")
        if self.ethical_floor_beta is not None and not np.isfinite(float(self.ethical_floor_beta)):
            problems.append(f"ethical_floor_beta deve essere un numero finito o None (trovato {self.ethical_floor_beta!r})")
        if self.detour_partial_diversion not in ("all", "trucks"):
            problems.append(f"detour_partial_diversion deve essere 'all' o 'trucks' (trovato {self.detour_partial_diversion!r})")
        if float(self.detour_km_cap) < 0.0:
            problems.append(f"detour_km_cap deve essere >= 0 (trovato {self.detour_km_cap})")
        if self.revenue_health_source not in ("truth", "assessed", "none"):
            problems.append(f"revenue_health_source deve essere 'truth', 'assessed' o 'none' (trovato {self.revenue_health_source!r})")
        if not 0.0 <= float(self.strict_uncertainty_drop) <= 1.0:
            problems.append(f"strict_uncertainty_drop deve stare in [0,1] (trovato {self.strict_uncertainty_drop})")
        if self.planner_information not in ("truth", "belief"):
            problems.append(f"planner_information deve essere 'truth' o 'belief' (trovato {self.planner_information!r})")
        if self.planner_objective not in ("asset_value", "risk_adjusted"):
            problems.append(f"planner_objective deve essere 'asset_value' o 'risk_adjusted' (trovato {self.planner_objective!r})")
        try:
            _lam = float(self.planner_safety_weight)
        except (TypeError, ValueError):
            _lam = float("nan")
        if not (np.isfinite(_lam) and _lam >= 0.0):
            problems.append(f"planner_safety_weight deve essere un numero finito >= 0 (trovato {self.planner_safety_weight!r})")
        if not isinstance(self.planner_terminal_quarters, (int, np.integer)) or isinstance(self.planner_terminal_quarters, bool)                 or self.planner_terminal_quarters < 0:
            problems.append(f"planner_terminal_quarters deve essere un intero >= 0 (trovato {self.planner_terminal_quarters!r})")
        from infralab.core.hazards.sudden_damage import validate_params as _validate_sudden
        problems.extend(_validate_sudden(self))
        from infralab.core.hazards.regional_events import validate_params as _validate_regional
        problems.extend(_validate_regional(self))
        from infralab.core.hazards.progressive_events import validate_params as _validate_progressive
        problems.extend(_validate_progressive(self))
        if problems:
            raise ValueError("SimParameters non validi: " + "; ".join(problems))


@dataclass
class MaintOpsConfig:
    # --- LIVELLI DI ISPEZIONE / VALUTAZIONE (Linee Guida Ponti 2020) ---
    # L0 censimento (Scheda anagrafica): sblocca la gestione; nessun guadagno
    # conoscitivo sulla condizione ma imposta l'asset come "censito".
    L0_census: Dict[str, Any] = field(
        default_factory=lambda: {"cost_m2": 0.05, "error_sigma": 0.20, "type": "analysis", "census": True, "co2_m2": 0.05})
    # L1 ispezione visiva speditiva (ciclica, su tutti): alimenta la CdA.
    L1_inspect: Dict[str, Any] = field(
        default_factory=lambda: {"cost_m2": 1.0, "error_sigma": 0.10, "type": "analysis", "inspection_level": 1, "co2_m2": 0.2})
    # Sorveglianza intermedia (Linee Guida Ponti, Istruzione Operativa ANSFISA 7.4.1.2,
    # alternativa 2 per le classi Alta e Medio-Alta di tipo 1): ispezioni cadenzate ogni
    # tre mesi eseguite da personale addestrato ma non tecnico specializzato, che
    # accertano lo stato di consistenza e verificano che non vi siano evoluzioni
    # significative rispetto all'ispezione annuale. Non sostituiscono l'ispezione
    # ordinaria: costano poco, vedono peggio e NON azzerano la cadenza ordinaria.
    sorveglianza_intermedia: Dict[str, Any] = field(
        default_factory=lambda: {"cost_m2": 0.3, "error_sigma": 0.18, "type": "analysis",
                                 "inspection_level": 1, "knowledge_gain": 0.05,
                                 "resets_cadence": False, "co2_m2": 0.05})
    # L2 ispezione ravvicinata/speciale: misura migliore, riduce l'incertezza.
    L2_inspect: Dict[str, Any] = field(
        default_factory=lambda: {"cost_m2": 18.0, "error_sigma": 0.05, "type": "analysis", "inspection_level": 2, "co2_m2": 1.0})
    # L4 valutazione accurata / verifica di sicurezza (chiude una CdA Alta).
    L4_verify: Dict[str, Any] = field(
        default_factory=lambda: {"cost_m2": 100.0, "error_sigma": 0.01, "type": "analysis", "inspection_level": 4, "is_assessment": True, "co2_m2": 2.0})
    monitoring: Dict[str, Any] = field(
        default_factory=lambda: {"cost_m2": 70.0, "error_sigma": 0.005, "type": "monitoring", "knowledge_gain": 0.30, "co2_m2": 5.0})
    proof_load_test: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 25.0,
            "error_sigma": 0.01,
            "type": "proof_load",
            "alpha": 0.95,
            "proof_load_cov": 0.05,
            "uncertainty_reduction": 0.45,
            "damage_limit": 0.05,
            "knowledge_gain": 0.50,
            "co2_m2": 3.0
        })

    # --- AZIONI DI SOSTITUZIONE ('REPLACE') CON BONUS DEL 15% AGLI ALTRI ---
    replace_impalcato: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 4000.0, "co2_m2": 450.0, "type": "intervention", "duration_quarters": 4, "revenue_impact": 0.7,
            "gains": {"impalcato": 1.0, "pile": 0.15, "spalle": 0.15, "appoggi": 0.15}, "gain_std_dev": 0.0
        })
    replace_pile: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 3000.0, "co2_m2": 320.0, "type": "intervention", "duration_quarters": 3, "revenue_impact": 0.5,
            "gains": {"impalcato": 0.15, "pile": 1.0, "spalle": 0.15, "appoggi": 0.15}, "gain_std_dev": 0.0
        })
    replace_spalle: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 2500.0, "co2_m2": 260.0, "type": "intervention", "duration_quarters": 3, "revenue_impact": 0.5,
            "gains": {"impalcato": 0.15, "pile": 0.15, "spalle": 1.0, "appoggi": 0.15}, "gain_std_dev": 0.0
        })
    replace_appoggi: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost": 100000, "co2": 8000.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 0.0,
            "gains": {"impalcato": 0.15, "pile": 0.15, "spalle": 0.15, "appoggi": 1.0}, "gain_std_dev": 0.0
        })

    # --- AZIONI COMPOSTE ---
    ripristino_strutturale: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 1500.0, "co2_m2": 180.0, "type": "intervention", "duration_quarters": 3, "revenue_impact": 0.4,
            "gains": {"pile": 0.5, "spalle": 0.5}, "gain_std_dev": 0.05
        })
    ripristino_completo: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 3500.0, "co2_m2": 400.0, "type": "intervention", "duration_quarters": 5, "revenue_impact": 0.8,
            "gains": {"impalcato": 0.7, "pile": 0.6, "spalle": 0.5, "appoggi": 0.4}, "gain_std_dev": 0.05
        })

    # --- MANUTENZIONE PREVENTIVA (rallenta il degrado, non recupera condizione) ---
    # `decay_rate_multiplier` < 1 riduce il tasso di degrado per `rate_duration_quarters`.
    manutenzione_ordinaria: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 8.0, "co2_m2": 3.0, "type": "preventive", "decay_rate_multiplier": 0.75,
            "rate_duration_quarters": 8, "revenue_impact": 0.0})
    protezione_anticorrosione: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 40.0, "co2_m2": 12.0, "type": "preventive", "decay_rate_multiplier": 0.60,
            "rate_duration_quarters": 20, "revenue_impact": 0.05})

    # --- MISURE OPERATIVE / LIMITAZIONI D'USO (riducono la DOMANDA, non la condizione) ---
    # `traffic_restriction_factor` scala il traffico effettivo (domanda variabile).
    limitazione_carico: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 0.5, "co2_m2": 0.1, "type": "operational", "traffic_restriction_factor": 0.6,
            "revenue_impact": 0.15})
    senso_unico_alternato: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 0.8, "co2_m2": 0.2, "type": "operational", "traffic_restriction_factor": 0.35,
            "revenue_impact": 0.45})
    chiusura_ponte: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 1.0, "co2_m2": 0.1, "type": "operational", "traffic_restriction_factor": 0.0,
            "revenue_impact": 1.0})
    riapertura: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 0.2, "co2_m2": 0.05, "type": "operational", "traffic_restriction_factor": 1.0,
            "revenue_impact": 0.0})

    # --- INTERVENTI SISMICI (leva sulla vulnerabilità sismica) ---
    # `seismic_sensitivity_target` porta la sensibilità sismica al valore dato.
    miglioramento_sismico: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 1200.0, "co2_m2": 180.0, "type": "intervention", "duration_quarters": 4, "revenue_impact": 0.6,
            "gains": {"pile": 0.3, "spalle": 0.3}, "gain_std_dev": 0.05, "seismic_sensitivity_target": 0.6})
    adeguamento_sismico: Dict[str, Any] = field(
        default_factory=lambda: {
            "cost_m2": 2800.0, "co2_m2": 340.0, "type": "intervention", "duration_quarters": 6, "revenue_impact": 0.85,
            "gains": {"pile": 0.6, "spalle": 0.6, "appoggi": 0.5}, "gain_std_dev": 0.05, "seismic_sensitivity_target": 0.25})


@dataclass
class CdaConfig:
    bassa_threshold: float = 0.9
    bassa_due_quarters: int = 8
    medio_bassa_threshold: float = 0.8
    medio_bassa_due_quarters: int = 6
    media_threshold: float = 0.6
    media_due_quarters: int = 4
    medio_alta_threshold: float = 0.4
    medio_alta_due_quarters: int = 2
    alta_due_quarters: int = 1

    def __post_init__(self):
        thresholds = [self.bassa_threshold, self.medio_bassa_threshold, self.media_threshold, self.medio_alta_threshold]
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError(f"Soglie CdA non decrescenti: {thresholds}")
        if any(not 0.0 <= t <= 1.0 for t in thresholds):
            raise ValueError(f"Soglie CdA fuori da [0,1]: {thresholds}")


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyJSONEncoder, self).default(obj)
