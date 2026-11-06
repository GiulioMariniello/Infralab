# Changelog del modello

Registro dei cambiamenti **numerici intenzionali** al motore di affidabilità e
degrado. Ogni voce spiega cosa cambia nei risultati e perché. Gli snapshot di
regressione (`tests/fixtures/`) vanno rigenerati (`UPDATE_SNAPSHOTS=1 pytest`)
contestualmente alla voce corrispondente.

## 2026-11-07 — Libreria 1.0.0 per il paper (nessun cambiamento numerico)

Configurazioni `paper`, `paper-short` e `paper-less-optimistic` (specifiche Z-princ, Z-corta, ZT-princ della
campagna del paper) nel pacchetto; `paper` diventa il default di `make`, `run`, `compare_laws`, della CLI e del
server MCP. Norma `regbench/N24`. Esiti nuovi nel riepilogo di una partita: crolli idraulici attesi da difetti
progressivi, ponti e anni-ponte sotto 3,02. Licenza MIT, CITATION.cff. Partite del paper riprodotte al bit
attraverso la libreria (`tests/paper_reproduction_check.py`). Motore invariato, snapshot TRAIN-01 identico.

## 2026-11-02 — Scomposizione dello scarto stima - vero (default invariati)

Nuovi parametri, tutti con default = comportamento storico, per isolare i meccanismi
che separano la stima del gestore (`observation_mode="inspected"`) dallo stato vero:
`assessment_parameter_mode` (`mean` | `sampled`: la stima usa i parametri campionati
del ponte vero), `assessment_state_mode` (`last_observation` | `projected`: l'età
efficace invertita dall'osservazione avanza del tempo trascorso dal trimestre in cui
la stima vede l'osservazione), `assessment_seed_mode` (`distinct` | `shared`: stessi
campioni Monte Carlo del vero, tramite l'attributo `_mc_seed_id` della copia) e
`inspection_sigma_scale` (moltiplicatore della deviazione standard delle osservazioni,
default 1.0). Con `sampled` + `shared` + `assessment_epistemic_mode="reference"` e
osservazione perfetta la stima coincide con il vero. Snapshot TRAIN-01 identico;
test in `tests/test_assessment_decomposition.py`.

## 2026-10-15 — Valutatore prudente (assessment_confidence_mode; default invariato)

Nuovi parametri `assessment_confidence_mode` (default `none` = nessun effetto) e
`assessment_confidence_levels` (gradini `[soglia, FC]`, default
`((0.20, 1.35), (0.12, 1.20), (0.0, 1.00))`). Con `knowledge_level`, nella STIMA
del gestore (`observation_mode="inspected"`) la resistenza media di ogni stato
limite è divisa per il fattore di confidenza del gradino corrispondente
all'incertezza epistemica del gestore sul componente, come i fattori di
confidenza delle norme sugli esistenti (NTC 2018, Circolare 2019 C8.5.4: FC
1,35 / 1,20 / 1,00 per i livelli di conoscenza LC1-LC3). Lo stato vero e la Pf
fisica non cambiano. Serve a confrontare un gestore che stima per difetto con
quello storico. Snapshot TRAIN-01 identico; test in
`tests/test_assessment_confidence.py`.

## 2026-10-12 — Libreria, server MCP e giocatori LLM a strumenti (nessun cambiamento numerico)

Solo moduli nuovi attorno al motore; la dinamica non cambia e lo snapshot TRAIN-01 resta identico.

- `pyproject.toml`: il pacchetto si installa con `pip install .` (dipendenze di base: numpy, scipy, pandas,
  PyYAML, filelock; extra `mcp`, `anthropic`, `openai`, `llm`, `web`, `dev`). Comandi `infralab` e `infralab-mcp`.
- `infralab/api.py`: API pubblica (`make`, `Env`, `run`, `compare_laws`, cataloghi). `compare_laws` usa gli
  stessi semi figli del runner delle campagne: la replica 0 della campagna N2 e' riprodotta al bit (test).
- `infralab/benchmarks/regbench/`: parchi (147 e 30 ponti), leggi D0-D3, G0-G3, J0-J1, K0 e le tre
  configurazioni del paper, come dati del pacchetto.
- `infralab/views.py`: osservazione compatta costruita dalla sola osservazione della policy (circa 18 kB per
  30 ponti contro 2 MB): la regola di informazione non cambia.
- `infralab/toolkit.py`, `infralab/mcp_server.py`: strumenti condivisi e server MCP (mcp 1.x e 2.x). Durante
  una partita gli strumenti non restituiscono esiti calcolati sullo stato vero (S1, crolli attesi).
- `infralab/llm/`: giocatore che decide attraverso gli strumenti; moduli separati per Claude (SDK Anthropic)
  e per il protocollo chat di OpenAI (anche endpoint compatibili). Conta decisioni senza piano e voci malformate.
- `docs/LIBRARY.md`: documentazione della libreria (in inglese, come le docstring dei moduli nuovi).
- 2026-10-13: `degradation=` in `make`, `run` e `compare_laws` (`weibull`, `gamma`, un identificativo
  `regbench/...` o il percorso di un file con `degrade(ctx)`); il file e' validato e registrato come legge utente
  del motore. Leggi fornite: `regbench/power` (legge a potenza) e `regbench/markov-nbi` (catena di Markov sulle
  valutazioni dell'inventario federale, stimata su quattro stati). Nessuna modifica al motore.

## 2026-07-02 — Normative testabili (CodeFramework; default invariato)

Nuovo parametro `code_framework` (default `custom` = passthrough, comportamento
storico invariato). Selezionando una normativa registrata (`EN1990`,
`fib_MC2010`, `AASHTO_LRFD`) i target di affidabilità vengono risolti dal codice
per la classe di conseguenza corrente (β target, periodo di riferimento, β SLS,
incremento fragile) e sovrascrivono i target manuali. Applicato in
`BridgeEnv.__init__` via `_apply_code_framework` (`dataclasses.replace`). Con
`custom` non cambia nulla → snapshot identico. Valori documentati nelle sorgenti
(EN 1990 Tab. B2; fib MC2010 §3.3/§7.3; AASHTO LRFD / Nowak NCHRP 368).

## 2026-07-02 — Fase D (estensioni fisiche; default invariati)

Tutte le estensioni sono **opt-in**: con i parametri di default il
comportamento (e lo snapshot TRAIN-01) è identico.

- **Processo gamma** (`degradation_model: "gamma"`): degrado con incrementi
  Gamma(ΔD/θ, θ) alla van Noortwijk; stessa traiettoria media della Weibull
  (E[incremento] = ΔD), varianza ∝ `gamma_process_theta`. Verificato: media e
  varianza degli incrementi vs analitico; per θ→0 il percorso converge alla
  media Weibull.
- **Modello ispezioni** (`infralab/core/inspection.py`): rumore (come prima),
  più `inspection_bias` (ottimismo/pessimismo sistematico) e POD opzionale
  (`inspection_pod_enabled`, P(detect)=1−C^k): danno non rilevato →
  osservazione ottimistica e nessun guadagno epistemico.
- **Non-stazionarietà**: `traffic_growth_annual` (ADT efficace cresce nel
  tempo, entra nel moltiplicatore di invecchiamento e nella domanda) e
  `climate_degradation_trend_annual` (moltiplicatore annuo del tasso di
  invecchiamento). La cache delle proiezioni include il fattore di crescita.

## 2026-07-02 — Fase A2 (refactor InfraLab)

### Posterior proof-load: condizionamento su domanda di prova aleatoria
Prima: il posterior analitico troncava R su una soglia **deterministica**
`x0 = dead_load_mean + α·code_live_effect`, mentre il passo Monte Carlo di
`assess_proof_load` campionava la domanda di prova con θ_test (COV
`proof_load_cov`) e peso proprio aleatorio (COV `dead_load_cov`). Le due
convenzioni erano inconsistenti: il posterior sovrastimava l'informatività del
test.

Ora: `_lognormal_survival_posterior(R_mean, R_cov, D_mean, D_cov)` condiziona
R lognormale sull'evento R > D_test con D_test lognormale indipendente
(momenti esatti in forma chiusa; per D_cov→0 si riducono alle formule della
troncata di Aitchison & Brown, verificato nei test). Il COV della domanda di
prova combina θ_test e peso proprio:
`cov_D = sqrt(proof_load_cov² + (dead_load_cov·dead_mean/D_mean)²)`.

**Effetto sui numeri:** il guadagno del proof load su β_posterior è
leggermente **minore** rispetto a prima (test meno informativo perché la
soglia effettiva è incerta). Direzione conservativa. L'etichetta
`conditioning_mode` diventa `survival_conditioning_random_test_demand`.

### CDF normale via `scipy.special.ndtr`
Sostituisce `np.vectorize(math.erf)` nel banco di campioni. Differenze
nell'ultimo ulp (≤2.2e-16) sui campioni Gumbel: nessun effetto pratico; lo
snapshot TRAIN-01 è rimasto identico.

### Rimozioni (nessun effetto: codice morto)
- `posterior_from_proof_load` e `lognormal_reliability` (percorso legacy mai
  chiamato, superseded da `assess_proof_load`).
- campione `theta_stop` (calcolato e mai usato). Il banco normale resta a 8
  colonne per non alterare le estrazioni delle altre variabili.

### Validazione input (errore invece di KeyError)
`evaluate_limit_states` e `assess_proof_load` ora sollevano
`LimitStateSpecError` con messaggio esplicito per stati limite vuoti, chiavi
obbligatorie mancanti (`resistance_mean`, `dead_load_mean`, `live_load_mean`,
più `code_live_effect` per il proof load), valori non finiti o negativi.

### RNG per-istanza e nuova semantica del seed (Fase A3)
Prima: `seed=0` (default) NON seedava nulla (`if seed > 0`), quindi il degrado
era non riproducibile mentre l'MC di affidabilità restava deterministico; i
draw usavano lo stato globale di `random`/`np.random` (contaminabile da
qualunque altro modulo).

Ora: ogni `BridgeEnv` ha il proprio `np.random.Generator`. Semantica:
qualunque seed intero (0 compreso) → run deterministico; `seed=None` →
entropia. L'MC di affidabilità resta sempre deterministico (base 0 se seed
None). `earthquake_module` accetta `rng` iniettato. Nessuno stato RNG globale
viene toccato (test dedicato).

**Effetto sui numeri:** le sequenze casuali cambiano (Generator PCG64 vs
Mersenne `random.gauss`): le traiettorie simulate con lo stesso seed di prima
sono diverse ma statisticamente equivalenti. Snapshot TRAIN-01 rigenerato.

### Danno sismico: inversione età efficace consistente (Fase A3)
`apply_earthquake_damage` aggiornava l'età efficace con una legge di potenza
`(1-C)^(1/n)` inconsistente con la curva di sopravvivenza Weibull. Ora usa
`Component.apply_damage` (stessa inversione del resto del modello, con
incremento dell'incertezza epistemica). Floor di condizione 0.05 invariato.
Effetto solo con sisma abilitato (default: off).

### Cache proiezioni life-cycle (Fase A3, nessun effetto numerico)
`_project_component_life_cycle` è cacheata per (ponte, componente, stato):
la proiezione è deterministica dato lo stato, i refresh ripetuti nello stesso
turno riusano il risultato.

### Costanti nominate (nessun effetto numerico)
`BETA_SAFE_CEILING=8.0`, `GOVERNING_MARGIN_SENTINEL=999`,
`POSTERIOR_MIN_MATERIAL_COV=0.01`, `POSTERIOR_MIN_MODEL_DEMAND_COV=0.02`,
`POSTERIOR_DEMAND_MODEL_FACTOR=0.95`, `DEFAULT_*_COV` (JCSS).

## 2026-07-04 — Azioni e workflow Linee Guida Ponti 2020

Aggiunti al catalogo azioni (MaintOpsConfig): livelli di ispezione L0_census / L1_inspect / L2_inspect / L4_verify (con inspection_level, is_assessment); manutenzione preventiva (manutenzione_ordinaria, protezione_anticorrosione, type="preventive", decay_rate_multiplier < 1 per rate_duration_quarters); misure operative (limitazione_carico, senso_unico_alternato, chiusura_ponte, riapertura, type="operational", traffic_restriction_factor); interventi sismici (miglioramento_sismico, adeguamento_sismico, seismic_sensitivity_target).

Nuovi parametri SimParameters (tutti OPT-IN, default neutro → golden snapshot TRAIN-01 invariato): cda_multi_hazard (default False), l4_escalation_quarters (default 0), l4_escalation_risk_penalty (2.0), inspection_overdue_grace_quarters (2). Nuovi attributi Bridge: censused (default True), last_L4_quarter, quarters_in_cda_alta_without_L4, traffic_restriction_factor (1.0), maintenance_rate_multiplier (1.0), rate_multiplier_until_quarter.

Effetti: traffic_restriction_factor scala _traffic_multiplier e _get_revenue_multiplier; maintenance_rate_multiplier scala il rate di degrado quando attivo; seismic_sensitivity_target abbassa la vulnerabilità sismica a intervento completato; _apply_linee_guida_workflow() (post-step) applica CdA multi-rischio con regola max() ed escalation L4 (penalità su risk_priority_index, che NON è nello snapshot). Nessuna azione/flag altera condizione/β/pf su traiettoria "do nothing": snapshot dorato invariato.

## 2026-09-17 — Regime normativo come variabile dell'ambiente (opt-in; default invariato)

Nuovo package `infralab.regimes`: la legge che governa il parco (obblighi
ispettivi, enforcement, sanzioni, tier di target, regole informative e sulle
risorse) è un oggetto leggibile dalla macchina (`RegulatoryRegime`, YAML/JSON)
selezionato dal nuovo parametro `SimParameters.regulatory_regime` (default
`none` = passthrough: snapshot dorato TRAIN-01 invariato, 379 test verdi).

Leve: `obligations` (trigger → azione dovuta: inspection_overdue,
l4_escalation_overdue, risk_above_threshold, beta_below_target,
condition_below_threshold, not_censused; con `grace_quarters`);
`enforcement.mode` ∈ {none, forced, sanction, both} con `sanction_ladder`
(misure operative progressive per durata dell'inadempienza) e
`fine_per_quarter_m2`; `target_tier` (override di `code_framework`);
`information.census_required`; `resources.earmarked_fraction` (quota di budget
riservata ad azioni di analisi); `param_overrides` (qualunque campo di
SimParameters fissato dalla legge).

Hook nel motore (tutti inattivi con regime `none`): (1) pre-step, esecuzione
d'ufficio: le azioni dovute non proposte dal gestore vengono inserite nel piano,
eseguite anche a budget esaurito e addebitate (`charged_to: owner`) o a carico
pubblico (`state`); prevalgono sull'azione del gestore per lo stesso asset;
(2) `_sanitize_agent_plan`: rifiuto per `earmarked_budget` e, con sanzione
attiva, per `regime_sanction_active` (qualunque azione operativa del gestore);
(3) post-step `_apply_regime_enforcement`: aggiorna la durata di ciascun
trigger, applica multe, impone la misura della scala sanzionatoria
(`traffic_restriction_factor` = min corrente/misura, letta dal catalogo completo
anche se lo scenario ha un catalogo ridotto) e la revoca quando l'obbligo è
assolto. Stato persistito per asset (`regime_trigger_since`, `regime_sanction`)
e per agente (`regime_stats`: forced_actions/cost/deferred,
sanction/closure_bridge_quarters, fines_total, rejected_actions,
breach_bridge_quarters). Osservazione: `obs["regime"]`, `obs["regime_stats"]`,
per asset `regime_mandatory_action/trigger` e `regime_sanction`. Il gym espone
`regime_*` e `regime_sanctioned_assets` nelle metriche.

Regimi built-in (`infralab/regimes/builtin/*.yaml`): `it_lg2020_signal`,
`it_lg2020_forced`, `it_lg2020_sanctioned` (Linee Guida Ponti 2020: solo
segnale / esecuzione d'ufficio / sanzione limitazione→senso alternato→chiusura),
`hyp_risk_based_existing` (ipotetica: tier NEN 8700, L4 forzata sopra soglia di
rischio, censimento obbligatorio, 10% budget vincolato). Regimi utente:
`get_regime("<path>.yaml")` o `load_regime_file`.

Nuovi tier di target (`infralab.codes`): `NEN8700_disapproval` (CC1/CC2/CC3 =
1.8/2.5/3.3, 15 anni) e `NEN8700_reconstruction` (2.8/3.3/3.8, 30 anni) per
strutture esistenti; valori indicativi da NEN 8700:2011+A1:2020 Tab. A.1, ISO
13822, fib Bulletin 80: da verificare sulla tabella prima di ogni uso normativo.

Nuovo tipo di agente `Basic` (`infralab.agents.basic`, policy: do_nothing,
danger_only, fixed_interval, worst_first, condition_hysteresis, risk_based,
compliance_minimal, inspect_and_repair): strategie elementari interpretabili
per gli esperimenti "stessa strategia, legge diversa". Nessun effetto sugli
agenti esistenti.

Effetto sui numeri: nessuno con `regulatory_regime: none`. Con un regime,
ricavi (sanzioni), budget (esecuzione d'ufficio, multe) e piani (rifiuti)
cambiano per costruzione.

## 2026-09-17 — Riproducibilità degli agenti di ricerca/apprendimento (seed globale per run)

GA a orizzonte mobile, memetico, MCTS e Q-learning tabellare usano il modulo
`random` globale di Python (mai seedato): due run con lo stesso seed davano
traiettorie diverse (verificato con due riesecuzioni dello stage smoke sullo
stesso commit; euristico e motore identici al bit). Ora `AssetEnvGym.reset(seed)`
e `arena.run_single_simulation` seedano `random` e `numpy.random` dal seed della
run (`seed_globals`). Effetto sui numeri: le traiettorie di GA/memetico/MCTS/RL
cambiano rispetto a prima (erano comunque non riproducibili) e diventano
riproducibili a parità di seed; nessun effetto su motore, euristico, agenti
Basic e snapshot dorato. Le campagne eseguite prima di questa voce (commit
ce3aec0) restano statisticamente valide ma non rigenerabili al bit per quelle
famiglie: vanno trattate come pilota.

## 2026-09-17 — Euristico: regola ispettiva separata da `enforce_guidelines`; tier NEN 8700 verificati

`HeuristicAgent` ha ora il parametro `follow_guidelines` (default True) che governa
"ispeziona alla scadenza"; prima leggeva `params.enforce_guidelines`, che controlla
anche il throttle sui ricavi del motore. Effetto sui numeri: nessuno con
`enforce_guidelines=True` (default); con `enforce_guidelines=False` l'euristico ora
ispeziona come previsto (prima non ispezionava mai). I tier `NEN8700_*` sono
verificati su HERON 68(2) 2023 Tab. 3 (riproduzione della tabella NEN 8700):
rinnovo con periodo di riferimento 15 anni (era 30). L'Arena accetta `--regime` e
`--param CAMPO=VALORE`.

## 2026-09-18 — Prestazioni: cache del banco Monte Carlo (nessun effetto numerico)

`_normal_bank` (campioni normali standard seedati per l'MC di affidabilità) aveva
una cache LRU di 24 posti, ma la proiezione life-cycle usa un seme distinto per
(asset, componente, anno): la cache veniva svuotata a ogni trimestre (hit 0,3%).
Ora 4096 posti: hit 98%, run tipico 2,6x più veloce, suite di test 221 s -> 106 s.
Verificato identico al bit: snapshot dorato e 42 run della campagna 2 rieseguiti
contro l'archivio del commit e66781a.

## 2026-09-18 — Costo atteso del rischio come metrica (nessun effetto sulla dinamica)

`AssetEnv.expected_quarterly_failure_loss(agent)`: per asset Pf annuale (posterior
se disponibile) / 4 x C0 x (1 + H/C0), C0 = valore iniziale (sostituzione),
H/C0 = nuovo parametro `failure_consequence_ratio` (default 3.0: caso di
riferimento di Steenbergen, Rozsas, Vrouwenvelder, HERON 63(3) 2018, Tab. 5;
classe "moderata" JCSS 2001, 1 <= H/C0 < 4). Il gym espone `risk_cost_quarter`,
`risk_cost_total`, `risk_cost_c0_total` (per ricalcolare il rischio con qualunque
H/C0 a posteriori) e `risk_adjusted_cost` = spesa effettiva + perdita attesa. Le
sanzioni (ricavi persi, multe) sono escluse per costruzione e si leggono a parte.
Anche l'Arena registra `risk_cost_total` e `risk_adjusted_cost`. Lettura pura dello
stato: snapshot dorato invariato, test di non interferenza dedicato.

## 2026-09-19 — Osservazioni stimate, collasso dalla Pf, leggi validate (opt-in; default invariati)

Correzioni emerse da una revisione avversaria (costruttore e distruttore) e
verificate nel codice.

- `observation_mode` ("full" storico | "inspected"). In "inspected" l'agente e i
  trigger del regime vedono uno stato STIMATO: condizione osservata dove
  ispezionato, credenza nominale altrove (eta' nominale, parametri medi del
  profilo), beta/Pf/rischio/proiezione ricalcolati con parametri nominali. Cache
  per chiave di conoscenza. Il punteggio resta sullo stato vero. La dinamica vera
  non dipende dalla modalita' (test dedicato). Prima gli agenti ricevevano
  condizione, eta', beta e Pf veri.
- `failure_model` ("condition" storico | "reliability"). In "reliability" il
  collasso e' estratto ogni trimestre con p = 1-(1-Pf)^(1/4) da un flusso dedicato
  (seme, asset, trimestre), cioe' numeri casuali comuni fra strategie, senza
  consumare l'RNG della simulazione. Il collasso e' registrato come perdita
  realizzata C0(1+H/C0) e l'opera e' ricostruita; la partita continua.
- Perdita attesa: stessa p trimestrale del campionatore, valutata nel passo sullo
  stato campionato (`env.last_expected_loss`). Coerenza verificata su 120 semi
  NBI-DE: attesa 530 M, realizzata 494 M (IC95 432-557 M).
- Regimi: azioni imposte validate al caricamento sul catalogo completo; l'autorita'
  esegue anche azioni fuori dal catalogo ridotto dello scenario. Nuovo trigger
  `pf_above_threshold`; la legge ipotetica lo usa (1e-3). Contatori
  `forced_unexecutable`, `overridden_owner_actions`.
- Agenti base: soglie di `risk_based` in scala di Pf annua (1e-4, 1e-3); variante
  auto-ispezionante di ogni regola (`self_inspect_interval_quarters`); nuova regola
  `target_keeper` che legge il target della norma.
- Metriche del gym: `failures_total`, `failures_quarter`, `failure_loss_total`,
  `failure_loss_c0_total`, `min_beta_annual_assessed`; le azioni non valide contano
  solo le richieste bloccate del gestore.

## 2026-09-19 — Copie dei pianificatori senza chiaroveggenza sui collassi

Con `failure_model="reliability"` il sorteggio dei collassi dipende da (seme,
asset, trimestre). GA, memetico e MCTS simulavano su copie con lo stesso seme e
avrebbero quindi conosciuto i collassi futuri reali. `BridgeEnv.planning_clone()`
crea la copia con un sale diverso del flusso dei collassi (dal random globale
seedato per run: riproducibile); l'ambiente reale ha sale 0. I tre punti di
clonazione dei pianificatori la usano. Effetto: solo sulle traiettorie interne
dei pianificatori; nessun effetto su motore, agenti non pianificatori, snapshot.

## 2026-09-20 — Scala nazionale: taratura assoluta, traffico di progetto, mandato finanziabile, prestazioni

Emersi passando da 4-10 ponti sintetici a 150 ponti reali (FHWA NBI Delaware) per 50 anni.
Tutti i parametri sono opt-in, i default riproducono il comportamento storico.

- `capacity_condition_sensitivity` (k) e `reliability_resistance_scale`: capacita' =
  1 - k(1 - condizione) e scala della resistenza. Con k=0.7 e scala 1.1137 il modello
  e' tarato su due ancore (mediana beta annuo 4.2 per rating NBI 7, 3.1 per rating 4),
  Pf media di rete 1.5e-4/anno (prima 8e-2: 18 crolli in 3 anni su 150 ponti).
- `traffic_demand_mode="self_designed"`: ogni ponte e' progettato per il proprio
  traffico; prima la domanda scalava con ADT/ADT mediano di rete e 10 ponti ad alto
  traffico producevano il 95% della Pf di rete.
- `bankruptcy_enabled=False`: il gestore pubblico accumula debito invece di fallire.
- `forced_funding` ("owner_unlimited" | "owner_budget" | "state") e
  `annual_budget_grant_fraction`: mandati a debito, non finanziati (con arretrato e
  sanzioni) o finanziati; trasferimento pubblico annuo in quota del valore del parco.
  Nuove statistiche: forced_unfunded, forced_unfunded_cost, state_funded_cost, grant_total.
- L'obbligo e' assolto se il gestore programma un'azione equivalente o superiore
  (prima l'azione imposta sostituiva sempre quella del gestore).
- `cda_config` nel regime: la legge fissa soglie e intervalli di ispezione per classe.
- Prestazioni: rivalutazione dell'affidabilita' con cadenza configurabile
  (`reliability_refresh_every_quarters`, gli asset toccati sono sempre rivalutati);
  il campionamento dei collassi rivaluta solo l'opera ricostruita; chiave della cache
  delle stime insensibile alla deriva dell'1%/trimestre dell'incertezza; cache delle
  stime e delle proiezioni limitate. Run di 50 anni su 150 ponti: da 16.5 a 5.7 minuti,
  output identici.


## 2026-09-22 — Inventario noto, soglia etica, oracolo (opt-in; default invariati)

Tre parametri, tutti spenti per default: snapshot TRAIN-01 identico, nessun flusso
casuale spostato. Le nuove statistiche compaiono comunque nelle metriche del runner
(a zero), accanto a quelle del regime.

- `initial_inventory_known` (False): il rating dell'inventario nazionale e' un'ispezione.
  Al reset ogni ponte riceve una condizione osservata come dopo un'ispezione L1 al
  trimestre 0 (`_inspect_bridge`: stesso rumore, bias, POD e guadagno di conoscenza;
  classe di attenzione e prossima scadenza calcolate al trimestre 0, ultima ispezione =
  0). Costo zero, nessuna emissione, non e' un'azione del gestore ne' assolve obblighi
  del regime (in `action_history` compare come evento `inventario_iniziale`, stato
  `inventory`). Flusso casuale dedicato sempre a chiave (fonte 4210), anche con
  `rng_streams="shared"`: l'RNG condiviso non viene consumato, quindi il degrado e gli
  altri processi sono identici a quelli senza inventario.
  Prima: le regole che non ispezionano partivano senza alcuna condizione nota e non
  agivano mai (do_nothing = danger_only = worst_first).
- `ethical_floor_enabled` (False), `ethical_floor_condition` (0.2), `ethical_floor_beta`
  (None): "non si puo' lasciar cadere consapevolmente un ponte; al piu', senza cassa, lo
  si chiude". Ogni trimestre, dopo il piano del gestore e le azioni imposte e prima del
  degrado, per ogni ponte con stato NOTO sotto soglia (condizione minima osservata <
  soglia, oppure beta annuo stimato < `ethical_floor_beta`: stato stimato con
  observation_mode="inspected"), senza lavori in corso ne' avviati nel trimestre, dal
  peggiore: se la cassa del gestore copre l'intervento strutturale scelto e la capacita'
  dei lavori (`_crew_admits`) lo ammette, lo si avvia a carico del gestore (conta nella
  spesa del trimestre); altrimenti il ponte e' chiuso (fattore di traffico 0: domanda,
  ricavi e costo di deviazione seguono come per la chiusura sanzionatoria). Scelta
  dell'intervento: il piu' economico il cui guadagno atteso riporta sopra soglia ogni
  componente osservato sotto soglia; in mancanza, il piu' economico che agisce su almeno
  uno di essi; poi il piu' economico in assoluto. Con cassa sufficiente ma squadre
  esaurite si chiude (e si riprova il trimestre dopo). La chiusura resta durante i
  lavori e viene revocata, ripristinando il fattore precedente (salvo sanzione o
  declassamento ancora in vigore), quando lo stato noto torna sopra soglia; nel
  frattempo il gestore non puo' cambiare l'uso dell'opera. Statistiche separate dalle
  sanzioni: `floor_interventions`, `floor_spend`, `floor_closures`,
  `floor_closure_bridge_quarters`, `floor_budget_blocked`, `floor_capacity_blocked`,
  `floor_spend_quarter`, `floor_closed_assets` (in runs.csv come `final_*`, in steps.csv
  come `metric_*`). Lo stato della soglia passa alle copie dei pianificatori.
- `oracle_observation` (False): i dizionari consegnati agli AGENTI portano lo stato vero
  (condizione minima osservata = vera, intervalli osservati per componente collassati
  sulla condizione vera, beta/Pf veri). Trigger di legge, soglia etica e punteggio
  restano su `observation_mode`. Serve come limite superiore del margine di un benchmark.

## 2026-09-22 — Copie dei pianificatori complete, osservazioni staccate dal motore (bug fix)

Nessun effetto con i default: snapshot TRAIN-01 invariato, osservazioni degli agenti
identiche byte per byte (modalita' full, inspected, oracolo; verificato su
DE-OLD-QL-40), righe finali di run identiche per worst_first ed Euristico.

- **Stato perso nelle copie.** `Bridge.from_dict` non ripristinava
  `posting_until_quarter` (declassamento di portata in corso) ne'
  `hydraulic_damage_quarter`. Nelle copie dei pianificatori (`planning_clone`: GA,
  memetico, MCTS, Euristico) e nelle sessioni ricaricate un ponte declassato perdeva
  la scadenza e restava a traffico ridotto per sempre. Ora entrambi i campi passano.
  `BridgeEnv.to_dict/from_dict` portano anche `crew_stats` (azioni rinviate per
  capacita'), `last_expected_loss`, `failure_stream_salt` e i contatori delle
  ispezioni del trimestre corrente (`inspection_counters`, chiave del flusso casuale
  con `rng_streams="keyed"`; quelli dei trimestri passati non servono piu').
  Effetto: cambiano solo le decisioni dei pianificatori con `posting_entry_rate_q > 0`
  su ponti declassati (le simulazioni interne ora vedono la fine del declassamento).
- **Osservazioni vive.** `Component.to_dict` restituiva il `__dict__` del componente e
  `Bridge.to_dict` i contenitori del ponte: un agente che modificava la propria
  osservazione modificava il motore, e le copie dei pianificatori condividevano con
  l'ambiente reale lo storico delle azioni (gli eventi simulati finivano nel
  `action_history` vero). Ora `to_dict` restituisce una copia (round-trip pickle, una
  per ponte). Eccezione per costo: gli alberi `reliability_details` /
  `posterior_reliability_details` sono copiati sui primi due livelli; le foglie piu'
  profonde (risultato del singolo stato limite, punto annuo del profilo) restano
  condivise in sola lettura. Il motore non le modifica mai sul posto (le sostituisce a
  ogni aggiornamento), quindi le copie dei pianificatori sono indipendenti; un agente
  che scrivesse deliberatamente a quella profondita' resta l'unico caso non protetto.
  `_assessed_bridge` non fa piu' un `deepcopy` in aggiunta.
- **Prestazioni** (40 ponti, 200 trimestri, worst_first, osservazione stimata): tempo
  CPU da ~28.5 a ~31.6 s (+11%). Copia profonda completa: +25-30%, scartata.

## 2026-09-22 — Azione imposta non finanziabile: non annulla piu' l'azione del gestore (bug fix, attivo di default)

Prima, con enforcement "forced" e `forced_funding` = `owner_budget` o `matched`,
`_regime_forced_actions` sostituiva l'azione programmata dal gestore con l'azione
dovuta *prima* di sapere se chi paga poteva finanziarla; in step() l'azione era poi
rifiutata per mancanza di fondi (`forced_unfunded`, ma con diagnostica
`forced_by_regime`). Risultato: sul ponte non accadeva nulla, e poiche' e' ammessa una
sola azione imposta per asset anche gli obblighi successivi (tipicamente l'ispezione
periodica, economica) non venivano mai imposti. Con un ripristino dovuto e non pagabile
e un gestore che ogni trimestre programmava l'ispezione dello stesso ponte (Euristico),
il blocco durava decine di anni: in quicklook-crew (N2/N14, replica 1) 153 trimestri sul
ponte 89, rischio 172-195 M contro 16 M con la legge a 10 anni.

Ora, prima di annullare l'azione del gestore, si verifica la capienza di chi paga
(`owner_budget`: costo intero; `matched`: quota del gestore; `owner_unlimited`/`state`:
sempre eseguibile), al netto delle azioni imposte gia' impegnate nel trimestre. Se non
c'e': l'obbligo resta pendente e conta in `forced_unfunded` / `forced_unfunded_cost`
(diagnostica `deferred`/`regime_forced_unfunded`), l'azione del gestore resta, e si
passa all'obbligo dovuto successivo. Il controllo di cassa in step() resta come rete di
sicurezza (la cassa puo' essere consumata da azioni del gestore eseguite prima).

- Parametro legacy `regime_override_unfunded` (False). True riproduce il comportamento
  precedente: verificato che rigenera esattamente runs.csv di quicklook-crew
  (Euristico e worst_first, crew 0, 16 run).
- **Campagne esplorative con leggi "forced" e `forced_funding` owner_budget/matched
  eseguite prima di questo commit sono interessate** (quicklook-crew e ogni campagna
  sulle leggi `*_nosanz`/owner_budget): vanno rieseguite, o riprodotte con
  `regime_override_unfunded: true`. Con il fix, quicklook-crew crew 0: Euristico
  rischio 2.2-2.8 M (replica 0) e 15.7-20.1 M (replica 1) su tutte le leggi, contro
  2.3-7.0 e 14.7-195 M; azioni del gestore annullate da 111-375 a 0-22. worst_first
  quasi invariato.
- Nessun effetto senza regime o con `owner_unlimited`/`state`: snapshot TRAIN-01 invariato.

## 2026-09-22 — Profilo di costo `fhwa_unit_costs_2025_b` (opt-in)

Nel catalogo base `replace_appoggi` costa 100000 EUR a intervento, indipendentemente dalla
superficie: e' l'unica azione sui ponti con costo fisso. Tutte le `replace_*` hanno lo stesso
guadagno totale, quindi sopra circa 33 m2 la sostituzione degli appoggi costa meno di quella
dell'impalcato a parita' di guadagno, e ogni regola che sceglie sul guadagno per euro (euristico,
worst_first, soglia etica) la preferisce quasi ovunque (0.1 M contro 175 M sul ponte da 58000 m2
del campione Delaware).

Nuovo profilo `fhwa_unit_costs_2025_b` = `fhwa_unit_costs_2025` + `replace_appoggi` a 190 EUR/m2
(conserva 0.1 M sul ponte mediano da 520 m2; valore provvisorio, da sostituire con un costo unitario
di fonte) e guadagni laterali ridotti da 0.15 a 0.05. Nessun effetto sui profili esistenti; snapshot
TRAIN-01 invariato.

## 2026-09-23 — Pianificatori sulla conoscenza del gestore, obiettivo allineato al benchmark (opt-in)

Prima: GA, memetico e MCTS simulavano su `planning_clone()`, copia completa del motore con
lo stato VERO, anche con `observation_mode="inspected"`: i pianificatori erano sempre
oracoli (MCTS dava risultati identici con `oracle_observation` False/True). Con
`rng_streams="keyed"` la copia aveva inoltre lo stesso seme del motore e rivedeva i
flussi futuri reali di degrado, ispezioni, esiti degli interventi, eventi idraulici e
declassamenti (solo i collassi avevano un sale proprio). MCTS e GA ottimizzavano budget +
valore patrimoniale, non il punteggio del benchmark. Tre parametri, default storici:
snapshot TRAIN-01 identico, flussi casuali del motore reale invariati.

- `planner_information` ("truth"): "belief" fa costruire la copia del pianificatore
  (`BridgeEnv.belief_clone`, chiamata da `planning_clone`) dalla conoscenza del gestore.
  Con `observation_mode="inspected"` e senza oracolo, per ogni componente lo stato
  nascosto (`BELIEF_HIDDEN_COMPONENT_FIELDS`: `current_condition`, `accumulated_damage`,
  `effective_age_years`, i parametri latenti `degradation_scale`,
  `degradation_shape_factor`, `traffic_sensitivity`, `environment_sensitivity`,
  `resistance_bias`, `demand_bias`, `crack_bias`, e `last_process_multiplier`) e'
  riestratto a ogni copia: parametri latenti dalla priori del profilo (stessa legge del
  costruttore, ora `Component.draw_latent_parameters`, stesso ordine di estrazione);
  componente osservato: condizione ~ N(centro dell'intervallo, ((max-min)/3.92)^2)
  troncata a [1e-6, 1] (centro come in `_assessed_bridge`, intervallo letto come 95%),
  eta' efficace invertita con i parametri estratti; componente mai osservato: eta'
  nominale (iniziale + tempo trascorso) come in `_assessed_bridge`. Restano noti:
  profilo, vita utile, eta' iniziale, intervalli osservati, incertezza epistemica, esiti
  delle prove di carico, tutto lo stato di gestione (budget, lavori, restrizioni,
  regime, soglia). A livello di ponte sono nascosti `hydraulic_damage_quarter` (0) e gli
  eventi `danno_idraulico` dello storico; beta, Pf, fragilita', indice di rischio,
  proiezioni e curve di monitoraggio sono ricalcolati sullo stato estratto;
  `last_expected_loss` (calcolata sul vero) e' azzerata. Con `oracle_observation=True`
  (oracolo = pianificatore sul vero) o `observation_mode="full"` lo stato resta vero.
  In ogni caso, con "belief" il futuro casuale della copia e' indipendente da quello
  reale: sale proprio dei flussi a chiave (`stream_salt`, 0 nel motore reale = chiave
  storica), generatore condiviso nuovo, sale dei collassi. Tutte le estrazioni vengono
  dal generatore del pianificatore (default: dal random globale seedato per run), mai
  dai flussi del motore. Il GA salta, in modalita' "belief", la sua sostituzione storica
  delle condizioni (cc6e12c: centro osservato o componente nominale, con parametri
  latenti ed eta' invertita sui valori veri).
- `planner_objective` ("asset_value"): "risk_adjusted" fa valere a un rollout
  -(somma sui trimestri simulati di spesa del gestore + perdita attesa da collasso del
  trimestre + `planner_safety_weight` x deficit di sicurezza), con spesa = costi
  restituiti da `step` (come `spent_quarter`), perdita attesa = `last_expected_loss`
  (come `metric_risk_cost_quarter`), deficit = somma sui ponti di
  max(0, target_beta_annual - beta annuo posterior) sullo stato dopo il passo (stesse
  grandezze di `final_min_beta_margin`, senza clamp). MCTS e GA sull'orizzonte del
  rollout/cromosoma; il memetico, con "risk_adjusted", simula l'intero cromosoma invece
  del punteggio composito a un passo. Override per agente: `objective`,
  `safety_weight` in `mcts_params` / `ga_params`.
- `planner_safety_weight` (0.0): lambda, in EUR per unita' di beta x ponte x trimestre;
  variandolo lo stesso pianificatore traccia un fronte sicurezza x costo.

## 2026-09-23 — Danno locale improvviso: idraulico, urto, natante, fatica, tipi utente (opt-in; default invariati)

Perche'. Con il solo degrado lento e monotono la cadenza ispettiva non puo' cambiare gli
esiti: fra due ispezioni non compare nulla di nuovo. I crolli reali sono dominati da
cause improvvise: idraulica 54.9% del campione nazionale USA e 80.5% dei 246 crolli
italiani 2000-2023, urto 12.9%, sovraccarico 11.3%, degrado 8.8% (Wardhana e Hadipriono
2003, doi 10.1061/(ASCE)0887-3828(2003)17:3(144): piene 33%, erosione 16%, urto 12%;
tasso di crollo per tutte le cause 2.13e-4 per ponte-anno, Cook, Barr e Halling 2015,
doi 10.1061/(ASCE)CF.1943-5509.0000571; fonti raccolte in
papers/infralab-benchmark-aei-2026/positioning/validation-engineering.md e
validation-empirical.md).

Modulo `infralab/core/hazards/sudden_damage.py`. Per ogni tipo attivo, ogni trimestre e
ogni ponte, un sorteggio da un flusso SEMPRE a chiave (seme, tipo, ponte, trimestre;
scopi 4221-4224, tipi utente 4300+, osservazioni 4240-4241) decide l'evento con un tasso
che dipende dall'esposizione. L'evento riduce la condizione VERA dei bersagli di una
perdita estratta dalla distribuzione del tipo; l'incertezza epistemica (conoscenza del
gestore) non cambia. Il danno e' invisibile a gestore e legge (osservazione, classe,
storico azioni, trigger del regime, soglia etica) finche' un'ispezione non rileva il
componente (stessa POD del degrado ordinario) o un monitoraggio attivo non lo rivela;
una frazione autodenunciante e' nota subito, con un'osservazione di rumore L1 senza POD,
riclassificazione e prossima ispezione mai posticipata. Un intervento sul componente
ripara anche il danno non scoperto; al collasso i danni ancora nascosti sono contati.
Il danno accelera la perdita di affidabilita' attraverso il legame condizione->capacita'
esistente; il collasso resta campionato dalla Pf. Le copie "belief" dei pianificatori
non vedono i danni nascosti.

Numeri casuali comuni: accendere un tipo non consuma l'RNG condiviso (verificato con
rng_streams "shared" e "keyed") e non sposta gli eventi degli altri tipi; due strategie
sullo stesso seme affrontano gli stessi eventi.

Parametri (default; PROVISIONAL = nessuna fonte nei documenti di validazione):
- Idraulico: `hydraulic_legacy_mode` (True = canale storico invariato: RNG del motore con
  rng_streams="shared", perdita fissa su pile e spalle, evento `danno_idraulico` scritto
  subito nello storico). False = stessa esposizione (item NBI 071 sull'acqua, tasso piu'
  alto con item 113 <= 3) e stessi tassi (`hydraulic_collapse_rate_*_q`,
  `hydraulic_damage_rate_multiplier`), ma flusso dedicato e danno del canale improvviso:
  `hydraulic_damage_targets` ("pile,spalle"), `hydraulic_damage_hits` ("all"),
  `hydraulic_damage_drop_dist` ("fixed"; media = `hydraulic_damage_condition_drop` 0.20),
  `hydraulic_damage_drop_cov` (0.5, PROVISIONAL), `hydraulic_damage_announce_prob`
  (0.0, PROVISIONAL: erosione sott'acqua).
- Urto di veicolo: `impact_hazard_enabled` (False); `impact_damage_rate_q` (6.9e-5 =
  12.9% x 2.13e-4 / 4 crolli per urto a trimestre x rapporto danno/crollo 10 del canale
  idraulico, PROVISIONAL); esposizione x (ADT/`traffic_reference_adt`)^`impact_adt_exponent`
  (1.0, PROVISIONAL) x (1 + `impact_truck_weight` (1.0, PROVISIONAL) x quota mezzi
  pesanti, item 109) x `impact_road_under_multiplier` (1.0, PROVISIONAL) se sotto passa
  una strada (item 042B codici 1, 4, 6, 8); bersagli "impalcato,pile", hits "one",
  lognormale media 0.15 CoV 0.5 (PROVISIONAL), autodenuncia 0.5 (PROVISIONAL).
- Urto di natante: `vessel_impact_enabled` (False), solo ponti con `navigable_waterway`
  (item NBI 038 = 1, ora importato da `infralab/data/nbi.py`); `vessel_impact_rate_q`
  1e-4 (PROVISIONAL); bersaglio "pile", lognormale 0.25 CoV 0.5, autodenuncia 0.8
  (PROVISIONAL).
- Fatica/frattura: `fatigue_hazard_enabled` (False); `fatigue_damage_rate_q` 4.7e-5
  (PROVISIONAL, limite superiore: intera quota "degrado" 8.8% x 2.13e-4 / 4 x 10);
  x (ADT/rif)^`fatigue_adt_exponent` (0.0) x (1 + `fatigue_truck_weight` (2.0) x quota
  mezzi pesanti); bersagli "impalcato,appoggi", lognormale 0.10 CoV 0.5, autodenuncia 0.0
  (PROVISIONAL).
- Tipi utente: `sudden_custom_events` ([]), lista di {name, rate_q, targets, hits,
  drop_dist, drop_mean, drop_cov, announce_prob, requires, multipliers, adt_exponent,
  truck_weight}; `requires` e `multipliers` leggono attributi del ponte (crosses_water,
  scour_critical, navigable_waterway, road_under, ...).
- `sudden_damage_monitoring_detect_prob` (0.0: il monitoraggio non vede i danni
  improvvisi, come prima); `sudden_damage_reveal_sigma` (0.10, come L1).
- Distribuzioni: "fixed", "uniform" (media +- sqrt(3) CoV media), "lognormal" (media e
  CoV); perdita troncata a [0,1].

Uscite (solo con il canale acceso; con i default le colonne del laboratorio non
cambiano): `metric_sudden_*` per trimestre e `final_sudden_*` per run: events_total,
events_quarter, events_<tipo>, announced, detected_by_other_announcement,
detected_by_inspection, detected_by_monitoring, repaired_before_detection,
undetected_at_failure, failures_with_undetected_damage, condition_loss_total,
pending_undetected. Esposti in `/api/config` (tre gruppi, metadati in
`sudden_damage.UI_FIELDS`, form generico del frontend con i nuovi tipi "text" e "json"),
nella UI Streamlit (riquadro avanzato del setup) e nelle spec del laboratorio
(`params.<nome>` in griglia).

Con i default nessun flusso nuovo e' letto, `to_dict` e le metriche non cambiano:
snapshot TRAIN-01 identico.
## 2026-09-23 — Sandbox delle policy per l'integrita' del benchmark (opt-in; nessun effetto sulla dinamica)

Nuovo parametro `policy_sandbox` (default `False` = storico: le policy ricevono il
motore in `decide(env)`, `AssetEnvGym` mette `info["env"]`; snapshot TRAIN-01 identico).
Nessun cambiamento numerico al motore: il sandbox agisce solo sull'interfaccia agenti
(`infralab/agents/sandbox.py`) e cambia cio' che una policy puo' leggere, non la fisica.
L'unico effetto collaterale con il sandbox attivo e' che la costruzione del handle
estrae un seme dal `random` globale (seedato per run): i pianificatori che usano lo
stesso `random` ricevono quindi sequenze diverse rispetto alla run senza sandbox.

Con `policy_sandbox=True`:
- Nessuna policy riceve il motore. `LegacyAgentAdapter` (lato harness) passa a
  `decide` un `PlanningHandle` nuovo a ogni decisione, costruito senza riferimenti al
  motore: `get_agent_observations()` (solo del proprietario), `params` (sola lettura,
  senza `seed`), `maint_ops` (copia), `rules()`, `feasible_actions()`,
  `planning_clone()`/`belief_clone()`. `AssetEnvGym` non mette `env` in `info` e
  restituisce l'osservazione sanitizzata; il runner passa il motore all'adapter con
  `gym.harness_env()`. Anche UI Streamlit (`run_turn`, cruscotto del giocatore) e API
  (`_asset_view`, `_asset_detail`, KPI di sessione, copilota) mostrano la vista del gestore.
- Osservazione sanitizzata (`sanitized_observation`): copia JSON pura; `game_settings`
  senza `seed`; `maintenance_options` copiato; `game_rules.my_emissions_tco2` (contabilita'
  propria); blocco `information` con la modalita' dichiarata. In `observation_mode=
  "full"` lo stato vero resta visibile per dichiarazione, con `oracle_observation=True`
  per concessione esplicita. In "inspected" senza oracolo, oltre alla vista stimata del
  motore: `hydraulic_damage_quarter` ed eventi `danno_idraulico` compaiono solo dopo
  un'osservazione del ponte in un trimestre successivo (ispezione/monitoraggio/prova di
  carico, rilievo post-intervento, ricostruzione); il referto della prova di carico
  conserva l'esito osservato (quarter, alpha, passi, passata/scartata) e riporta beta/Pf
  prior e posterior dello stato STIMATO (`basis: "assessed"`), senza probabilita' di
  rottura/scarto e beta calcolati sul vero; `future_curve_range` e' proiettata dallo
  stato stimato (parametri medi del profilo); dai componenti sono tolti
  `accumulated_damage`, `last_process_multiplier`, `vita_utile_anni` (campionata),
  `last_proof_load_risk` e i parametri latenti.
- Copie dei pianificatori: sempre il mondo del gestore (`belief_clone`), a prescindere da
  `planner_information`, generate da una fotografia gia' ripulita (`belief_snapshot`):
  seme sostituito, `agent_config` ridotto al tipo, in "inspected" ponti stimati e, per
  gli asset di altri gestori, credenza da solo inventario (evidenza privata tolta),
  danni sismici per ponte e perdita attesa (vere) tolti. Futuro casuale proprio
  (seme, sale dei flussi a chiave, generatore condiviso e sale dei collassi nuovi). Nota:
  lo stimatore MC di affidabilita' della copia e' seedato dal seme della copia, quindi
  beta/Pf della copia differiscono dall'osservazione per rumore numerico.
- Tool agentici (`state`, `feasible_actions`, `simulate`) e osservazione dell'agente
  Custom costruiti dall'osservazione sanitizzata; `simulate` gira su `planning_clone()`.
- Diagnostica: ogni accesso non esposto dal handle solleva `SandboxViolation`, registrata
  in `LegacyAgentAdapter.sandbox_violations`; se la violazione esce da `decide`, la
  decisione del trimestre e' "nessuna azione".

Limiti noti (non chiusi, documentati): isolamento in-process (codice Python arbitrario
puo' risalire agli oggetti del harness via gc/frame: per sottomissioni non fidate
eseguire la policy fuori processo sull'osservazione JSON); segnali derivati dal vero nel
motore stesso, che il sandbox non ricalcola: `epistemic_uncertainty` cresce del 10% a un
danno (idraulico, sismico, prova fallita), l'`alpha` della prova di carico e' scelto sullo
stato vero, la stima e la copia "belief" usano la vita utile campionata per i componenti
mai osservati, la classificazione di un regime con tabella propria legge il beta del
ponte vero (`_classifier_values`), e metriche/reward di `AssetEnvGym` sono calcolate sul
vero (punteggio del harness: non vanno inoltrate alla policy).

## 2026-09-23 — Valore terminale dell'obiettivo dei pianificatori (opt-in)

Con `planner_objective="risk_adjusted"` e un rollout di pochi trimestri il costo di un
intervento cade dentro l'orizzonte e il rischio evitato cade fuori: l'obiettivo premiava
il non fare nulla. `planner_terminal_quarters` (int, default 0 = storico) aggiunge a fine
rollout N trimestri di perdita attesa + lambda x deficit di sicurezza calcolati sullo stato
finale tenuto fermo (stima per difetto: nessun degrado ulteriore). Vale per GA, memetico e
MCTS. Esposto in `/api/config` (gruppo Benchmark, con `planner_objective` e
`planner_safety_weight`) e come `params.planner_terminal_quarters` nelle spec. Default
invariati, snapshot TRAIN-01 identico.

## 2026-09-23 — Informazione stretta: chiusi i segnali dal vero nel motore (opt-in)

Il red team metodologico (RT001, FATALE) ha rilevato che, anche con `policy_sandbox=True`
e `observation_mode="inspected"`, quattro segnali derivati dallo stato vero arrivavano
alla vista del gestore (limiti gia' elencati nella voce del sandbox). Nuovo parametro
`strict_information` (bool, default `False` = storico; snapshot TRAIN-01 identico) che,
se `True`, li chiude tutti:

1. **Incertezza epistemica solo da evidenza.** Il danno non osservato (idraulico legacy,
   sismico; il danno improvviso gia' non la toccava) non aumenta piu' l'incertezza
   (`Component.apply_damage(..., raise_uncertainty=False)` via `_apply_hidden_damage`;
   `apply_earthquake_damage(..., raise_uncertainty=False)`). L'incertezza cresce (x1.10,
   tetto 0.40) quando un'osservazione mostra un calo del centro dell'intervallo osservato,
   rispetto all'osservazione precedente dello stesso componente, maggiore di
   `strict_uncertainty_drop` (nuovo, default 0.10; il centro precedente e' salvato in
   `strict_last_observed_centre`). La prova di carico fallita (esito osservato) la aumenta
   come prima. Chiuso anche un quinto canale trovato nell'audit: con la POD attiva il
   guadagno di conoscenza dell'ispezione dipendeva dall'esito vero della detection;
   con `strict_information` si applica sempre (il gestore non sa se ha rilevato).
2. **Prova di carico pianificata sulla stima.** Alpha, numero di passi e componenti da
   provare (prima scelti sulle sensibilita' al traffico latenti) sono decisi sullo stato
   stimato (`_assessed_bridge`); rischio di rottura/scarto, sorteggio degli esiti e
   posteriori restano sul ponte vero. Vale se il gestore non vede il vero (non in
   "full" ne' con l'oracolo). `referto.planned_on="assessed"`.
3. **Vita utile nominale per la stima.** Alla creazione degli asset si salva
   `nominal_service_life` (vita dell'inventario/tipo senza la perturbazione campionata:
   esplicita, vita del tipo, o vita media dello scenario), solo con il flag attivo. La
   stima del gestore e la copia "belief" (componenti mai osservati e inversione
   condizione->eta' efficace di quelli osservati) usano quella; la stima azzera anche
   `last_process_multiplier` (moltiplicatore vero).
4. **Classificazione del regime sulla stima.** Con `observation_mode="inspected"` la
   tabella di classificazione di un regime legge beta annuo ed eta' dallo stato stimato,
   come gia' i trigger degli obblighi (anche con l'oracolo: la legge non e' l'oracolo).

La soglia etica e i trigger degli obblighi leggevano gia' la stima: ora la stima stessa
non contiene piu' segnali dal vero. La documentazione del sandbox raccomanda
`strict_information=True` per ogni run di benchmark (non forzato).

Diagnostica: `infralab/lab/leak_check.py` (`python -m infralab.lab.leak_check --scenario
ID --param k=v ... [--agents N] [--json out]`, uscita 0 pulito / 1 fuga / 2 errore;
API `run_leak_check`). Per N trimestri di una traiettoria con ispezioni a rotazione e
prove di carico confronta il mondo reale con una copia a stessa evidenza e verita'
perturbata (parametri latenti, vita campionata, condizione vera, un danno idraulico
nascosto per asset, numeri veri del referto di prova) su: osservazione consegnata alla
policy, classificazione del regime, trigger degli obblighi e della soglia etica, piano
della prova di carico (anche in variante a griglia), copia belief a seme fisso.
Risultati (inspected + sandbox, POD, sisma annuo, fatica, soglia etica, it_lg2020_forced):
con `strict_information=False` divergono osservazione, copia belief, classificazione e
piano di prova; con `True` nessuna divergenza su EXP-A-BUDGET, NBI-DE (anche 2 gestori),
PAPERA-SHORT-A, BLDG-DEMO, WIND-DEMO, WALL-DEMO, TRAIN-01. Senza sandbox resta solo il
danno idraulico non ancora rilevato nello storico del motore (lo chiude il sandbox).
Costo: trascurabile (run di 20 trimestri: 3.6 -> 3.9 s EXP-A-BUDGET, 7.6 -> 7.1 s NBI-DE).

Non chiusi (documentati): la probabilita' dei declassamenti di portata
(`posting_entry_rate_q`) e dei collassi dipende dalla condizione vera: l'esito e' un
evento pubblico (evidenza legittima, non una fuga dell'osservazione) ma la sua
frequenza e' informativa sul vero; metriche/reward del harness restano sul vero.

Esposto in `/api/config` (gruppo Benchmark: `strict_information`,
`strict_uncertainty_drop`), UI Streamlit (riquadro "Informazione e integrita' del
benchmark") e spec del laboratorio (`params.strict_information`,
`params.strict_uncertainty_drop`).

## 2026-09-23 — Danno improvviso nascosto fuori dalle copie dei pianificatori; leak_check con danni improvvisi e a traiettoria

Red team metodologico (RT009-RT011). Nessun effetto sulla dinamica (snapshot TRAIN-01
identico): cambia solo cio' che le copie "belief" dei pianificatori contengono, e la
diagnostica.

**RT009 (fuga dimostrata, chiusa).** `belief_snapshot` (sandbox: la fotografia da cui il
`PlanningHandle` genera le copie di GA/MCTS/memetico/Euristico, conservata nel handle
della policy) partiva da `env.to_dict()` e non toglieva `sudden_pending` (lista vera dei
danni improvvisi non rilevati: componente, trimestre, perdita) ne' `sudden_stats`
(contatori veri del canale: eventi avvenuti per tipo, perdita di condizione totale,
danni riparati prima della scoperta). In "inspected" senza oracolo ora li toglie
entrambi (la copia li reinizializza come un motore nuovo). `BridgeEnv.belief_clone`
svuotava gia' `sudden_pending` e riestrae dall'evidenza la condizione vera (quindi anche
la perdita del danno nascosto: `current_condition`, `effective_age_years`,
`accumulated_damage` sono tra i campi riestratti), ma copiava `sudden_stats`: ora li
azzera. Con oracolo o "full" nulla cambia (verita' concessa). Altre vie verificate: la
sanitizzazione dell'osservazione non contiene nulla del canale (l'evento
"danno_improvviso" entra nello storico solo quando il danno e' reso noto); `to_dict` /
`from_dict` restano complete (sono del harness). Prima della correzione il leak_check
esteso segnalava `planner_snapshot.sudden_pending` e `belief_clone.sudden_stats` su
NBI-DE; dopo, nessuna divergenza.

**RT010.** `perturb_hidden_truth(..., sudden=True)` (default) perturba anche il canale
del danno improvviso, acceso o no (`perturb_hidden_sudden`): per asset un evento
nascosto in piu' (tipo attivo, componente bersaglio del tipo; tipo "leak_check" se il
canale e' spento), i pendenti con intensita' diversa o tolti, contatori veri aggiornati;
intervalli osservati e incertezza invariati. Nuova uscita confrontata
`planner_snapshot` (con il sandbox). CLI: `--sudden-hazards` accende una
configurazione di prova (`TEST_HAZARD_PARAMS`: idraulico non storico, urto, natante,
fatica a tassi alti, rilevazione da monitoraggio 0.5; i `--param` espliciti
prevalgono); `--no-sudden-injection` toglie l'iniezione.

**RT011.** Modalita' traiettoria (`--trajectory`, `leak_check_trajectory`): la verita' e'
perturbata una volta, poi i due mondi avanzano con lo stesso piano, ciascuno con la
propria dinamica (degrado con i propri parametri latenti, danni, riparazioni; con
`sudden` altri danni improvvisi nascosti in B a ogni trimestre). L'evidenza e' tenuta
identica cosi': ogni lettura d'ispezione di B (ispezioni, rilievi post-intervento, prove,
autodenunce, monitoraggio; esito POD compreso) e' quella di A (`step_recording` /
`step_replaying`, aggancio a `inspect_component`); le etichette di danno nello storico
("danno_idraulico" rivelato, "danno_improvviso" rilevato) sono quelle di A; il ricavo del
trimestre e' quello di A. Tutto il resto dello stato del gestore (incertezza, classi,
scadenze, regime, soglia etica, budget, diagnostica, cache) evolve in B senza essere
toccato, cosi' una fuga che si accumula nel tempo emerge (test: una fuga sintetica che
scrive la verita' nell'incertezza a ogni passo e' vista dalla traiettoria e non dal
controllo istantaneo). L'evidenza che dipende legittimamente dal vero e non si puo'
forzare (collassi, gestori attivi, declassamenti, esito delle prove di carico) e le
letture non appaiate sono controllate dopo ogni trimestre: alla prima differenza il
confronto si ferma (`evidence_divergence`). `--policy inspect-only` esclude le prove di
carico (il loro piano resta confrontato a ogni trimestre). Il rapporto riporta
`truth_gap` (scarto della condizione vera e danni pendenti A/B per trimestre: il
controllo non e' vuoto). Limiti: e' un controfattuale "stessa evidenza" (le letture di
B sono quelle di A anche dove la verita' di B le avrebbe cambiate), copre i trimestri
prima della prima divergenza legittima; i processi pubblici su RNG condiviso possono
separare presto le storie (`rng_streams="keyed"` le allinea). Con il modello di
collasso storico (`failure_model="condition"`: un ponte sotto `failure_threshold`
elimina il gestore) la perturbazione di B non porta la condizione vera sotto
`failure_threshold + 0.20` (`min_condition`), altrimenti su parchi invecchiati B
collasserebbe al primo passo e il confronto si fermerebbe subito.

Canale documentato trovato dalla traiettoria (non chiuso, evidenza tenuta fissa): il
ricavo trimestrale dipende dall'indice di salute VERO (`_get_revenue_multiplier`, sotto
0.5 con `enforce_guidelines`): il budget del gestore porta un segnale del vero. Il
rapporto lo mostra in `revenue_truth_dependence`.

## 2026-09-23 — Fonte della salute nei ricavi: `revenue_health_source` (opt-in)

Il red team di infralab-regbench (RT012, RT013) ha trovato che i ricavi trimestrali
(`_get_revenue_multiplier`) si riducono con l'indice di salute VERO quando
`enforce_guidelines` e' attivo, e con lo stato vero anche negli adattatori di produzione
per tipo di asset. Il budget che arriva a ogni policy (`game_rules.my_budget`) portava
quindi un segnale dal vero in ogni modalita' di osservazione (fino a circa 100k EUR a
trimestre su NBI-DE in modalita' traiettoria).

`revenue_health_source` ("truth" = storico; "assessed" = stato stimato dal gestore con
observation_mode="inspected"; "none" = ricavi indipendenti dalla condizione; restano la
limitazione del traffico e i lavori in corso, che sono pubblici). `leak_check` ora FALLISCE
se il ricavo dipende dalla salute vera (prima lo annotava soltanto). Esposto in
`/api/config` (gruppo Benchmark), nel blocco benchmark della UI Streamlit e come
`params.revenue_health_source`. Default invariato: snapshot TRAIN-01 identico.

## 2026-10-04 — Modello del costo utenti delle deviazioni: opzioni `detour_*` (opt-in)

La campagna confermativa B di infralab-regbench (T2 non superato) ha mostrato che il costo
utenti delle deviazioni pesava circa il 95% dell'asse costo. Tre difetti del modello:
(1) il codice NBI 199 dell'item 019 significa "nessuna deviazione possibile" ma era letto come
199 km: un solo ponte chiuso con ADT 1.900 costava circa 15 M EUR a trimestre; (2) una
limitazione parziale (portata, senso unico) deviava la quota (1 - disponibilita') di TUTTO il
traffico, mentre una limitazione di portata ferma solo i carichi pesanti; (3) i lavori con
riduzione di servizio contavano come deviazione.

Nuovi parametri (default = storico): `detour_km_cap` (0 = nessun tetto), `detour_partial_diversion`
("all" storico | "trucks": con disponibilita' parziale deviano solo i mezzi pesanti, al costo
unitario dei mezzi pesanti; la chiusura vera devia tutti), `detour_works_diversion` (True storico |
False: solo la chiusura vera devia). Costi unitari invariati (CE Delft 2019). Esposti in
`/api/config` (gruppo Benchmark), UI Streamlit e `params.<nome>`. Snapshot TRAIN-01 identico.
## 2026-10-04 — Eventi estremi regionali: shock correlati nello spazio (opt-in; default invariati)

Nuovo canale `infralab/core/hazards/regional_events.py`. Il canale del danno locale
improvviso sorteggia gli eventi ponte per ponte in modo indipendente; mancavano gli
eventi rari che colpiscono molti ponti nello stesso trimestre (piena di bacino,
tempesta) e mettono sotto pressione insieme budget, squadre e soglia etica.

Dinamica. Numero di eventi per trimestre ~ Poisson(`regional_event_rate_y`/4) da un
flusso a chiave dedicato (seme, trimestre). Centro: un ponte a caso fra quelli con
coordinate (`regional_event_center_mode="random_bridge"`) o un punto fisso
(`"fixed"`, `regional_event_center_lat/lon`). Area: ponti ESPOSTI entro
`regional_event_radius_km` (haversine). Intensita' locale I(d): `exponential`
exp(-d/decay), `gaussian` exp(-(d/decay)^2) o `uniform` (`regional_event_decay_model`,
`regional_event_decay_km`). Per ogni ponte dell'area, da un flusso a chiave (seme,
evento, ponte, trimestre): danno con probabilita' min(1, `regional_event_hit_prob` x I x
moltiplicatori di esposizione); perdita di condizione VERA dei componenti bersaglio
(`regional_event_targets`, `regional_event_hits`) estratta come nel canale improvviso
(`regional_event_drop_dist/mean/cov`) e moltiplicata per I se
`regional_event_scale_damage`; autodenuncia con `regional_event_announce_prob`.
Esposizione: `regional_event_requires` (attributi che devono essere veri, es.
`crosses_water`; vuoto = tutti) e `regional_event_multipliers` ({attributo: fattore}).
Il collasso resta campionato dalla Pf (nessun aumento diretto).

| parametro | default | stato |
|---|---|---|
| `regional_events_enabled` | False | opt-in |
| `regional_event_rate_y` | 0.05 (uno ogni 20 anni sul parco) | PROVISIONAL |
| `regional_event_center_mode` | random_bridge | scelta di modello |
| `regional_event_radius_km` | 25 | PROVISIONAL |
| `regional_event_decay_model` / `_decay_km` | exponential / 15 | PROVISIONAL |
| `regional_event_requires` | crosses_water | default "piena di bacino" |
| `regional_event_multipliers` | {scour_critical: 3} | PROVISIONAL |
| `regional_event_hit_prob` | 0.3 al centro | PROVISIONAL |
| `regional_event_targets` / `_hits` | pile,spalle / all | default "piena" |
| `regional_event_drop_dist/mean/cov` | lognormal / 0.15 / 0.5 | PROVISIONAL |
| `regional_event_scale_damage` | True | scelta di modello |
| `regional_event_announce_prob` | 0.2 | PROVISIONAL |

Nessun valore e' tarato: servono serie di eventi regionali con danni per ponte (es.
rapporti post-evento delle regioni o dei DOT) per fissarli.

Informazione. L'EVENTO e' pubblico: nell'osservazione (`regional_events`: trimestre,
centro, raggio, decadimento, ponti esposti nell'area) e per ponte
(`regional_event_uninspected`, `regional_event_quarter`, `regional_event_intensity`:
area di un evento non ancora seguito da un'ispezione; solo geografia e date). Il danno
per ponte e' nascosto finche' un'ispezione (POD) o il monitoraggio
(`sudden_damage_monitoring_detect_prob`) non lo rileva, salvo la quota autodenunciante.
I danni pendenti stanno nella stessa lista del canale improvviso (`sudden_pending`, tipo
"regional"): stesse chiusure in `belief_clone` (svuotata) e nella fotografia del sandbox
(tolta). I contatori veri del canale (`regional_stats`) sono tolti dalla fotografia del
sandbox e azzerati in `belief_clone`; l'elenco pubblico degli eventi resta. Il canale
improvviso conta ora solo i propri danni (anche `sudden_pending_undetected`); il nome
"regional" e' riservato fra i tipi utente.

Legge. Nuovo trigger di regime `regional_event_in_area`: vero per un ponte esposto
nell'area di un evento non ancora seguito da un'ispezione (`last_inspection_quarter`;
un'ispezione nello stesso trimestre precede l'evento). `threshold` = intensita' locale
minima (None = tutta l'area). Con `grace_quarters` una legge impone l'ispezione
straordinaria entro N trimestri (Linee Guida 2020: ispezioni speciali dopo eventi
eccezionali). Limite: le ispezioni di sorveglianza con `resets_cadence=False` non
aggiornano `last_inspection_quarter` e quindi non assolvono l'obbligo.

Numeri casuali comuni. Flussi a chiave dedicati (scopi 200101, 200102, fuori
dall'intervallo dei tipi utente del canale improvviso), indipendenti dallo stato: stesso
seme -> stessi eventi, centri e sorteggi per ponte per ogni strategia. Accendere il
canale non sposta l'RNG condiviso ne' gli altri flussi: con eventi ma
`regional_event_hit_prob=0` la traiettoria TRAIN-01 coincide con lo snapshot dorato
(test).

Uscite (solo a canale acceso): `metric_regional_*` per trimestre e `final_regional_*`
per run: `events_total`, `events_quarter`, `bridges_in_area`, `bridges_hit`,
`bridges_hit_quarter`, `components_hit`, `announced`, `detected_by_inspection`,
`detected_by_monitoring`, `detected_by_other_announcement`,
`repaired_before_detection`, `undetected_at_failure`, `condition_loss_total`,
`pending_undetected` (i rilevati/nascosti contano componenti danneggiati).

leak_check: `perturb_hidden_regional` (danni regionali nascosti diversi nell'area degli
eventi pubblici, contatori veri variati; anche a canale spento) nel mondo B, istantaneo
e a traiettoria; `--regional-events` accende una configurazione di prova
(`TEST_REGIONAL_PARAMS`); l'elenco pubblico degli eventi entra nell'impronta
dell'evidenza; "danno_regionale" (rilevazione da ispezione) e' un'etichetta di danno
come "danno_improvviso". TRAIN-01 con sandbox e informazione stretta: pulito in entrambe
le modalita'.

Interfacce: `UI_FIELDS` del modulo riusati da `/api/config` (gruppo "Eventi estremi
regionali") e dalla UI Streamlit (riquadro avanzato nel setup); nelle specifiche di
laboratorio `params.<nome>`. Costo di calcolo trascurabile (stessa durata per
trimestre a canale acceso e spento sul parco di 30 ponti). Default invariati: snapshot
TRAIN-01 identico.

## 2026-10-06 — Eventi progressivi (latenti), registro dei collassi, fine partita al collasso (opt-in; default invariati)

Motivo. Il crollo idraulico diretto e' istantaneo e identico per ogni strategia: nessun
gestore puo' evitarlo (86% dei collassi delle strategie migliori nell'ultima campagna,
contro circa 55% nei dati reali). Serve un meccanismo in cui il fenomeno si innesca, si
sviluppa nel tempo e un'ispezione nella finestra permette di fermarlo. Modulo
`infralab/core/hazards/progressive_events.py`, struttura delay-time (arrivo del difetto,
ritardo casuale, conseguenza se non trattato, ispezione imperfetta nel ritardo).

Fonti e provenienza dei numeri: nota di letteratura
`papers/infralab-regbench/experiments/v3-design/letteratura-eventi-progressivi.md`
(tabella delle ancore A1-A35 e tabella dei parametri proposti). Come nella nota, ogni
default e' marcato FROM SOURCE (letto o derivato da una fonte) o PROVISIONAL (scelta di
modello senza numero in letteratura, da sottoporre a sensibilita').

Meccanismo (tipi: `hydraulic`, `geotechnical`, tipi utente `progressive_custom_events`).
1. Innesco: per tipo e trimestre un flusso a chiave dedicato estrae un numero per ponte;
   innesco se minore del tasso del ponte, che dipende solo dall'inventario. Idraulico:
   ponti su acqua, con la stessa esposizione del crollo diretto. Geotecnico: attributo
   `on_unstable_slope` dello scenario o, se assente, quota
   `progressive_geotechnical_exposed_fraction` assegnata alla costruzione con un flusso a
   chiave (default 0 = nessuno). Facoltativo: un ponte colpito da un evento regionale
   innesca un difetto idraulico con probabilita'
   `progressive_hydraulic_regional_trigger_prob` (default 0; sono inneschi IN PIU'
   rispetto alla taratura).
2. Anti-sovradimensionamento: `hydraulic_collapse_progressive_share` in [0,1] (default 0)
   e' la quota dei crolli idraulici diretti RIASSEGNATA al canale. Il crollo diretto
   avviene per u < tasso x (1 - quota); la fascia tolta non produce nulla nel canale
   diretto (il danno locale idraulico resta com'era). Modo `calibrated` (default): tasso
   di innesco del ponte = quota x tasso diretto storico del ponte /
   `p_collapse_if_untreated`, quindi per chi non ispeziona e non arresta il tasso atteso
   di crollo idraulico resta quello storico per costruzione. Modo `fixed`: tasso per
   ponte-anno dato dall'utente. La quota vale solo con il tipo idraulico acceso
   (altrimenti sarebbe una sottrazione senza sostituzione). Diagnostica:
   `python -m infralab.lab.hydraulic_rate --scenario ... --horizon 200 --param ...`
   (`progressive_events.expected_hydraulic_collapse_rates`) stampa il tasso atteso per
   ponte-anno, diretto + latente, a regime e sull'orizzonte, contro l'ancora di circa
   1e-4 per ponte-anno (A3, A6; 0.7e-4 - 1.2e-4 sui ponti in acqua). Su orizzonte finito
   gli inneschi vicini alla fine non maturano: con finestra media di 20 trimestri e 200
   trimestri di orizzonte matura il 90% (totale = 95% dello storico con quota 0.5).
3. Stato latente: durata estratta alla nascita (`geometric`, `lognormal` arrotondata,
   `fixed`; minimo un trimestre). Fa parte dello stato vero.
4. Rilevazione: a ogni ispezione, probabilita' = POD dell'ispezione x fattore del tipo.
   Livelli: fattore `detect_factor_detailed` per le azioni con `inspection_level` >=
   `progressive_detailed_inspection_min_level` (default 2: L2 e L4) e per il
   monitoraggio; fattore `detect_factor_l1` per L1, sorveglianza intermedia, censimento
   L0 e rilievo post-intervento. POD dell'ispezione: 1 se il modello POD e' spento; se e'
   acceso (`progressive_detection_pod_mode="inspection"`) il difetto si vede solo se
   l'ispezione ha rilevato almeno un componente bersaglio; `"independent"` ignora la POD.
   Una quota e' autoevidente alla nascita. Monitoraggio attivo:
   `progressive_monitoring_detect_prob` (default 0). Il difetto noto compare
   nell'osservazione (`known_progressive_defects`: tipo, trimestre di scoperta), mai con
   la data di fine.
5. Arresto: azione `arresto_difetto` (tipo `arrest`), aggiunta al catalogo solo a canale
   acceso, eseguibile solo con un difetto noto; costo = `arrest_cost_ratio` del tipo x
   costo dell'intervento di riferimento, sommato sui difetti noti del ponte; durata
   `progressive_arrest_duration_quarters` (1), durante la quale il ponte e' occupato e
   impegna una squadra di lavoro; efficace dall'avvio. Intervento di riferimento:
   `ripristino_strutturale` (`progressive_arrest_reference_intervention`): e' l'unico
   intervento del catalogo che agisce esattamente sui componenti bersaglio (pile e
   spalle) e il suo guadagno (0.5) recupera per intero la perdita di una maturazione
   (0.20 idraulico, 0.35 geotecnico); le sostituzioni riparano un solo componente e la
   loro somma darebbe un arresto piu' caro del ripristino stesso. Arresta anche
   qualunque intervento il cui guadagno previsto su ogni componente bersaglio e' almeno
   `progressive_arrest_min_gain` (0.3: il bonus 0.15 delle sostituzioni altrui non
   basta), dall'avvio dei lavori, anche per difetti non noti.
   Soglia etica: un difetto NOTO e' un trigger (`progressive_floor_trigger`, default
   True). Regola agisci-o-chiudi come per lo stato sotto soglia: arresto d'ufficio se
   cassa e squadre lo consentono, altrimenti chiusura, e la soglia riprova ogni
   trimestre. Se il ponte e' anche sotto soglia e l'intervento scelto ripristina i
   bersagli si esegue quello; altrimenti prima l'arresto. Contatori propri
   (`progressive_floor_*`); la spesa entra anche in `floor_spend`.
6. Conseguenza a fine finestra: con probabilita' `p_collapse_if_untreated` crollo,
   trattato come un crollo diretto (perdita realizzata, ricostruzione, evento pubblico
   "crollo_progressivo"); altrimenti perdita di condizione dei bersagli (idraulico: i
   parametri del danno locale idraulico; geotecnico: lognormale), nascosta fino
   all'ispezione con la lista dei pendenti del canale improvviso (tipo
   `progressive:<tipo>`, contabilita' propria). Se il difetto era noto l'esito e'
   visibile subito.

Default e provenienza.

| Parametro | Default | Flag |
|---|---|---|
| `progressive_events_enabled` | False | - |
| `hydraulic_collapse_progressive_share` | 0.0 | PROVISIONAL |
| `progressive_hydraulic_enabled` / `progressive_geotechnical_enabled` | True / True | - (senza quota ed esposizione non fanno nulla) |
| `progressive_hydraulic_rate_mode` | calibrated | vincolo FROM SOURCE (A3, A6) |
| `progressive_hydraulic_rate_y` (solo fixed) | 1e-3 | PROVISIONAL (3e-4 - 3e-3) |
| `progressive_hydraulic_scour_multiplier` (solo fixed) | 3.0 | FROM SOURCE (derivato, A17) |
| `progressive_hydraulic_duration_dist` | geometric | PROVISIONAL |
| `progressive_hydraulic_duration_mean_q` | 20 | FROM SOURCE il campo (A12, A13, A22: 2-25 anni), PROVISIONAL il default |
| `progressive_hydraulic_duration_cov` (solo lognormal) | 1.0 | PROVISIONAL |
| `progressive_hydraulic_detect_factor_l1` | 0.5 | PROVISIONAL (direzione FROM SOURCE; 0.2 - 0.8) |
| `progressive_hydraulic_detect_factor_detailed` | 1.0 | PROVISIONAL |
| `progressive_hydraulic_self_evident_prob` | 0.1 | PROVISIONAL (0 - 0.3) |
| `progressive_hydraulic_p_collapse_if_untreated` | 0.05 | FROM SOURCE il campo (A7-A11: 0.01 - 0.5), PROVISIONAL il default |
| `progressive_hydraulic_arrest_cost_ratio` | 0.3 | PROVISIONAL (0.1 - 0.5) |
| `progressive_hydraulic_regional_trigger_prob` | 0.0 | PROVISIONAL |
| perdita idraulica a fine finestra | parametri `hydraulic_damage_*` | PROVISIONAL |
| `progressive_geotechnical_exposed_fraction` | 0.0 | PROVISIONAL (nessun dato) |
| `progressive_geotechnical_rate_y` / `_rate_other_y` | 1e-3 / 0 | PROVISIONAL (1e-4 - 5e-3) |
| `progressive_geotechnical_duration_dist` / `_mean_q` / `_cov` | lognormal / 4 / 1.0 | FROM SOURCE il campo (A30, A31, A33: 1-12 trimestri), PROVISIONAL default, famiglia e CoV |
| `progressive_geotechnical_detect_factor_l1` / `_detailed` | 1.0 / 1.0 | PROVISIONAL |
| `progressive_geotechnical_self_evident_prob` | 0.3 | PROVISIONAL (0.1 - 0.6) |
| `progressive_geotechnical_p_collapse_if_untreated` | 0.10 | PROVISIONAL (0.02 - 0.5) |
| `progressive_geotechnical_targets` / `_hits` | pile,spalle / all | PROVISIONAL |
| `progressive_geotechnical_drop_dist` / `_mean` / `_cov` | lognormal / 0.35 / 0.5 | FROM SOURCE qualitativo (A29), PROVISIONAL i valori |
| `progressive_geotechnical_arrest_cost_ratio` | 0.3 | PROVISIONAL (0.1 - 0.7) |
| `progressive_custom_events` | [] | - |
| `progressive_arrest_reference_intervention` | ripristino_strutturale | PROVISIONAL |
| `progressive_arrest_duration_quarters` | 1 | PROVISIONAL |
| `progressive_arrest_min_gain` | 0.3 | PROVISIONAL |
| `progressive_detailed_inspection_min_level` | 2 | PROVISIONAL |
| `progressive_detection_pod_mode` | inspection | PROVISIONAL |
| `progressive_monitoring_detect_prob` | 0.0 | PROVISIONAL |
| `progressive_floor_trigger` | True | scelta di regola |
| `collapse_game_over` / `collapse_game_over_exempt_closed` / `collapse_log_metrics` | none / True / False | scelta di regola |

Registro dei collassi. `env.collapse_log` ha una riga per collasso, sempre: trimestre,
ponte, gestore, causa (`pf`, `hydraulic_direct`, `progressive_<tipo>`,
`condition_threshold` per il modello storico), Pf annua e probabilita' trimestrale usate,
stato noto al gestore ALL'INIZIO del trimestre (condizione minima osservata e stimata,
beta annuo stimato, ultima ispezione, difetti in sviluppo noti, trigger della soglia
etica attivo, `known_critical`), danni e difetti nascosti pendenti, condizione vera,
limitazioni e lavori in corso al momento del collasso. `known_critical` = difetto noto
oppure stato noto sotto `ethical_floor_condition` / `ethical_floor_beta`, anche a soglia
spenta. Il runner scrive `collapses.csv` accanto a runs e steps (sola intestazione se non
ci sono collassi). Contatori `failures_cause_*`, `failures_hydraulic_total`,
`failures_known_critical`, `failures_with_known_defect`: solo a canale acceso, con
`collapse_log_metrics=True` o con `collapse_game_over` diverso da "none".

Fine partita. `collapse_game_over`: "none" (default: la partita continua, come oggi con
`failure_model="reliability"`), "any", "known_critical" (solo se il ponte era noto come
critico all'inizio del trimestre; non se era chiuso al traffico, salvo
`collapse_game_over_exempt_closed=False`). Il gestore esce come nel modello storico
(`last_failure_info.reason = "BRIDGE_COLLAPSE"`, con causa e regola).

Informazione. Difetti non noti, inizio e fine veri, contatori veri e registro dei
collassi non raggiungono policy e pianificatori in "inspected": tolti dalla fotografia
del sandbox e da `belief_clone`, dove dei difetti noti resta tipo e trimestre di
scoperta e la durata residua e' riestratta dal generatore del pianificatore. Nessun
evento dello storico per i difetti mai noti. leak_check: `perturb_hidden_progressive`
(difetti latenti tolti, spostati, aggiunti; inizio e fine dei noti; contatori e registro),
istantaneo e a traiettoria (le rilevazioni di B sono quelle di A, come le letture
d'ispezione), `--progressive-events` (`TEST_PROGRESSIVE_PARAMS`). TRAIN-01 con sandbox e
informazione stretta: pulito in entrambe le modalita'.

Correzione collegata (solo con `strict_information=True`): lo storico delle azioni tiene
gli ultimi 12 eventi, e un danno idraulico NASCOSTO del canale storico
(`hydraulic_legacy_mode=True`) ne spingeva fuori uno visibile, rivelandosi dalla
lunghezza dello storico ripulito. Ora con informazione stretta i 12 posti contano i soli
eventi visibili. Nessun effetto senza eventi "danno_idraulico" (quindi nessuno con
`hydraulic_legacy_mode=False`) ne' senza informazione stretta. Nel leak_check a
traiettoria lo storico di un mondo che e' la coda di quello dell'altro (effetto delle
sole etichette di danno) viene allineato.

Numeri casuali comuni. Flussi a chiave dedicati (scopi 210101-210108), indipendenti
dallo stato e da `rng_streams`: stessi inneschi, durate ed esiti per ogni strategia; i
difetti possono coesistere sullo stesso ponte, cosi' l'innesco non dipende dagli arresti.
Con il canale acceso e nessuna conseguenza la traiettoria TRAIN-01 coincide con lo
snapshot dorato (test).

Uscite (solo a canale acceso): `metric_progressive_*` / `final_progressive_*`
(`initiated`, `self_evident`, `detected_by_inspection`, `detected_by_monitoring`,
`inspections_missed`, `arrested` e per via, `matured`, `matured_known`, `collapses`,
`condition_loss_total`, `damage_*`, `floor_*`, per tipo `initiated_/known_/arrested_/
matured_/collapses_<tipo>`, `pending_hidden`, `pending_known`).

Interfacce: `UI_FIELDS` del modulo in `/api/config` e nella UI Streamlit (riquadro
avanzato); `params.<nome>` nelle specifiche di laboratorio; l'API mostra
`known_progressive_defects` per asset e, con la regola di fine partita, `failure_info`.

Limiti dichiarati. Nessuna famiglia Weibull per la finestra. Un collasso per altra causa
cancella i difetti in sviluppo del ponte (effetto di secondo ordine sulla taratura). Le
strategie Basic non richiedono l'arresto da sole: lo fa la soglia etica. Nessun trigger
di legge sul difetto noto. Default invariati: snapshot TRAIN-01 identico.

## 2026-10-09 — Tre aggiunte opt-in per la validazione sull'inventario reale (default invariati)

Nate dall'analisi parametrica del confronto con l'inventario americano dei ponti
(quota di ponti limitati, chiusi e con calo di due voti per fascia di voto). Con i
default nessuna delle tre ha effetto: snapshot TRAIN-01 identico, nessun flusso
casuale esistente spostato. Test: `tests/test_validation_options.py`.

- **Stima del gestore sulla scala del vero** (`assessment_epistemic_mode`, default
  `"manager"` = storico). Con `true_reliability_mode="physical"` il beta vero usa
  un'incertezza epistemica fissa (`physical_epistemic_reference`), mentre la stima del
  gestore usa la sua, che le ispezioni riducono; nel modello meno incertezza vuol dire
  resistenza media più alta e meno dispersione. Misurato su 147 ponti, gestore che
  ispeziona ogni due anni: la stima supera il vero di 0,26 in media (da 0,19 al sesto
  anno a 0,32 al quarantesimo), di cui 0,18 per questa sola differenza e 0,07 per i
  parametri latenti che il gestore non conosce. Chi non ispeziona mai ha l'effetto
  opposto. Con `"reference"` la stima usa la stessa incertezza di riferimento del
  vero: la conoscenza del gestore entra solo dalla condizione osservata. L'incertezza
  del gestore resta invariata nello stato (rumore delle ispezioni, osservazione).
  **Effetto sui numeri (solo se acceso):** beta stimato più basso per chi ispeziona;
  soglie di legge e soglia etica sul beta stimato scattano prima.
- **Cedimento locale di un elemento già degradato** (tipi utente del danno improvviso,
  `sudden_custom_events`): due chiavi nuove, `condition_max` (l'evento avviene solo se
  la condizione vera minima dei bersagli è sotto il valore) e `condition_exponent`
  (tasso × (1 − condizione minima)^esponente), e il valore `"worst"` per `hits`
  (colpisce il bersaglio in condizione peggiore, senza estrazioni; accettato anche dai
  canali regionale e progressivo). Motivo: nell'inventario reale i ponti in cattivo
  stato perdono due voti in un anno molto più spesso di quelli in buono stato (7,4%
  l'anno al voto 2, 2,4% al voto 3, 0,3–0,5% dal voto 4 al 7), mentre il degrado lento
  del modello non produce salti. Nessun valore di default: tasso, soglia ed esponente
  sono scelte dell'utente, PROVISIONAL finché non tarate.
- **Chiusura temporanea indipendente dallo stato** (`temporary_closure_rate_q`,
  default 0 = spento; `temporary_closure_mean_duration_quarters`, default 4,
  PROVISIONAL). Stesso schema del declassamento di portata: ingresso per ponte e
  trimestre, durata geometrica, quota attesa = tasso × durata media. Nell'inventario
  reale anche i ponti in buono stato risultano chiusi in piccola quota (0,04% dei
  ponti-anno ai voti 7–9, 0,1–0,2% ai voti 5–6). Alla scadenza torna il fattore di
  traffico del gestore con le limitazioni ancora in vigore (sanzione del regime,
  declassamento, soglia etica). Flussi a chiave dedicati (4204, 4205). Il
  declassamento che scade durante una chiusura temporanea non riapre il ponte.

Esposti alla piattaforma (`backend/app.py`, gruppo Benchmark) insieme a
`posting_entry_rate_q`, che non lo era.

## 2026-10-09 — Gestore tipico, chiusura legata allo stato, declassamento revocato dai lavori (opt-in, default invariati)

Seconda serie di aggiunte nate dalla taratura sull'inventario americano dei ponti.
Con i default nessuna ha effetto: snapshot TRAIN-01 identico, nessun flusso casuale
esistente spostato. Test: `tests/test_typical_owner_options.py`.

- **Policy elementare `typical_owner`** (`infralab/agents/basic.py`). Ogni anno
  interviene su un ponte con una probabilità che dipende solo dal voto osservato
  (0-9, dal centro dell'intervallo osservato del componente peggiore). Parametri:
  `annual_rate_by_rating` (tabella voto → probabilità annua; default = probabilità
  annua che il voto peggiore salga di almeno un punto nell'inventario reale, stati
  PA, TX, OH, IA, 2015-2022), `rate_scale`, `major_share` (quota di interventi
  maggiori; gli altri sostituiscono il solo componente peggiore), `seed`.
  Estrazione a chiave (seme, ponte, trimestre). La policy non conosce il seme della
  partita: per repliche diverse va passato un `seed` diverso. È una regola veloce,
  senza simulazioni: serve a dare al parco una distribuzione di voti realistica,
  non è una strategia ottimizzata.
- **Chiusura temporanea legata allo stato** (`temporary_closure_condition_rate_q`,
  default 0 = spento). La probabilità di ingresso per ponte e trimestre diventa
  `temporary_closure_rate_q + temporary_closure_condition_rate_q × 2 × (1 − condizione
  vera minima)`. Stesso flusso a chiave (4204): con il valore a zero le estrazioni
  sono identiche a prima. **Effetto sui numeri (solo se acceso):** più chiusure sui
  ponti in cattivo stato, a parità di chiusure su quelli buoni.
- **Declassamento di portata: due opzioni.** `posting_severity_floor` (default 0,25
  = valore che prima era fisso nel codice) è il minimo della severità
  `max(pavimento, 2 × (1 − condizione))`; a 0 un ponte in ottimo stato non viene quasi
  mai declassato. `posting_lift_on_intervention` (default spento): un intervento
  strutturale in corso revoca il declassamento casuale e, finché durano i lavori,
  non ne nasce uno nuovo; il traffico torna pieno solo se non ci sono sanzione,
  soglia etica o chiusura temporanea. La manutenzione preventiva non revoca nulla.
  **Effetto sui numeri (solo se accesi):** meno ponti limitati fra quelli in buono
  stato e appena rifatti.

I valori usati nella taratura sono PROVISIONAL finché la verifica sugli stati non
ancora guardati non è fatta. Esposti alla piattaforma (`backend/app.py`, gruppo
Benchmark); la policy compare nell'elenco delle policy elementari.

## 2026-10-09 — Modalità di applicazione per singolo obbligo (opt-in, default invariati)

Grammatica delle leggi (`infralab/regimes/base.py`): ogni obbligo può dichiarare il
campo `enforcement` (`forced`, `sanction`, `both`, `none`). Se manca vale la modalità
della legge (`enforcement.mode`), come prima; la serializzazione delle leggi senza
il campo è identica. Serve a scrivere una legge con obblighi di tipo diverso: per
esempio un'ispezione eseguita d'ufficio accanto a un obbligo di intervento fatto
valere con limitazione di carico e chiusura. Nel motore: l'esecuzione d'ufficio
salta gli obblighi non `forced`; la scala delle sanzioni e la revoca guardano solo
gli obblighi `sanction`. Il conteggio delle inadempienze e le multe continuano a
guardare tutti gli obblighi dovuti. Test: `tests/test_obligation_enforcement.py`.
Snapshot TRAIN-01 identico.

## 2026-10-10 — Risposta della soglia etica a un problema noto (opt-in, default invariato)

**Perché.** Una revisione esterna del lavoro sulle norme ha osservato che l'effetto di un obbligo di
ispezione sul gestore passivo misura la catena "ispezione, rilevazione, intervento imposto", e che la
soglia etica è di fatto una politica operativa di sicurezza. Per separare il valore della rilevazione
da quello dell'intervento serve poter cambiare la risposta a parità di informazione.

**Cosa.** Nuovo parametro `ethical_floor_response`:
- `"repair_or_close"` (default): comportamento precedente, ripara se cassa e squadre lo consentono,
  altrimenti chiude;
- `"close_only"`: la soglia non ripara mai; il ponte con stato noto sotto soglia viene chiuso e
  riaperto quando lo stato noto torna sopra soglia;
- `"restrict_only"`: la soglia non ripara né chiude; limita il carico col fattore
  `posting_restriction_factor` finché lo stato noto resta sotto soglia.
Con le due risposte nuove i contatori `budget_blocked` e `capacity_blocked` della soglia non crescono
(l'intervento non viene tentato); `closures` e `closure_bridge_quarters` contano anche le limitazioni.

**Default invariati.** Con il default lo snapshot TRAIN-01 è identico. Test: `tests/test_floor_response.py`.
