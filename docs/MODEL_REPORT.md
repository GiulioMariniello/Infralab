# BridgeManager 2025 — Rapporto Tecnico del Modello di Simulazione

**Versione:** 2025 · **Autore:** Giulio Mariniello  
**Piattaforma:** Streamlit + Docker su Hugging Face Spaces  
**Riferimenti normativi principali:** EN 1990, EN 1991-2, JCSS PMC (2001), NTC 2018, MIT Linee Guida Ponti (2020)

---

## 1. Architettura Generale

Il simulatore modella la gestione di un portafoglio di ponti su un orizzonte temporale discretizzato in **trimestri** (un passo = 3 mesi). Ogni trimestre, un agente (umano o algoritmico) osserva lo stato dei propri ponti e pianifica interventi. L'ambiente avanza deterministicamente la fisica del degrado, applica le azioni pianificate, aggiorna l'affidabilità strutturale, calcola i flussi economici e verifica le condizioni di fine-partita.

```
Turno t
├── 1. Completamento interventi scaduti (gain sui componenti)
├── 2. Raccolta piani degli agenti
├── 3. Esecuzione azioni (analisi / monitoraggio / interventi / proof load)
├── 4. Degrado stocastico (componenti non in manutenzione)
├── 5. Refresh previsioni monitoraggio attivo
├── 6. Evento sismico stocastico (se abilitato)
├── 7. Interessi su budget + incassi trimestrali
├── 8. Verifica condizioni di fallimento (crollo strutturale / bancarotta)
└── 9. Aggiornamento affidabilità di rete → quarter += 1
```

I dati di stato vengono serializzati in `game_state.json` ad ogni turno; i log storici (azioni, budget, KPI) vengono aggiunti ai rispettivi CSV di sessione.

---

## 2. Modello di Degrado dei Componenti

### 2.1 Struttura di un Componente

Ogni ponte è composto da 4 **componenti strutturali**: `impalcato`, `pile`, `spalle`, `appoggi`. Ogni componente è descritto da:

| Attributo | Significato |
|-----------|-------------|
| `current_condition` ∈ [0,1] | Condizione strutturale reale (0 = collasso, 1 = perfetto) |
| `effective_age_years` | Età efficace (aumenta in base a traffico, ambiente, stocasticità) |
| `vita_utile_anni` | Vita utile nominale del componente (variabile casuale ~N(L, 0.10L)) |
| `epistemic_uncertainty` ∈ [0.03, 0.40] | Incertezza epistemica sulla resistenza |
| `observed_condition_min/max` | Intervallo di confidenza post-ispezione (None se mai ispezionato) |

### 2.2 Legge di Degrado (Weibull Survival)

La condizione di un componente all'età efficace *t* è:

```
C(t) = exp(−λ · (t/L)^k)
```

dove:
- `L` = vita utile componente (anni)  
- `k` = `exponent × degradation_shape_factor` (esponente di Weibull, default k ≈ 3.0 × 1.05–1.35 a seconda del componente)  
- `λ` = `degradation_scale` (scala casuale ~N(scale_mean, 0.10), clampato ≥ 0.30)

I parametri `scale_mean` e `shape_factor` sono specifici per componente (tabella `COMPONENT_PROFILES`):

| Componente | shape_factor | scale_mean | traffic_sens | env_sens |
|------------|-------------|------------|-------------|---------|
| impalcato  | 1.05        | 0.95       | 0.95        | 0.85    |
| pile       | 1.20        | 1.10       | 0.70        | 0.95    |
| spalle     | 1.10        | 1.00       | 0.60        | 1.00    |
| appoggi    | 1.35        | 1.12       | 0.75        | 1.10    |

Tutti i parametri personali di ogni componente sono campionati da distribuzioni normali con la media del profilo e σ ≈ 0.05–0.10, garantendo eterogeneità tra ponti diversi.

### 2.3 Avanzamento dell'Età Efficace per Turno

Ogni trimestre (se il componente non è in manutenzione attiva):

```
Δt_efficace = 0.25 · τ_traffico^traffic_sensitivity · E_env^env_sensitivity · P_process
```

dove:
- `τ_traffico = 1.0 + k_traffic · (ADT / ADT_ref)` — moltiplicatore traffico (default: k_traffic = 0.05, ADT_ref = 10,000 veicoli/giorno)  
- `E_env ~ N(1.0, 0.10)` — fattore ambientale stocastico (campionato una volta per turno, identico per tutti i ponti)  
- `P_process ~ LogN(0, σ_process)` — moltiplicatore di processo stocastico per componente (σ_process = 0.12 di default, clampato ≥ 0.35)

Questo approccio cattura: (a) accelerazione del degrado per alto traffico, (b) correlazione parziale tra ponti tramite il fattore ambientale condiviso, (c) variabilità idiosincratica per componente tramite il moltiplicatore lognormale.

### 2.4 Effetto degli Interventi sull'Età Efficace

Quando un intervento termina, ogni componente target riceve un **gain** g ~ N(g_nominale, σ_gain) ≥ 0:

- Se g ≥ 1.0 (sostituzione completa): condizione → 1.0, età efficace → 0
- Altrimenti: condizione += g (clampato a 1.0), poi si **inverte la legge di Weibull** per ricavare la nuova età efficace:
  ```
  degradation = −ln(C_new)
  age_ratio = (degradation / λ)^(1/k)
  effective_age = L · age_ratio
  ```

Dopo ogni intervento: `epistemic_uncertainty = max(0.06, ε × 0.90)` (riduzione del 10%, il componente è meglio conosciuto).

### 2.5 Aumento Epistemico per Invecchiamento

Ogni trimestre senza ispezione, per ogni componente:  
`epistemic_uncertainty = min(0.40, ε × 1.01)` — +1% per trimestre, simula l'aumento dell'incertezza con il passare del tempo senza ispezioni.

---

## 3. Modello di Affidabilità Strutturale

### 3.1 Gerarchia: Componente → Stato Limite → Sistema

Per ogni componente vengono valutati **3 stati limite**:

| Stato Limite | Categoria | Modalità di Rottura |
|-------------|-----------|---------------------|
| Flessione (flexure) | ULS | Duttile |
| Taglio (shear) | ULS | Fragile |
| Fessurazione (cracking) | SLS | Servizio |

Per ogni stato limite si esegue una stima FOSM campionaria. I risultati ULS del componente vengono combinati in **serie** (sistema componente). I risultati di tutti i componenti strutturali del ponte vengono combinati in **serie** (sistema ponte).

### 3.2 Variabili Aleatorie per Stato Limite

Per ogni stato limite vengono campionati `N` (default 256) vettori di variabili aleatorie:

```
R_i = R_mean · ξ_mat · ξ_geom · θ_R     (resistenza)
S_i = θ_S · (G_i + Q_i)                  (domanda)
```

dove:
- `ξ_mat ~ LogN(1, v_mat)` — incertezza materiale (v_mat = f(resistance_cov, condizione, epistemic))
- `ξ_geom ~ LogN(1, v_geom)` — incertezza geometrica (v_geom = f(geometry_cov, shape_factor))
- `θ_R ~ LogN(1, v_mod_R)` — incertezza di modello sulla resistenza (v_mod_R = f(model_cov, epistemic))
- `θ_S ~ LogN(1, v_mod_S)` — incertezza di modello sulla domanda
- `G ~ N(g_dead, v_dead)` — carico permanente (normale)
- `Q ~ Gumbel(q_live, v_live)` — carico variabile (Gumbel)

### 3.3 Parametri delle Distribuzioni

**Resistenza media** (per componente, per stato limite):

```
R_mean = resistance_bias · max(C, 0.02) · (2.24 + 0.20 · (1 − ε))
```

dove C = condizione corrente, ε = epistemic_uncertainty. Il fattore 2.24 è il bias nominale (da calibrare su dati AINOP/JCSS).  
Per flessione: R_mean × 1.08 | Per taglio: R_mean × (0.92 + 0.06 · traffic_sens) | Per cracking: R_mean × crack_bias × 0.76

**Domanda media**:

```
demand_base = demand_bias · (0.54 + 0.18 · τ_traffico + 0.10 · ADT/ADT_ref)
             × (1 + 0.12 / η_traffico)
G_mean = 0.38 · demand_base     Q_mean = 0.62 · demand_base
```

dove η_traffico è la mediana della fragilità al traffico (vedi §7).

**COV**:

| Parametro | Formula |
|-----------|---------|
| v_mat | min(0.24, 0.04 + 0.40·v_res + 0.05·(1−C)) |
| v_geom | min(0.10, v_geom_base · (1 + 0.12·shape_factor)) |
| v_mod_R | min(0.16, v_mod_base · (1 + 0.75·ε)) |
| v_mod_S | min(0.14, 0.04 + 0.30·v_demand + 0.05·ε) |
| v_dead | min(0.08, max(0.05, v_demand·0.50)) |
| v_live | min(0.18, max(v_demand, 0.04 + 0.22·β_fragilità + 0.06·(1−C))) |

### 3.4 Stima FOSM dai Momenti Campionari

Invece del conteggio diretto dei fallimenti (instabile a N basso per β > 3), si usa la formula **FOSM lognormale** sui momenti campionari:

```
σ_lnR = sqrt(ln(1 + COV_R²))
σ_lnS = sqrt(ln(1 + COV_S²))
β_annual = ln(μ_R / μ_S) / sqrt(σ_lnR² + σ_lnS²)
pf_annual = Φ(−β_annual)
```

Questa formulazione è identica all'approccio FOSM standard (Hasofer-Lind per variabili lognormali) ma usa momenti stimati da campioni, garantendo stabilità per qualsiasi valore di β e N.

### 3.5 Periodo di Riferimento

La probabilità di fallimento su periodo di riferimento T = 50 anni è calcolata come:

```
pf_ref = 1 − (1 − pf_annual)^T
β_ref = −Φ⁻¹(pf_ref)
```

Per la valutazione della **non-stazionarietà** (affidabilità del ponte nel ciclo di vita), si proietta il profilo anno-per-anno per i prossimi `reliability_projection_years` anni e si calcola:

```
pf_TV = 1 − Π_{t=1}^{T} (1 − pf_annual(t))
```

in accordo con Rackwitz (2001) e Frangopol et al. (2004). Questo è usato come `bridge.failure_probability` (riferimento al periodo residuo di vita).

### 3.6 Target di Affidabilità (JCSS PMC / EN 1990)

| Classe di Conseguenza | β_rif (T=50 anni) | β_ann (derivato) |
|----------------------|-------------------|-----------------|
| CC1 | 3.3 | 4.2 |
| CC2 (default) | 3.8 | 4.7 |
| CC3 | 4.3 | 5.2 |

Target SLS: β_rif = 1.5 (indipendente dalla CC).  
Target stati fragili: β += 0.30 rispetto al target globale della CC.

La violazione del target viene segnalata nei `beta_margin_reference/annual` negativi visualizzati nella dashboard.

### 3.7 Combinazione a Sistema (Serie)

L'ipotesi di **indipendenza** (conservativa per componenti positivamente correlati) dà:

```
pf_sistema = 1 − Π_i (1 − pf_i)
```

Prima applicata ai 3 stati limite di un componente (→ affidabilità componente), poi ai N componenti strutturali del ponte (→ affidabilità strutturale ponte), con possibile aggiunta della componente sismica `extra_annual_pf` se il terremoto è abilitato.

---

## 4. Modello Informativo: Ispezioni e Incertezza Epistemica

### 4.1 Osservazione della Condizione

Un'ispezione di tipo `analysis` o `monitoring` genera un'**osservazione rumorosa** della condizione reale:

```
σ_obs = max(0.005, σ_base · (0.65 + ε))
C_obs ~ N(C_reale, σ_obs)
[obs_min, obs_max] = [C_obs − 1.96·σ_obs, C_obs + 1.96·σ_obs] (clampato in [0,1])
```

dove σ_base è l'`error_sigma` dell'azione (L1: 0.10, L4: 0.01, monitoring: 0.005) e ε = `epistemic_uncertainty` del componente. Questo riflette l'asimmetria informativa fondamentale: ispezioni visive hanno alta incertezza, verifiche strutturali bassa.

### 4.2 Aggiornamento dell'Incertezza Epistemica

Ogni ispezione riduce l'incertezza epistemica del componente tramite il `knowledge_gain` dell'azione:

```
ε_new = max(0.03, ε · (1 − knowledge_gain))
```

Valori di knowledge_gain: monitoring = 0.30, proof_load_test = 0.50. L1_inspect e L4_verify non hanno knowledge_gain esplicito (aggiornano l'osservazione ma non ε direttamente; solo indirettamente attraverso la riduzione post-manutenzione ×0.90).

**Nota**: il proof load test aggiorna ε solo tramite `_inspect_bridge` con knowledge_gain = 0.50, non duplicando la riduzione nel modulo di affidabilità.

### 4.3 Classe di Attenzione (CdA)

Calcolata sulla `min_observed_condition` (minimo su tutti i componenti osservati):

| CdA | Soglia | Scadenza ispezione |
|-----|--------|-------------------|
| Bassa | > 0.90 | +8 trimestri |
| Medio-Bassa | > 0.80 | +6 trimestri |
| Media | > 0.60 | +4 trimestri |
| Medio-Alta | > 0.40 | +2 trimestri |
| Alta | ≤ 0.40 | +1 trimestre |
| Sconosciuta | mai ispezionato | — |

Con `enforce_guidelines = True`, il mancato rispetto delle scadenze genera un backlog conteggiato nei KPI.

### 4.4 Monitoraggio Continuo

Quando il monitoraggio è attivo (`monitoring_ends_quarter > quarter`), il simulatore calcola ad ogni turno la **curva predittiva di degrado** per i prossimi `monitoring_duration_years × 4` trimestri, proiettando l'età efficace con il moltiplicatore medio di traffico e ambiente (senza il termine stocastico). La banda di confidenza è ±1.96·σ_obs. Queste previsioni vengono aggiornate ad ogni turno finché il monitoraggio è attivo.

---

## 5. Proof Load Test

### 5.1 Logica del Test a Passi

Il test viene eseguito a `step_count` livelli crescenti (default 3) da α/step_count a α, dove α = fattore di carico (default 0.95 × carico codice):

```
test_demand_k = θ_test · (G + α_k · Q_codice)
```

Per ogni campione i e ogni step k:
- Se R_i ≤ test_demand_k su uno stato SLS → campione **schermato** (serviceability failure durante test)
- Se R_i ≤ test_demand_k su uno stato ULS → campione **fallito** (collasso durante test)

I campioni schermati o falliti non partecipano ai passi successivi. Al termine:
- `pass_probability` = P(sopravvissuto a tutti i passi)
- `screening_probability` = P(schermato senza collasso)
- `failure_probability_during_test` = P(collasso durante il test)

### 5.2 Aggiornamento Bayesiano Analitico (Lognormale Troncata)

Per ogni stato limite, la resistenza R ~ LogN è condizionata su R > x₀ (soglia di test):

```
x₀ = G_mean + α · Q_codice
```

Momenti esatti della distribuzione troncata (Aitchison & Brown 1957; JCSS PMC §3.7):

```
E[R | R > x₀] = R_mean · Φ(σ − k) / Φ(−k)
E[R² | R > x₀] = R_mean² · exp(σ²) · Φ(2σ − k) / Φ(−k)
Var[R | R > x₀] = E[R²] − E[R]²
COV_post = sqrt(Var) / E[R]
```

dove k = (ln x₀ − μ_lnR) / σ_lnR e Φ è la CDF normale standard.

**Decomposizione dell'incertezza**: il COV posteriore è scomposto in componenti irreducibili (geometria, modello) e aggiornabile (materiale):

```
v_mat_post = sqrt(max(0, COV_post² − v_geom² − v_mod²))
```

Proprietà formale: il test proof load **non può diminuire la media di R** (è un'evidenza positiva per i sopravvissuti). Il β posteriore è ≥ β priore per costruzione.

### 5.3 Selezione Adattiva dell'Intensità

Se `max_failure_probability` è specificato nella configurazione del test, il simulatore esplora una griglia di valori α ∈ [α_min, α_max] e seleziona l'α massimo che soddisfa il vincolo sul rischio di danno durante il test.

---

## 6. Modello di Fragilità al Traffico

### 6.1 Modelli di Carico Disponibili

| Modello | η mediana | β dispersione | Uso tipico |
|---------|-----------|---------------|------------|
| GL_light | 7.17 | 0.40 | Traffico leggero |
| GL_medium | 3.19 | 0.36 | Traffico misto |
| GL_heavy | 2.33 | 0.33 | Traffico pesante (default) |
| NTC | 1.68 | 0.33 | NTC 2018 LM1 |

Il parametro η è il rapporto mediano tra la capacità e l'effetto del carico codice (Q_codice). Un η alto significa struttura più resistente al traffico.

### 6.2 Fattori di Correzione

```
η_eff = η_base × lane_penalty × offset_bonus × remaining_area_bonus × age_penalty
```

- `lane_penalty`: +10% per 2 corsie, +35% per 1 corsia
- `offset_bonus = 1 + 0.22 × d_offset` — beneficio dell'eccentricità del carico
- `remaining_area_bonus = 1.03` se l'area rimanente è inattiva
- `age_penalty = 1 − 0.18 × (1 − structural_min_condition)` — degrado strutturale riduce la capacità al traffico

Il tasso di fallimento annuale al traffico dipende dal modello:

```
per GL_heavy: λ_ann = pf_codice × P_overload_month × P_congestion
per NTC: λ_ann = pf_codice × 0.001
```

---

## 7. Modello Sismico

L'analisi sismica usa un **modello di fragilità lognormale** calibrato su classi di ponti (standard/regular/irregular) e stati limite (GdL, SD, NC):

```
P(danno | PGA) = Φ((ln(PGA/η_sism) / β_sism))
```

Il PGA controllante viene stimato come funzione della probabilità di fallimento strutturale annuale:

```
PGA_ctrl = 0.04 + 0.18 · pf_annual + 0.015 · M_max
```

La probabilità annuale di eccedenza sismica è: `P_sism_ann = P(danno | PGA_ctrl) × P(terremoto)`.

Il terremoto stocastico è campionato ogni trimestre con probabilità `earthquake_annual_prob / 4`, con magnitudo ~ U[M_min, M_max] e posizione casuale nel raggio configurato attorno alle coordinate del sito.

---

## 8. Modello di Rete: Importanza e Priorità

### 8.1 Fattore di Importanza della Rete

Per ogni ponte:

```
NIF = w_adt × (ADT_ponte / ADT_max) + (1 − w_adt) × (centralità / centralità_max)
```

La **centralità** è l'inverso della distanza media ai 3 ponti più vicini (distanza geografica haversine). Il peso predefinito è `w_adt = 0.65`.

### 8.2 Risk Priority Index

```
RPI = NIF × pf_annual
```

Permette di priorizzare i ponti con alta importanza strategica e alta probabilità di fallimento.

---

## 9. Modello Economico

### 9.1 Flusso di Budget per Turno

All'inizio del turno, il budget matura **interessi asimmetrici**:

```
Se budget ≥ 0:  budget_new = budget × (1 + r_deposito)^0.25    r_deposito = 2%/anno
Se budget < 0:  budget_new = budget × (1 + r_debito)^0.25      r_debito = 5%/anno
```

I costi degli interventi vengono sottratti **prima** dell'incasso. L'incasso trimestrale viene poi aggiunto:

```
budget_finale = budget_dopo_costi × (1 + r) + incasso_trimestrale
```

### 9.2 Incasso Trimestrale

L'incasso del ponte dipende dal suo stato operativo:

- In manutenzione attiva: `incasso × (1 − revenue_impact_operazione)`
- Operativo con HI > 0.5: incasso pieno
- Operativo con HI ∈ [0.2, 0.5]: `incasso × (HI − 0.2) / 0.30`  
- Operativo con HI ≤ 0.2: incasso zero

dove HI = Health Index (vedi §9.3). Il revenue multiplier usa HI e non la condizione minima, catturando il degrado funzionale graduale.

### 9.3 Health Index (HI)

Media ponderata della condizione dei componenti, pesata per i pesi architetturali:

```
HI = Σ_i (w_i × C_i) / Σ_i w_i
```

Pesi default: impalcato 50%, pile 30%, spalle 15%, appoggi 5%.

### 9.4 Net Asset Value (NAV)

```
NAV = V_operativa − costo_ripristino_differito − perdita_ricavi_attuale
```

con:
- `V_operativa = V_iniziale × HI` — valore corrente proporzionale alla salute
- `costo_ripristino_differito = costo_ripristino_completo × (1 − HI)` — stima del backlog manutentivo
- `perdita_ricavi_attuale = incasso × revenue_impact × trimestri_rimanenti` (solo se in manutenzione attiva)

Il NAV è clampato a ≥ 0 e fornisce una stima del valore economico netto dell'asset strutturale, tenendo conto sia del degrado che dei costi di ripristino impliciti e dell'impatto degli interventi in corso sui ricavi.

---

## 10. Azioni di Manutenzione

### 10.1 Tipologie

| Tipo | Azione | Costo | Durata | Effetto |
|------|--------|-------|--------|---------|
| analysis | L1_inspect | 1 €/m² | Istantaneo | Osservazione ± 10% |
| analysis | L4_verify | 100 €/m² | Istantaneo | Osservazione ± 1% |
| monitoring | monitoring | 70 €/m² | Istantaneo (attivo 5 anni) | Osservazione ± 0.5%, previsioni future |
| proof_load | proof_load_test | 25 €/m² | Istantaneo | Aggiornamento Bayesiano β |
| intervention | replace_impalcato | 4000 €/m² | 4 trimestri | gain = 1.0 su impalcato, 0.15 sugli altri |
| intervention | replace_pile | 3000 €/m² | 3 trimestri | gain = 1.0 su pile, 0.15 sugli altri |
| intervention | replace_spalle | 2500 €/m² | 3 trimestri | gain = 1.0 su spalle, 0.15 sugli altri |
| intervention | replace_appoggi | 100,000 € fissi | 2 trimestri | gain = 1.0 su appoggi, 0.15 sugli altri |
| intervention | ripristino_strutturale | 1500 €/m² | 3 trimestri | gain = 0.5 su pile + spalle (σ = 0.05) |
| intervention | ripristino_completo | 3500 €/m² | 5 trimestri | gain impalcato 0.7, pile 0.6, spalle 0.5, appoggi 0.4 (σ = 0.05) |

Gli interventi strutturali completi (replace_*) ripristinano completamente (gain = 1.0) il componente target e forniscono un **bonus del 15%** agli altri componenti (effetto trascinamento da ispezione/lavori accessori).

Il gain effettivo è campionato da N(g_nominale, σ_gain) clampato a ≥ 0 per evitare gain negativi.

### 10.2 Vincoli di Pianificazione

- Un ponte in manutenzione attiva non può ricevere nuove azioni.
- Non si può pianificare la stessa azione su un ponte per due trimestri consecutivi (eccezione: L1_inspect).
- Non si può avviare il monitoraggio se già attivo.
- Il budget viene verificato al momento dell'esecuzione: se insufficiente, l'azione è annullata.
- Le azioni `allowed_actions` per scenario filtrano le opzioni disponibili sin dall'avvio della sessione.

---

## 11. Condizioni di Fine-Partita

### 11.1 Collasso Strutturale

L'agente viene eliminato se qualsiasi ponte sotto la sua gestione raggiunge:

```
structural_min_condition ≤ failure_threshold (default 0.10)
```

dove `structural_min_condition = min(C_impalcato, C_pile, C_spalle)` esclude gli appoggi (componente funzionale, non strutturale portante).

### 11.2 Bancarotta

L'agente viene eliminato se il budget rimane **negativo per 4 trimestri consecutivi** (= 1 anno):

```
negative_budget_streak ≥ 4
```

### 11.3 Fine Naturale

La simulazione termina quando viene raggiunto il numero massimo di trimestri configurato (`max_quarters`) o al termine dell'orizzonte temporale (`years × 4`).

---

## 12. Flusso Sperimentale (Modalità Ricerca)

```
Accesso giocatore (link diretto o codice partita)
        ↓
Creazione sessione: scenarioID_Nome_Timestamp
        ↓
Phase: tutorial  (5 passi didattici)
        ↓
Phase: playing   (max_quarters trimestri, dashboard interattiva)
        ↓
Phase: questionnaire  (se show_questionnaire = True nel scenario)
        ↓
Phase: done  (debrief con confronto vs baseline euristica)
```

Il debrief include una simulazione post-hoc della baseline euristica sullo stesso scenario, permettendo il confronto oggettivo delle strategie.

Il questionario raccoglie: anni di esperienza, ruolo, familiarità con affidabilità, strategia dichiarata, consapevolezza dell'asimmetria informativa, uso dell'indice β nelle decisioni.

---

## 13. KPI Registrati per Sessione

| KPI | Calcolo |
|-----|---------|
| `health_index` | Σ(C_min_ponte × ADT) / Σ ADT (media pesata per traffico) |
| `total_asset_value` | Σ initial_asset_value × min_condition |
| `net_worth` | budget + total_asset_value |
| `backlog_count` | ponti con next_inspection_quarter < quarter |
| `inspection_coverage` | ponti con almeno un'osservazione / totale |
| `knowledge_gap_count` | ponti mai ispezionati |
| `active_monitoring_count` | ponti con monitoraggio attivo |
| `network_reliability_index` | min(β_post) su tutti i ponti |
| `network_failure_probability` | max(pf_post) su tutti i ponti |

Tutti i KPI vengono registrati in CSV ad ogni turno per analisi scientifiche successive.

---

## 14. Limitazioni e Assunzioni Note

| Limitazione | Impatto | Stato |
|-------------|---------|-------|
| Indipendenza componenti nel sistema serie | Conservativo (pf_sistema sovrastimata) | Accettato per semplicità |
| Fattore 2.24 in base_resistance non calibrato su dati reali | R_mean relativo, non assoluto — coerente internamente | Da calibrare su AINOP/JCSS |
| pf_post ≥ pf_prior per proof load | Proprietà matematica del condizionamento su R > x₀ | Corretta e dichiarabile |
| Correlazione tra ponti via solo fattore ambientale | Correlazione strutturale non modellata | Accettato |
| Fragilità sismica semplificata (PGA empirica) | Non sostituisce analisi sismica dettagliata | Per scopi didattici |
| revenue disconnesso da β | Design sperimentale intenzionale | Non da correggere |

---

*Documento generato automaticamente — versione del codice corrispondente al commit post-sessione 2026-05-16.*
