# Campagna X2 - scomposizione dello scarto fra stima del gestore e vero (nota scritta prima delle partite, 2026-11-02)

## Perché
Cinque strutturisti su sei (control/review-structural.json) mostrano dal codice che lo scarto stima - vero e' fatto da
meccanismi del modello, non da un effetto delle ispezioni:
(a) parametri medi del profilo nella stima, campionati nel vero (piu' selezione quando si guardano i ponti sotto il limite);
(b) stato osservato fermo all'ultima ispezione, non proiettato in avanti;
(c) incertezza epistemica del gestore nella resistenza media (termine 2,24 + 0,20 (1 - eu)) e nei CoV, mentre il vero
    la tiene a 0,22: chi ispeziona alza la propria stima;
(d) campioni Monte Carlo (256) diversi per stima e vero;
(e) rumore delle osservazioni (circa 0,4 voti, contro circa 1,2 nella letteratura sulla variabilita' delle ispezioni).
Sostituisce la campagna X (control/NOTA-campagna-X.md), mai giocata. Il braccio del valutatore prudente e' tolto: conta
due volte l'incertezza (fattore sulla resistenza dentro un beta che ha gia' il CoV crescente con eu).
La configurazione del paper resta con il modello del gestore diverso da quello del mondo (decisione dell'autore,
2026-10-15): i bracci sono diagnostica, non una nuova configurazione.

## Cosa si gioca
Gestore passivo, partite di Z-princ (control/NOTA-campagna-Z.md), 147 ponti, 50 anni, D0 e D1, soglia sulla stima
3,02, 10 semi. Sette bracci, ciascuno con opzioni del motore nuove o esistenti (spente di default):
- base: Z. Deve riprodurre al centesimo le partite del passivo di Z-princ.
- rif: assessment_epistemic_mode "reference" (spegne c).
- param: assessment_parameter_mode "sampled" (spegne a).
- semi: assessment_seed_mode "shared" (spegne d).
- proiez: assessment_state_mode "projected" (spegne b; l'eta' efficace avanza dal trimestre in cui la stima vede
  l'osservazione).
- tutti: rif + param + semi. Con osservazione perfetta la stima coincide con il vero (test del motore
  test_tutto_allineato_stima_uguale_al_vero): resta solo il rumore di osservazione e l'osservazione ferma.
- rumore3: inspection_sigma_scale 3 (rumore al livello della letteratura).
140 partite. In ogni partita si registra ogni controllo della soglia (stima e vero).

## Lettura fissata ora
Per ogni partita: scarto medio stima - vero; stima mediana sui controlli con vero < 3,02; rilevazione (quota dei
controlli con vero < 3,02 e stima < 3,02); falsi allarmi (quota dei controlli con vero >= 3,02 e stima < 3,02); AUC
della stima come classificatore del vero sotto il limite; interventi della soglia; S1, crolli attesi, ponti e quota
di tempo sotto il limite, spesa, quota chiusa. Differenze appaiate per seme dal braccio base, intervallo al 95%.
- Scomposizione: per ogni meccanismo, quota dello scarto medio base tolta spegnendolo (base - braccio) / base, su D0 e
  D1. Si riporta tutto, senza soglia di successo; le parti possono non sommarsi a uno (interazioni).
- Domanda "con l'obbligo la stima e' piu' ottimista dove conta": si scrive solo se, nel braccio base, la stima mediana
  condizionata e' piu' alta con D1 che con D0 in almeno 9 semi su 10 e, nel braccio rif, la differenza D1 - D0 si
  riduce in almeno 9 semi su 10. Altrimenti nel paper si scrive solo l'osservazione (rilevazione, falsi allarmi, AUC
  con e senza obbligo).
Attese: rif e tutti riducono lo scarto; rumore3 abbassa l'AUC. Nessuna attesa sul verso della domanda.
Nulla si sceglie in base all'esito.

## Limiti noti prima di giocare
1. Solo il gestore passivo.
2. Le popolazioni di controlli con vero sotto il limite differiscono fra D0 e D1 (selezione dopo il trattamento): per
   questo si riportano rilevazione, falsi allarmi e AUC accanto alle mediane condizionate.
3. La proiezione usa la curva della stima (parametri medi), non la legge del mondo.
4. Niente distorsione sistematica degli ispettori ne' probabilita' di rilevazione (opzioni esistenti, non giocate).
