# Campagna ZT - Esempio 1 con una taratura diversa del rapporto condizione-affidabilita' (nota scritta prima delle partite, 2026-11-05)

## Perche'
Revisione esterna (criticita' 1) e esperto di affidabilita' (scratchpad review/E2_affidabilita.md): il motore e'
ottimista al voto 3 (+0,86) e, per la taratura a componenti uniformi, di circa 0,3 in piu' sui ponti reali. La
rimappatura post-hoc (supplemento SW.6) tiene ferma la dinamica; qui la si lascia rispondere, in particolare la
soglia etica sul beta stimato. Sensibilita', non nuova configurazione del paper.

## Cosa si gioca
Identico a Z-princ (control/NOTA-campagna-Z.md: 147 ponti, 50 anni, soglia accesa, stessi semi, stesso motore
6098d88) salvo: capacity_condition_sensitivity 0,6 e reliability_resistance_scale 1,30 (residuo al voto 3 circa
+0,18, RMSE pesato circa 0,33 sul bersaglio senza phi_c). Norme D0 e D1. Gestori passivo, tipico, peggiore prima.
3 x 2 x 10 = 60 partite.

## Lettura fissata ora
Per ogni gestore: livelli con D0 e differenze appaiate D1 - D0 per seme, intervallo bootstrap al 95% (4000), su S1,
crolli attesi per resistenza, crolli idraulici attesi da difetti progressivi, ponti e anni-ponte sotto 3,02, spesa,
interventi della soglia (distinti fra beta stimato e difetto progressivo).
- R1. Tipico: D1 - D0 su S1 positivo in almeno 9 semi su 10. Se si', nel paper il segno "regge alla taratura".
- R2. Tipico: rapporto dei crolli attesi per resistenza D1/D0 fra 0,40 e 0,65. Se si', "dimezza" resta; altrimenti
  si scrive il rapporto osservato.
- R3. Passivo: si riporta D1 - D0 su S1 e quante volte la soglia agisce sul beta stimato. La frase "la soglia sul
  beta stimato agisce di rado" resta solo se gli interventi sul beta stimato sono in media meno di 1 per partita con
  D1; altrimenti si riscrive.
- R4. Segno di D1 - D0 sui crolli idraulici attesi per tutti e tre i gestori (atteso negativo: il canale non dipende
  dalla taratura del beta).
Ogni esito riportato nel supplemento, anche quando una regola non regge.
