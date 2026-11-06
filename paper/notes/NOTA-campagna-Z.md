# Campagna Z - le partite del paper con la taratura v3 e il limite operativo (nota scritta prima delle partite, 2026-11-02)

## Perché
Revisione di sei strutturisti (control/review-structural.json, 2026-10-16), tutti revisione maggiore. La campagna Y
(control/NOTA-campagna-Y.md) correggeva solo la base temporale degli ancoraggi e non risolveva:
1. il limite 3,66 (2,5 su 50 anni) e' piu' severo dell'ancoraggio operativo 3,02 (2,5 su 5 anni, NCHRP 454): un ponte
   con RF operativo = 1, legale e aperto, finiva sotto il limite;
2. il fattore di valutazione LRFR contiene il fattore di condizione phi_c, che e' un fattore di norma e non capacita'
   persa;
3. D0 (nessuna ispezione) non e' il caso reale: negli Stati Uniti vale l'ispezione ogni 24 mesi (NBIS).
Decisione dell'autore (2026-11-02, "go"): Y si ferma (15 partite giocate, non usate); si riscrivono le note; si gioca Z.

## Cosa cambia rispetto a Y
- Taratura v3 (taratura_beta_nbi_v3.py, leggi_degrado/taratura_beta_nbi_v3.json): bersaglio per voto con RF / phi_c
  (MBE 6A.4.2.3: 1,00 per voto >= 6, 0,95 per voto 5, 0,85 per voto <= 4), ancoraggi NCHRP 454 (progetto 3,5 su 75
  anni = 4,52 annuo; operativo 2,5 su 5 anni = 3,02 annuo), stessa griglia di v2 (k_c 0-1, rho 1,0-2,4; il minimo e'
  interno), stesso criterio (scarto quadratico pesato per numero di ponti). Migliore coppia k_c 0,4 e rho 1,25:
  scarto 0,30 pesato, 0,42 non pesato. Residui per voto 3..9: +0,86 / -0,25 / +0,04 / +0,04 / +0,41 / -0,36 / -0,33.
  Il motore resta ottimista al voto 3 (58 ponti su 18.433): si dichiara. Sensibilita' calcolate e non giocate:
  ancoraggio su 2 anni 0,5 e 1,3; con phi_c dentro 0,6 e 1,3 (= Y).
- Limite e soglia: beta annuo 3,02 (livello operativo). La soglia etica legge la stima del gestore a 3,02. I conteggi
  "sotto il limite" si riportano anche a 3,30 e 3,66 come sensibilita' (calcolati dalle stesse partite).
- Norma N24 (leggi/N24_nbis.yaml): ispezione ogni 8 trimestri, ogni 4 per condizione < 0,39 (voto <= 3), sul modello di
  23 CFR 650.311(a). Aggiunta in princ e off.
- Motore: commit con le opzioni di scomposizione (spente di default, snapshot TRAIN-01 identico); per Z nessuna di
  esse e' accesa.
Al via: parco di 147 ponti, beta annuo vero mediano 4,66, nessun ponte sotto 3,02 (ne' sotto 3,66); partita corta
mediana 4,73, nessuno sotto.
Totale 350 partite (princ 90, off 90, corta 50, markov 40, potenza 40, gamma 40), stessi semi di W e Y.

## Lettura fissata ora
- Esempio 1: per ogni gestore e ogni norma (D0, D1, N24) differenze appaiate D1 - D0, N24 - D0 e D1 - N24 su: S1,
  crolli attesi, anni-ponte sotto il limite, ponti sotto il limite almeno una volta, spesa, quota chiusa, interventi
  della soglia; intervalli al 95% sui semi e ricampionando ponti e semi. Il confronto principale per il lettore e'
  D1 - N24 (norma graduata contro riferimento a 24 mesi). Per il passivo: si scrive che "l'obbligo vale quanto cio'
  che segue un ritrovamento" solo se D1 - D0 su S1 con la soglia spenta sta entro 0,10 da zero e con la soglia accesa
  e' almeno 0,10 sopra zero (stessa regola di W e Y). Altrimenti si riportano i numeri e basta.
- Esempio 2: si riportano le differenze assolute del passivo per legge con intervallo, non il rapporto. Verso: positivo
  in almeno 9 semi su 10. Si dichiara che le leggi differiscono per usura media e che la proiezione del gestore segue
  sempre la curva base.
- Esempio 3: differenze appaiate dal passivo, nessuna classifica; confronti fra giocatori diversi dal passivo come
  analisi successive dichiarate.
Nessuna attesa sul verso. Z diventa la configurazione del paper qualunque sia l'esito; W e Y (non giocata) vanno nel
supplementare come storia della taratura.

## Limiti noti prima di giocare
1. Ancoraggi da NCHRP 454; 5 anni per il livello operativo e' una scelta.
2. Confronto trasversale; taratura sui 30 ponti della partita corta applicata al parco di 147; voto 3 ottimista.
3. Conversione fra periodi con anni indipendenti e resistenza costante: sovrastima il beta annuo equivalente.
4. N24 approssima NBIS: niente ispezioni subacquee, di elementi critici a frattura, ne' piani d'azione per lo scalzamento.
5. Le leggi dell'Esempio 2 non sono allineate sull'usura media.
