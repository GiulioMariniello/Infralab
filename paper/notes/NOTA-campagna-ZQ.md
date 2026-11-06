# Campagna ZQ - intervallo d'ispezione imposto (nota scritta prima delle partite, 2026-11-06)

## Perche'
Richiesta dell'autore: vedere come il parco degrada al variare del tempo fra ispezioni imposte, da 6 mesi a 30 anni.
Esempio 1 confronta solo D0 (nessun obbligo), N24 (24 mesi, 12 sotto 0,39) e D1 (graduato); qui l'intervallo e'
la variabile, a parita' di tutto il resto.

## Cosa si gioca
Identico a Z-princ (motore 6098d88, stessi semi, 147 ponti, 50 anni, soglia accesa, taratura di base) salvo la norma.
Solo gestore tipico. Nove norme con intervallo fisso uguale per tutte le classi di condizione, stessa applicazione di
N24 (forzata, nessuna tolleranza, a carico del gestore): 6, 12, 24, 36, 48, 60, 120, 240, 360 mesi
(2, 4, 8, 12, 16, 20, 40, 80, 120 trimestri; file leggi/Q<trimestri>_fissa.yaml). 9 x 10 = 90 partite.
Riferimenti senza nuove partite: D0, N24 e D1 del gestore tipico in Z-princ (stessi semi).
Con 120 trimestri su 200 il parco riceve al piu' due ispezioni imposte per ponte; l'intervallo di 360 mesi e'
vicino a D0 per costruzione. I difetti idraulici progressivi maturano in media in 20 trimestri: oltre 60 mesi ci si
attende che l'obbligo li trovi di rado.

## Lettura fissata ora
Esiti per partita: S1, crolli attesi per resistenza (somma 1-(1-pf)^0,25 sul beta vero), crolli idraulici attesi
da difetti progressivi (0,05 x difetti maturati non arrestati), anni-ponte sotto 3,02, spesa, ispezioni imposte,
interventi della soglia. Differenze appaiate per seme rispetto a D0 di Z-princ, bootstrap 95% (4000).
- Q1. Si riporta per ogni intervallo livello e differenza da D0 di ogni esito (tabella e una figura, esito contro
  intervallo, scala logaritmica dei mesi). Nessun intervallo viene scelto come "ottimo".
- Q2. La frase "il guadagno cala al crescere dell'intervallo" si scrive per un esito solo se la differenza media da
  D0 non migliora mai passando a un intervallo piu' lungo oltre l'ampiezza dei due intervalli bootstrap
  (sovrapposizione ammessa). Altrimenti si scrive dove la curva non e' monotona.
- Q3. Si riporta il piu' lungo intervallo la cui differenza da D0 esclude lo zero, separatamente per S1, crolli
  attesi per resistenza e crolli idraulici attesi.
- Q4. Controllo: Q8 (24 mesi fissi) e N24 differiscono solo per i ponti sotto 0,39; S1 di Q8 - N24 atteso entro
  +-0,01. Se fuori, si cerca la causa prima di leggere il resto.
Ogni esito riportato nel supplemento; nel paper al piu' un paragrafo e una figura nell'Esempio 1, con il permesso
dell'autore.
