# Campagna ZI - sensibilita' del canale idraulico progressivo (nota scritta prima delle partite, 2026-11-05)

## Perche'
Esperto di idraulica (scratchpad review/E1_idraulica.md): il beneficio principale delle ispezioni nel gioco passa
dai difetti idraulici progressivi, scoperti da un'ispezione L1 con probabilita' 0,5 (parametro PROVISIONAL, giudicato
ottimistico per lo scalzamento sott'acqua). Sullo stesso canale D1 toglie 0,06 [0,035; 0,09] crolli idraulici attesi
in piu' di N24 per il gestore tipico. Si controlla quanto queste due affermazioni dipendano dalla probabilita' di scoperta.

## Cosa si gioca
Identico a Z-princ (stesso motore 6098d88, stessi semi) salvo progressive_hydraulic_detect_factor_l1 = 0,2 e 0,8
(0,5 e' Z). Solo gestore tipico. Norme D0, N24, D1. 2 x 3 x 10 = 60 partite.
La quota del rischio assegnata al canale progressivo (0,5) non si varia: con quota 0 il canale sparisce, con quota 1
l'effetto scala circa in proporzione; lo si dice nel testo.

## Lettura fissata ora
Crolli idraulici attesi da difetti progressivi = 0,05 x difetti maturati non arrestati, per partita. Differenze
appaiate per seme, bootstrap 95% (4000), a ogni probabilita' di scoperta (0,2; 0,5 da Z; 0,8).
- I1. D1 - D0 e N24 - D0 negativi in almeno 9 semi su 10 a ogni probabilita'. Se si', "l'obbligo toglie la maggior
  parte dei crolli idraulici attesi" resta, con l'ampiezza a 0,2 e 0,8.
- I2. D1 - N24: si riporta segno e intervallo a 0,2 e 0,8. La frase "il graduato scopre piu' difetti" resta solo se
  l'intervallo esclude lo zero a tutte e tre le probabilita'; altrimenti si scrive per quali.
- Si riportano anche S1, crolli attesi per resistenza, spesa, interventi della soglia.
Ogni esito riportato nel supplemento.
