# Calibrazione del degrado su dati NBI reali — Delaware

Fonte: FHWA National Bridge Inventory, anni 2016–2023 (8 rilevazioni annuali dello stesso parco).

- Ponti distinti: **674**
- Osservazioni (ponte × anno × componente) usate: **15467**
- Vita utile assunta L = 100 anni · esponente globale = 3.0

## Parametri calibrati vs default del modello

Curva: C(t) = exp(−scale·(t/L)^k). Fit per linearizzazione Weibull-plot (ln(−ln C) vs ln(t/L)); R² = bontà della retta.

| Componente | scale (fit → default) | k (fit → default) | shape_factor calibrato | R² | n |
|---|---|---|---|---|---|
| deck (impalcato) | 0.3573 → 0.95 | 0.2138 → 3.15 | 0.0713 (def 1.05) | 0.1769 | 4974 |
| superstructure (appoggi) | 0.4198 → 1.12 | 0.3024 → 4.05 | 0.1008 (def 1.35) | 0.2416 | 5114 |
| substructure (pile) | 0.4663 → 1.1 | 0.2875 → 3.6 | 0.0958 (def 1.2) | 0.3444 | 5147 |

## Tasso di deterioramento osservato (longitudinale)

Anni medi per perdere **un livello** di condition rating, sui ponti in declino netto: (anni fra prima e ultima misura) / (livelli persi), aggregato:

- deck: **5.0** anni/livello
- superstructure: **5.61** anni/livello
- substructure: **5.76** anni/livello

## Interpretazione

Il fattore di forma osservato (k medio ≈ 0.27) è nettamente inferiore al default del modello (k medio ≈ 3.60): la traiettoria reale dei rating NBI è più concava (degrado front-loaded, decelera in età normalizzata). Il modello di default, con k alto, concentra il degrado a fine vita (curva a "scogliera") più di quanto i dati mostrino. L'R² moderato riflette la granularità dei rating (interi 0–9) e l'eterogeneità del parco: la calibrazione va letta come ancoraggio dell'ordine di grandezza, non come fit puntuale.

## Note metodologiche

- Rating NBI interi 0–9 mappati a C = r/9; usati solo 0<C<1 ed età>0.
- Longitudinale: lo stesso ponte compare a età diverse negli anni, infittendo il campionamento della curva e attenuando l'eterogeneità non osservata.
- Limite: L=100 assunto (non stimato congiuntamente); i rating aggregano modalità di degrado eterogenee; nessuna correzione per interventi di manutenzione intercorsi (un rating che risale indica manutenzione ed è escluso dai decrementi).
