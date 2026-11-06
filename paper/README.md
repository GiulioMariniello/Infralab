# Games of the paper

Record of the games behind the examples of the paper on InfraLab, engine commit 6098d88 (the engine of release 1.0.0).

- `specs/`: campaign specifications (stock, horizon, seeds, parameter grid, players). Campaign Z holds Examples 1
  and 2 (arms `princ`, `off`, `markov`, `potenza` (power), `gamma`, `corta` (short game)); ZT the sensitivity to the
  reliability calibration; ZI the sensitivity to the probability of detection of progressive hydraulic defects;
  ZQ the length of the inspection interval.
- `regulations/`, `scenarios/`: the regulation files and stocks the specifications point to
  (D0 = `D0_nessun_obbligo.yaml`, D1 = `D1_periodica.yaml`, N24 = `N24_nbis.yaml`, fixed intervals `Q<quarters>_fissa.yaml`).
- `notes/`: dated notes in which the reading of each set of games was fixed before play (in Italian).
- `analysis/`: block runner and analysis scripts (in Italian; they read the per-bridge series of the data archive).
- `results/`: one row per game (`*_runs.csv`, final metrics of the engine; `*_per_game.csv`, outcomes of the paper)
  and the analysis outputs (`analisi_*.json`).

The per-bridge, per-quarter series of every game are archived with the data DOI given in the paper.
Through the library the same games are played with `config="paper"` (Z princ), `"paper-short"` (Z corta) and
`"paper-less-optimistic"` (ZT); `tests/paper_reproduction_check.py` replays one game of each length and compares it
with the stored series.
