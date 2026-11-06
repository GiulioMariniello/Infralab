"""Campagne B2 (B v2) e B2abl (ablazione): esecuzione a blocchi, replay esatto del lab runner + store per-ponte.

Derivato da phase1/T2/replay.py e phase1/T1/replay.py (replay validati 20/20 e 180/180).
Payload identici al lab runner: load_spec -> resolve_scenario -> expand_runs; ogni run e'
una copia di infralab.lab.runner.run_one (motore fbd6026: stesse righe runs/steps, stesse
colonne) con letture di sola lettura dello stato vero dei ponti (nessun RNG toccato):
  - ogni trimestre (0..H): beta_annual vero per ponte (come T2: posterior_annual_reliability_index
    se non None, altrimenti annual_reliability_index);
  - ogni anno (trimestre 0, 4, ..., H): condizione, rating, fattore di restrizione, posting,
    chiusura per soglia etica, sanzione di regime (come T1).

Campagne (--campaign, default B2): B2 = B2-<stock>.yaml (147 ponti, 50 anni, 9 policy);
B2abl = B2abl-<stock>.yaml (30 ponti, 20 anni, 9 policy). Ogni campagna ha la sua cartella
blocks/<campagna>/ e il suo piano.

Blocchi: per ogni parco, le run sono raggruppate per replica (seed); un blocco = seeds_per_block
seed completi x tutte le policy. Il piano e' congelato in blocks/<campagna>/plan.json alla prima
esecuzione insieme allo sha256 delle spec e al roster: se una spec cambia, il piano viene
rifiutato (nessuna mescolanza fra roster/impostazioni diversi).

Struttura:  blocks/<campagna>/<stock>/bNN/
              _work/<tag>.pkl            riga run + righe step (scrittura atomica, = run completata)
              perbridge/<tag>_beta.parquet, <tag>_annual.parquet
              runs.csv, steps.csv, timing.csv   (alla chiusura del blocco)
              DONE.json                  marcatore atomico di blocco completato
            blocks/<campagna>/<stock>/bNN.lock      claim del blocco (pid); se il pid e' morto il lock e' stantio
Un blocco interrotto riparte riusando le run gia' completate in _work/.

Uso:
  python run_blocks.py [--campaign B2abl] --plan                    # mostra il piano blocchi
  python run_blocks.py --block vecchio-b03       # un blocco (anche --block 7 = indice globale)
  python run_blocks.py --next                    # il prossimo blocco pendente
  python run_blocks.py --all                     # tutti i blocchi pendenti, in sequenza
  python run_blocks.py --smoke vecchio:30,60 --out DIR   # payload scelti per run_id (test)
Opzioni: --workers N (max 8; workers.txt lo sostituisce). Tutti i processi a priorita' IDLE.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.environ.get("RB_REPO", r"<repository root at the engine commit>")  # RB_REPO: albero di lavoro a un commit precedente
ENGINE_COMMIT = "94dc303"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_WORKERS = 8
CAMPAIGNS = {
    # campagna C (T4): 1 seme x 36 celle (3 policy x 3 obblighi x 2 soglie x 2 durate) = 36 run per blocco
    "C": {"seeds_per_block": 1, "specs": {"vecchio": "C-vecchio.yaml"}},
    # campagna dimostrativa D (verifica di una norma): 12 celle per seme (3 gestori x 4 leggi);
    # blocchi di 3 semi = 36 run; braccio principale (10 semi) e di controllo (5 semi, canale spento)
    "D": {"seeds_per_block": 3, "specs": {"princ": "D-princ.yaml", "ctrl": "D-ctrl.yaml"}},
    # S: sensibilita' un fattore alla volta (2 gestori x 2 leggi x 5 semi = 20 run per variante, semi di D)
    "S": {"seeds_per_block": 5, "specs": {"flooroff": "S-floor-off.yaml", "floor01": "S-floor01.yaml",
                                           "floor04": "S-floor04.yaml", "danni10": "S-danni10.yaml",
                                           "eventi4": "S-eventi4.yaml"}},
    # M: margine fra MCTS onesto e MCTS con stato vero, 15 ponti x 10 anni, 10 semi; base = policy semplici
    "M": {"seeds_per_block": 5, "specs": {"mcts": "M-mcts.yaml", "base": "M-base.yaml"}},
    # P: pilota esplorativo del pavimento etico sul beta (40 ponti, 50 anni, 1 seme), escluso dalle analisi
    "P": {"seeds_per_block": 1, "specs": {"beta": "P-beta.yaml"}},
    # F: effetto di D1 al variare del pavimento etico sul beta (40 ponti, 50 anni, 10 semi); off = pavimento spento
    "F": {"seeds_per_block": 2, "specs": {"beta": "F-beta.yaml", "off": "F-off.yaml"}},
    # G: le norme di D sopra la configurazione tarata (control/NOTA-campagna-G.md); 12 celle per seme,
    # motore con le opzioni aggiunte dopo 94dc303 (le campagne precedenti restano su 94dc303)
    "H": {"seeds_per_block": 5, "specs": {"flooroff": "H-flooroff.yaml"}},
    "G": {"seeds_per_block": 2, "specs": {"princ": "G-princ.yaml"}, "engine": "8d60f2f"},
    # J: l'obbligo di ispezione sopra la configurazione regolata sul Delaware (control/NOTA-campagna-J.md)
    "J": {"seeds_per_block": 5, "specs": {"princ": "J-princ.yaml"}, "engine": "8d60f2f"},
    # B: evidenze in risposta alla revisione (control/NOTA-evidenze-revisione.md), tutte su una versione del motore
    "B": {"seeds_per_block": 5, "engine": "98a61a4",
          "specs": {"base": "B0-base.yaml", "risposta": "B1-risposta.yaml", "b2a": "B2a-soglia005.yaml",
                    "b2b": "B2b-ispezione8.yaml", "b2c": "B2c-pratica.yaml", "b2d": "B2d-eventi.yaml",
                    "b2e": "B2e-stima.yaml", "gamma": "B3-gamma.yaml"}},
    # N: dimostrazione dei giocatori (control/NOTA-dimostrazione-giocatori.md): una partita per giocatore,
    # 30 ponti x 10 anni, legge D1, un seme; non e' un confronto
    "N": {"seeds_per_block": 1, "engine": "98a61a4", "specs": {"giocatori": "N-giocatori.yaml"}},
    # K, N2: chiusura per la sottomissione (control/NOTA-chiusura.md)
    "K": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"f2x2": "K-f2x2.yaml", "biennale": "K-biennale.yaml"}},
    "N2": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"alg": "N2-alg.yaml", "llm": "N2-llm.yaml"}},
    # R: gestore tipico sulla prima configurazione, soglia accesa e spenta (control/NOTA-campagna-R.md)
    "R": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"on": "R-on.yaml", "off": "R-off.yaml"}},
    # U: partita corta della sessione con giocatori umani (control/NOTA-sessione-umana.md), algoritmi su 10 semi
    "U": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"alg": "U-alg.yaml"}},
    # L: legge di degrado a potenza scritta come file utente (control/NOTA-campagna-L.md), stessa specifica di B3
    "L": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"potenza": "L-potenza.yaml"}},
    # V: legge di degrado a catena di Markov tarata sull'inventario federale, file utente (control/NOTA-campagna-V.md)
    "V": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"markov": "V-markov.yaml"}},
    # W: gli esempi del paper sulla configurazione tarata sulla capacita' valutata (control/NOTA-campagna-W.md)
    "W": {"seeds_per_block": 5, "engine": "98a61a4", "specs": {"princ": "W-princ.yaml", "off": "W-off.yaml", "corta": "W-corta.yaml",
                                                         "markov": "W-markov.yaml", "potenza": "W-potenza.yaml", "gamma": "W-gamma.yaml"}},
    # Y (control/NOTA-campagna-Y.md): W con la taratura corretta (ancoraggio operativo su 5 anni), k_c 0,6 e rho 1,3
    "Y": {"seeds_per_block": 5, "engine": "4714fa5", "specs": {"corta": "Y-corta.yaml", "princ": "Y-princ.yaml", "off": "Y-off.yaml",
                                                         "markov": "Y-markov.yaml", "potenza": "Y-potenza.yaml", "gamma": "Y-gamma.yaml"}},
    # Z (control/NOTA-campagna-Z.md): taratura v3 (phi_c tolto), k_c 0,4 e rho 1,25, soglia 3,02, norma N24 in princ/off
    "Z": {"seeds_per_block": 5, "engine": "6098d88", "specs": {"corta": "Z-corta.yaml", "princ": "Z-princ.yaml", "off": "Z-off.yaml",
                                                         "markov": "Z-markov.yaml", "potenza": "Z-potenza.yaml", "gamma": "Z-gamma.yaml"}},
    # ZT (control/NOTA-campagna-ZT.md): Z-princ con k_c 0,6 e rho 1,30, D0/D1 (sensibilita' della taratura)
    "ZT": {"seeds_per_block": 5, "engine": "6098d88", "specs": {"princ": "ZT-princ.yaml"}},
    # ZI (control/NOTA-campagna-ZI.md): Z-princ, gestore tipico, scoperta L1 dei difetti idraulici 0,2 e 0,8
    "ZI": {"seeds_per_block": 5, "engine": "6098d88", "specs": {"det02": "ZI-det02.yaml", "det08": "ZI-det08.yaml"}},
    # ZQ (control/NOTA-campagna-ZQ.md): Z-princ, gestore tipico, intervallo d'ispezione fisso da 6 a 360 mesi
    "ZQ": {"seeds_per_block": 5, "engine": "6098d88", "specs": {"int": "ZQ-int.yaml"}},
}
CAMPAIGN = BLOCKS = STOCKS = SEEDS_PER_BLOCK = None


def set_campaign(name: str) -> None:
    global CAMPAIGN, BLOCKS, STOCKS, SEEDS_PER_BLOCK, ENGINE_COMMIT
    if name not in CAMPAIGNS:
        sys.exit(f"campagna sconosciuta: {name} (note: {list(CAMPAIGNS)})")
    CAMPAIGN = name
    BLOCKS = os.path.join(HERE, "blocks", name)
    STOCKS = dict(CAMPAIGNS[name]["specs"])
    SEEDS_PER_BLOCK = CAMPAIGNS[name]["seeds_per_block"]
    ENGINE_COMMIT = CAMPAIGNS[name].get("engine", "94dc303")


set_campaign("C")


# ----------------------------------------------------------------------------- utilita'
def _now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _idle_priority() -> None:
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, 0x40)  # IDLE_PRIORITY_CLASS


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_df(df, path: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    if path.endswith(".parquet"):
        df.to_parquet(tmp, index=False, compression="zstd")
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def check_engine() -> str:
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if not head.startswith(ENGINE_COMMIT) or dirty:
        sys.exit(f"motore non conforme: HEAD={head[:7]} (atteso {ENGINE_COMMIT}), dirty={bool(dirty)}")
    return head


# ----------------------------------------------------------------------------- payload e piano
def build_payloads(stock: str):
    from infralab.lab.runner import expand_runs
    from infralab.lab.spec import load_spec, resolve_scenario
    spec = load_spec(os.path.join(HERE, STOCKS[stock]))
    payloads = expand_runs(spec, resolve_scenario(spec.scenario))
    for p in payloads:
        # il gestore tipico non conosce il seme della partita: gli si passa quello della replica
        cfg = p.get("agent_config") or {}
        if cfg.get("policy") == "typical_owner":
            cfg.setdefault("params", {})["seed"] = int(p["seed"])
    return payloads


def tag(payload) -> str:
    # il run_id distingue i punti di griglia (stessa policy e replica, legge/soglia/durata diverse)
    return f"{payload['agent_id']}_r{payload['replicate']}_id{payload['run_id']:03d}"


def _spec_fingerprint() -> dict:
    import hashlib
    from infralab.lab.spec import load_spec
    out = {}
    for stock, fname in STOCKS.items():
        path = os.path.join(HERE, fname)
        spec = load_spec(path)
        out[stock] = {"spec": fname, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                      "agents": [a.id for a in spec.agents], "replicates": spec.replicates,
                      "run_count": spec.run_count}
    return out


def make_plan() -> dict:
    """Piano congelato: per parco, blocchi di seed consecutivi; ordine globale alternato fra parchi.
    Rifiuta un piano esistente se le spec (sha256) o il roster non coincidono piu'."""
    path = os.path.join(BLOCKS, "plan.json")
    fp = _spec_fingerprint()
    if os.path.exists(path):
        plan = json.load(open(path))
        if plan.get("campaign") != CAMPAIGN or plan.get("specs") != fp:
            sys.exit(f"{path}: piano non coerente con le spec attuali (roster/impostazioni cambiate). "
                     f"Se nessun blocco e' stato eseguito, cancellare blocks/{CAMPAIGN} e rigenerare.")
        return plan
    seeds_per_block = SEEDS_PER_BLOCK
    per_stock = {}
    for stock in STOCKS:
        spec = fp[stock]
        reps = list(range(spec["replicates"]))
        per_stock[stock] = [reps[i:i + seeds_per_block] for i in range(0, len(reps), seeds_per_block)]
    order = []
    for k in range(max(len(v) for v in per_stock.values())):
        for stock in STOCKS:
            if k < len(per_stock[stock]):
                order.append({"name": f"{stock}-b{k:02d}", "stock": stock, "k": k,
                              "replicates": per_stock[stock][k]})
    counts = {stock: [p["replicate"] for p in build_payloads(stock)] for stock in STOCKS}
    for b in order:
        b["n_runs"] = sum(1 for r in counts[b["stock"]] if r in b["replicates"])
    plan = {"campaign": CAMPAIGN, "created": _now(), "engine": ENGINE_COMMIT,
            "seeds_per_block": seeds_per_block, "specs": fp, "blocks": order}
    os.makedirs(BLOCKS, exist_ok=True)
    _atomic_write_bytes(path, json.dumps(plan, indent=1).encode())
    return plan


def block_dir(b) -> str:
    return os.path.join(BLOCKS, b["stock"], f"b{b['k']:02d}")


def is_done(b) -> bool:
    return os.path.exists(os.path.join(block_dir(b), "DONE.json"))


# ----------------------------------------------------------------------------- una run
def _snap_beta(env, q, rows):
    for b in env.bridges.values():
        ba = b.posterior_annual_reliability_index
        if ba is None:
            ba = b.annual_reliability_index
        rows.append({"quarter": q, "bridge": int(b.id), "area": float(b.area),
                     "beta_annual": float("nan") if ba is None else float(ba)})


def _snap_annual(env, year, rows):
    import numpy as np
    q = int(env.quarter)
    for b in env.bridges.values():
        c = float(b.min_condition)
        rows.append({
            "year": year, "quarter": q, "bridge": int(b.id), "area": float(b.area),
            "condition": c, "rating": int(np.clip(round(c * 9.0), 0, 9)),
            "factor": float(getattr(b, "traffic_restriction_factor", 1.0)),
            "posting_active": int(int(getattr(b, "posting_until_quarter", 0) or 0) > q),
            "floor_closed": int(bool(getattr(b, "floor_closure", None))),
            "regime_sanction": int(bool(getattr(b, "regime_sanction", None))),
        })


def replay_one(payload: dict):
    """Copia di infralab.lab.runner.run_one (fbd6026) + snapshot di sola lettura."""
    from infralab.agents.adapters import make_agent
    from infralab.agents.protocol import AssetEnvGym, RewardConfig
    from infralab.lab.build import build_env_from_scenario

    param_overrides = {k[len("params."):]: v for k, v in payload["grid_point"].items()}
    reward_config = RewardConfig(**payload["reward"]) if payload["reward"] else RewardConfig()

    def env_factory(seed):
        return build_env_from_scenario(payload["scenario"], agent_name=payload["agent_id"],
                                       agent_config=payload["agent_config"],
                                       param_overrides=param_overrides, seed=seed)

    gym = AssetEnvGym(env_factory, payload["agent_id"], reward_config=reward_config,
                      max_quarters=payload["horizon_quarters"])
    obs, info = gym.reset(seed=payload["seed"])
    agent = make_agent(payload["agent_id"], payload["agent_config"])
    agent.reset(obs, {**info, "env": gym.harness_env()})

    base = {"run_id": payload["run_id"], "agent_id": payload["agent_id"],
            "replicate": payload["replicate"], "seed": payload["seed"], **payload["grid_point"]}
    beta, annual = [], []
    _snap_beta(gym.harness_env(), 0, beta)
    _snap_annual(gym.harness_env(), 0, annual)
    steps = []
    planned = {}
    agent_errors, empty_plans, malformed = [], 0, 0
    total_reward = 0.0
    terminated = truncated = False
    t_start = time.perf_counter()
    while not (terminated or truncated):
        action = agent.act(obs)
        # conteggio (sola lettura) delle azioni richieste dalla policy, per nome: serve a separare
        # le ispezioni scelte dal gestore da quelle imposte dalla legge (RT028)
        _plan = action.get("maintenance", action) if isinstance(action, dict) else {}
        for _name in (_plan.values() if isinstance(_plan, dict) else []):
            if isinstance(_name, str):
                planned[_name] = planned.get(_name, 0) + 1
        # conteggio (sola lettura) dei turni in cui un agente LLM ha fallito la chiamata o non ha dato un piano
        _rat = getattr(agent, "last_rationale", None)
        if isinstance(_rat, tuple) and len(_rat) == 2 and isinstance(_rat[1], str) and _rat[1].startswith("Errore"):
            agent_errors.append(_rat[1][:160])
        if not _plan:
            empty_plans += 1
        # voci di piano con chiave non numerica (un agente LLM puo' scrivere '113_preventive_ordinaria'): il motore
        # si ferma su int(); qui la voce viene scartata e contata, il resto del piano passa invariato
        if isinstance(_plan, dict):
            _bad = [k for k in _plan if not str(k).strip().lstrip("-").isdigit()]
            for k in _bad:
                del _plan[k]
            malformed += len(_bad)
        obs, reward, terminated, truncated, info = gym.step(action)
        total_reward += reward
        steps.append({**base, "quarter": info["quarter"] - 1, "reward": reward,
                      **{f"metric_{k}": v for k, v in info["metrics"].items()}})
        n = len(steps)
        env = gym.harness_env()
        _snap_beta(env, n, beta)
        if n % 4 == 0:
            _snap_annual(env, n // 4, annual)
    failure = info.get("failure_info") or {}
    run_row = {**base, "quarters_survived": len(steps),
               "duration_s": round(time.perf_counter() - t_start, 3),
               "total_reward": total_reward, "terminated": terminated, "truncated": truncated,
               "failure_reason": failure.get("reason") if terminated else None,
               **{f"final_{k}": v for k, v in info["metrics"].items()}}
    from infralab.lab.runner import _collapse_rows
    # colonne aggiuntive (extra_*), in coda e di sola lettura: non alterano le colonne del runner
    env = gym.harness_env()
    for _k, _v in sorted(planned.items()):
        run_row[f"extra_policy_planned_{_k}"] = _v
    run_row["extra_agent_errors"] = len(agent_errors)
    run_row["extra_agent_first_error"] = agent_errors[0] if agent_errors else ""
    run_row["extra_empty_plans"] = empty_plans
    run_row["extra_malformed_entries"] = malformed
    for _label in ("regime_stats", "progressive_stats", "floor_stats"):
        _st = (getattr(env, _label, None) or {}).get(payload["agent_id"]) or {}
        for _k, _v in _st.items():
            if isinstance(_v, (int, float, bool)) and f"final_{_label.split('_')[0]}_{_k}" not in run_row:
                run_row[f"extra_{_label}_{_k}"] = _v
    return run_row, steps, beta, annual, _collapse_rows(env, base)


def _task(args):
    """Esegue una run in un worker; il pickle in _work/ e' il marcatore di run completata."""
    payload, stock, bdir = args
    _idle_priority()
    import pandas as pd
    t = tag(payload)
    work = os.path.join(bdir, "_work", t + ".pkl")
    if os.path.exists(work):
        return t, "exists", 0.0
    t0 = time.time()
    run_row, steps, beta, annual, collapses = replay_one(payload)
    ident = {"stock": stock, "agent_id": payload["agent_id"], "replicate": payload["replicate"],
             "seed": payload["seed"], "run_id": payload["run_id"]}
    pb = os.path.join(bdir, "perbridge")
    _atomic_df(pd.DataFrame(beta).assign(**ident), os.path.join(pb, t + "_beta.parquet"))
    _atomic_df(pd.DataFrame(annual).assign(**ident), os.path.join(pb, t + "_annual.parquet"))
    sec = round(time.time() - t0, 1)
    _atomic_write_bytes(work, pickle.dumps({"run": run_row, "steps": steps, "collapses": collapses, "seconds": sec,
                                            "host": socket.gethostname()}))
    return t, "done", sec


def run_payloads(payloads, stock, bdir, workers, log=print):
    os.makedirs(os.path.join(bdir, "_work"), exist_ok=True)
    os.makedirs(os.path.join(bdir, "perbridge"), exist_ok=True)
    for f in os.listdir(os.path.join(bdir, "perbridge")):     # residui di scritture interrotte
        if ".tmp" in f:
            os.remove(os.path.join(bdir, "perbridge", f))
    todo = [p for p in payloads if not os.path.exists(os.path.join(bdir, "_work", tag(p) + ".pkl"))]
    # le run Euristico (~2-2.5x piu' lente) per prime: meno coda a fine blocco; non tocca i risultati
    todo.sort(key=lambda p: p["agent_id"] != "heuristic")
    items = [(p, stock, bdir) for p in todo]
    log(f"  {len(payloads) - len(todo)} run gia' completate, {len(todo)} da eseguire, {workers} worker")
    if workers == 1:
        for it in items:
            log("  " + " ".join(map(str, _task(it))))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers, initializer=_idle_priority) as pool:
            futs = [pool.submit(_task, it) for it in items]
            for f in as_completed(futs):
                t, st, sec = f.result()
                log(f"  {t} {st} {sec} s")


def finalize(payloads, bdir, extra: dict):
    """Concatena le run del blocco in runs.csv / steps.csv (come lab runner) + timing + DONE."""
    import pandas as pd
    runs, steps, timing, collapses = [], [], [], []
    for p in payloads:
        d = pickle.load(open(os.path.join(bdir, "_work", tag(p) + ".pkl"), "rb"))
        runs.append(d["run"])
        steps.extend(d["steps"])
        collapses.extend(d.get("collapses") or [])
        timing.append({"tag": tag(p), "run_id": p["run_id"], "seconds": d["seconds"]})
    runs.sort(key=lambda r: r["run_id"])
    steps.sort(key=lambda r: (r["run_id"], r["quarter"]))
    _atomic_df(pd.DataFrame(runs), os.path.join(bdir, "runs.csv"))
    _atomic_df(pd.DataFrame(steps), os.path.join(bdir, "steps.csv"))
    _atomic_df(pd.DataFrame(timing), os.path.join(bdir, "timing.csv"))
    from infralab.lab.runner import COLLAPSE_COLUMNS
    ccols = ["run_id", "agent_id", "replicate", "seed"] + [c for c in (collapses[0] if collapses else {}) if c.startswith("params.")] + list(COLLAPSE_COLUMNS)
    collapses.sort(key=lambda r: (r["run_id"], r["quarter"], r["bridge_id"]))
    _atomic_df(pd.DataFrame(collapses, columns=ccols if collapses else ["run_id", "agent_id", "replicate", "seed"] + list(COLLAPSE_COLUMNS)), os.path.join(bdir, "collapses.csv"))
    done = {**extra, "finished": _now(), "n_runs": len(runs),
            "run_seconds_sum": round(sum(t["seconds"] for t in timing), 1)}
    _atomic_write_bytes(os.path.join(bdir, "DONE.json"), json.dumps(done, indent=1).encode())
    shutil.rmtree(os.path.join(bdir, "_work"), ignore_errors=True)


# ----------------------------------------------------------------------------- blocchi
def claim(b) -> str | None:
    lock = block_dir(b) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                     "since": _now()}).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                info = json.load(open(lock))
                stale = info.get("host") == socket.gethostname() and not _pid_alive(int(info["pid"]))
            except (ValueError, OSError, KeyError):
                stale = True
            if not stale:
                return None
            os.remove(lock)
    return None


def run_block(b, workers, log=print) -> str:
    if is_done(b):
        return "done-already"
    lock = claim(b)
    if lock is None:
        return "claimed-elsewhere"
    try:
        if is_done(b):
            return "done-already"
        bdir = block_dir(b)
        payloads = [p for p in build_payloads(b["stock"]) if p["replicate"] in b["replicates"]]
        t0 = time.time()
        started = _now()
        log(f"[{started}] {CAMPAIGN} blocco {b['name']}: {len(payloads)} run (repliche {b['replicates']})")
        run_payloads(payloads, b["stock"], bdir, workers, log)
        wall = round(time.time() - t0, 1)
        if len(payloads) != b["n_runs"]:
            sys.exit(f"{b['name']}: {len(payloads)} payload, il piano ne prevede {b['n_runs']}")
        finalize(payloads, bdir, {"campaign": CAMPAIGN, "block": b["name"], "stock": b["stock"], "replicates": b["replicates"],
                                  "engine": ENGINE_COMMIT, "workers": workers, "started": started,
                                  "wall_seconds_last_session": wall, "host": socket.gethostname()})
        log(f"[{_now()}] {CAMPAIGN} blocco {b['name']} completato in {wall / 60:.1f} min")
        return "done"
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def pick(plan, sel: str):
    if sel.isdigit():
        return plan["blocks"][int(sel)]
    for b in plan["blocks"]:
        if b["name"] == sel:
            return b
    sys.exit(f"blocco sconosciuto: {sel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="C", choices=list(CAMPAIGNS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--block")
    g.add_argument("--next", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--smoke", help="STOCK:ID,ID,...  payload con questi run_id, eseguiti in --out (test)")
    ap.add_argument("--out", help="cartella per --smoke")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args()
    # workers.txt accanto allo script, se presente, sostituisce --workers dal blocco successivo
    # (il numero di worker non cambia i risultati: ogni run ha il suo seme).
    _wf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers.txt")
    if os.path.exists(_wf):
        try:
            a.workers = int(open(_wf, encoding="utf-8").read().strip())
        except ValueError:
            pass
    if not 1 <= a.workers <= MAX_WORKERS:
        sys.exit(f"workers fra 1 e {MAX_WORKERS}")
    _idle_priority()
    set_campaign(a.campaign)
    check_engine()

    if a.smoke:
        stock, ids = a.smoke.split(":")
        ids = {int(i) for i in ids.split(",")}
        ps = [p for p in build_payloads(stock) if p["run_id"] in ids]
        run_payloads(ps, stock, a.out, a.workers)
        finalize(ps, a.out, {"smoke": a.smoke})
        return 0

    plan = make_plan()
    if a.plan:
        for i, b in enumerate(plan["blocks"]):
            print(f"{i:2d} {CAMPAIGN} {b['name']} repliche={b['replicates']} run={b['n_runs']} "
                  f"{'DONE' if is_done(b) else 'pending'}")
        return 0
    if a.block:
        print(run_block(pick(plan, a.block), a.workers))
        return 0
    for b in plan["blocks"]:
        if is_done(b):
            continue
        st = run_block(b, a.workers)
        if st == "claimed-elsewhere":
            print(f"{b['name']}: in corso altrove, salto")
            continue
        if a.next:
            return 0
    print(f"[{_now()}] {CAMPAIGN}: nessun blocco pendente" if all(is_done(b) for b in plan["blocks"])
          else f"[{_now()}] nessun blocco eseguibile: i restanti sono in corso altrove")
    return 3 if a.next else 0   # --next: 3 = niente da fare (usato da _esegui.cmd)


if __name__ == "__main__":
    sys.exit(main())
