"""Libreria: API pubblica, strumenti condivisi, server MCP e giocatore LLM (con un modello finto)."""
import asyncio
import json

import pytest

import infralab
from infralab.toolkit import ToolError, Toolkit

KW = dict(law="regbench/D1", seed=1, horizon_years=1)


def test_cataloghi():
    assert "regbench/aged-30" in infralab.scenarios() and "regbench/aged-147" in infralab.scenarios()
    assert {"regbench/D0", "regbench/D1", "regbench/K0"} <= set(infralab.laws())
    assert set(infralab.configurations()) >= {"first", "second", "third"}
    assert "worst_first" in infralab.policies() and "do_nothing" in infralab.basic_policies()
    assert infralab.describe_law("regbench/D1")["enforcement"]["mode"] if isinstance(
        infralab.describe_law("regbench/D1").get("enforcement"), dict) else True


def test_env_passo_e_resoconto():
    env = infralab.make("regbench/aged-30", **KW)
    obs, info = env.reset()
    assert len(obs["my_bridges"]) == 30 and env.quarter == 1
    aid = str(obs["my_bridges"][0]["id"])
    obs, reward, term, trunc, info = env.step({aid: "L1_inspect", "999999": "L1_inspect"})
    rep = info["report"]
    assert rep["requested"] == 2 and len(rep["dropped"]) == 1 and rep["dropped"][0]["reason"] == "unknown_bridge"
    while not env.done:
        env.step({})
    s = env.summary()
    assert s["quarters"] == 4 and 0 < s["S1"] < 10 and s["expected_collapses"] >= 0 and "emissions_tco2" in s


def test_osservazione_compatta_senza_stato_vero():
    env = infralab.make("regbench/aged-30", **KW)
    env.reset()
    c = env.compact_observation()
    assert len(c["assets"]) == 30
    assert len(json.dumps(c)) < 40_000  # l'osservazione completa supera 1,5 MB
    a = c["assets"][0]
    assert set(a["known_condition_range"]) == {"impalcato", "pile", "spalle", "appoggi"}
    # la vista compatta e' costruita dalla sola osservazione: nessuna chiave dello stato vero
    testo = json.dumps(c)
    for vietata in ("true_", "latent", "hidden", "seed"):
        assert vietata not in testo


def test_run_riproduce_la_partita_del_paper():
    """Replica 0 della campagna N2 (gestore passivo, 30 ponti, 10 anni, legge D1): stessi numeri al bit."""
    seme = infralab.child_seeds(20261901, 10)[0]
    r = infralab.run("basic:do_nothing", "regbench/aged-30", law="regbench/D1", config="first", seed=seme, horizon_years=10)
    assert r["S1"] == pytest.approx(3.390360796928728, abs=1e-12)
    assert r["cash_spend"] == pytest.approx(3401550.01, abs=0.01)


def test_configurazione_predefinita_riproduce_la_campagna_W():
    """Configurazione tarata (predefinita dal 2026-10-15): replica 0 della partita corta della campagna W, gestore passivo."""
    seme = infralab.child_seeds(infralab.configuration("calibrated-short")["root_seed"], 10)[0]
    assert seme == 241429190
    r = infralab.run("passive", "regbench/aged-30", law="regbench/D1", config="calibrated-short", seed=seme)
    assert r["quarters"] == 20 and r["collapses"] == 0
    assert r["cash_spend"] == pytest.approx(113423.25, abs=0.01)


def test_run_con_funzione_utente():
    def ispeziona_tutto(obs):
        return {str(b["id"]): "L1_inspect" for b in obs["my_bridges"][:3]}
    r = infralab.run(ispeziona_tutto, "regbench/aged-30", **KW)
    assert r["quarters"] == 4 and r["cash_spend"] > 0


def test_compare_laws_appaiato():
    r = infralab.compare_laws("regbench/D0", "regbench/D1", policy="passive", n_seeds=2, horizon_years=2)
    assert len(r["seeds"]) == 2 and r["seeds"] == infralab.child_seeds(20261007, 2)
    d = r["difference_b_minus_a"]
    assert set(d) >= {"S1", "cash_spend", "emissions_tco2"} and len(d["S1"]) == 3
    assert d["cash_spend"][0] > 0  # la legge impone ispezioni, che il gestore paga


def test_legge_di_degrado_come_file(tmp_path, monkeypatch):
    # la legge utente si registra nella cartella dati corrente: qui una cartella temporanea
    import infralab.persistence as persistence
    monkeypatch.setattr(persistence, "DATA_DIR", str(tmp_path))
    assert {"weibull", "gamma", "regbench/power", "regbench/markov-nbi"} <= set(infralab.degradation_laws())
    base = infralab.run("passive", "regbench/aged-30", **KW)
    esiti = {d: infralab.run("passive", "regbench/aged-30", degradation=d, **KW)["S1"]
             for d in ("weibull", "regbench/power", "regbench/markov-nbi")}
    assert esiti["weibull"] == base["S1"]                      # il default non cambia
    assert len({round(v, 9) for v in esiti.values()}) == 3     # le leggi utente sono davvero usate
    mia = tmp_path / "mia_legge.py"
    mia.write_text("def degrade(ctx):\n    return ctx['condition'] - 0.01\n", encoding="utf-8")
    assert infralab.resolve_degradation(str(mia)) == "custom:mia_legge"
    assert infralab.run("passive", "regbench/aged-30", degradation=str(mia), **KW)["S1"] < base["S1"]
    with pytest.raises(KeyError):
        infralab.resolve_degradation("regbench/inesistente")


def test_toolkit_non_rilascia_lo_stato_vero_durante_la_partita():
    tk = Toolkit()
    rid = tk.new_run(horizon_years=1, seed=1)["run_id"]
    s = tk.score(rid)
    assert not s["done"] and "S1" not in s and "expected_collapses" not in s
    out = tk.step(rid, {}, quarters=8)
    while not out["done"]:
        out = tk.step(rid, {}, quarters=8)
    assert "S1" in out["score"] and out["score"]["done"]
    with pytest.raises(ToolError):
        tk.step(rid, {})
    with pytest.raises(ToolError):
        tk.observe("inesistente")


def test_server_mcp_in_processo():
    pytest.importorskip("mcp")
    from infralab.mcp_server import build_server
    srv = build_server()

    async def giro():
        nomi = {t.name for t in await srv.list_tools()}
        assert {"new_run", "observe", "asset_actions", "step", "score", "compare_laws", "list_laws"} <= nomi
        res = await srv.call_tool("new_run", {"horizon_years": 1, "seed": 1})
        return res

    res = asyncio.run(giro())
    testo = json.dumps(res, default=lambda o: getattr(o, "text", str(o)))
    assert "run_id" in testo


class _ModelloFinto:
    """Chiede le azioni del primo ponte, poi consegna un piano con una voce valida e una malformata."""
    provider = "finto"

    def __init__(self):
        self.turni = 0

    def run_turn(self, system, user_text, tools, execute, max_rounds=6):
        self.turni += 1
        stato = json.loads(user_text)
        primo = str(stato["assets"][0]["id"])
        if self.turni == 2:  # un turno fallito: nessun piano consegnato
            return {"finished": False, "error": "no tool call", "usage": {"input_tokens": 10, "output_tokens": 1, "calls": 1}}
        azioni = execute("asset_actions", {"asset_id": primo})["result"]["actions"]
        assert any(a["action"] == "L1_inspect" for a in azioni)
        out = execute("submit_plan", {"plan": {primo: "L1_inspect", "113_preventive": "x"}, "reason": "prova", "memo": "nota"})
        return {"finished": bool(out.get("finished")), "usage": {"input_tokens": 100, "output_tokens": 20, "calls": 2}}


def test_giocatore_llm_con_modello_finto():
    from infralab.llm import play
    r = play("finto:modello", "regbench/aged-30", law="regbench/D1", seed=1, horizon_years=1, backend=_ModelloFinto())
    assert r["done"] and r["quarters"] == 4 and r["decisions"] == 4
    assert r["failed_decisions"] == 1 and r["malformed_entries"] == 3
    assert r["usage"]["input_tokens"] == 310 and "S1" in r


def test_paper_configuration_outcomes():
    """The paper configuration is the default and the outcomes of the paper are reported."""
    import inspect
    assert inspect.signature(infralab.run).parameters["config"].default == "paper"
    assert "regbench/N24" in infralab.laws()
    r = infralab.run("passive", "regbench/aged-30", law="regbench/N24", config="paper-short", seed=0, horizon_years=3)
    for k in ("S1", "expected_collapses", "expected_hydraulic_collapses", "bridges_below_limit",
              "bridge_years_below_limit", "cash_spend", "inspections_forced_by_law"):
        assert k in r
    assert r["inspections_forced_by_law"] > 0
