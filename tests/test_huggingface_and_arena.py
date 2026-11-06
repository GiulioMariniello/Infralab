"""Test per il provider Hugging Face e il modulo Arena."""
from unittest.mock import MagicMock, patch
import pytest

from infralab.agents.legacy import LLMAgent, agent_factory
from infralab.agents.agentic import AgenticLLMAgent
from infralab.lab.arena import run_single_simulation, DEFAULT_COMPETITORS


def test_llm_agent_hf_initialization():
    """Verifica che LLMAgent si inizializzi correttamente con il provider HuggingFace."""
    agent = LLMAgent(
        name="TestHF",
        profile="Bilanciato",
        provider="HuggingFace",
        model="meta-llama/Llama-3.3-70B-Instruct",
        hf_token="fake_token_test",
    )
    assert agent.provider == "HuggingFace"
    assert agent.hf_model == "meta-llama/Llama-3.3-70B-Instruct"
    assert agent.hf_token == "fake_token_test"


def test_agent_factory_hf():
    """Verifica che agent_factory istanzi correttamente LLMAgent con HuggingFace."""
    config = {
        "type": "LLM",
        "provider": "HuggingFace",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "profile": "Prudente",
        "hf_token": "token_xyz",
    }
    agent = agent_factory("AgentHF", config)
    assert isinstance(agent, LLMAgent)
    assert agent.provider == "HuggingFace"
    assert agent.hf_model == "Qwen/Qwen2.5-72B-Instruct"
    assert agent.hf_token == "token_xyz"


def test_agent_factory_agentic_hf():
    """Verifica che agent_factory istanzi correttamente AgenticLLMAgent con HuggingFace."""
    config = {
        "type": "LLM Agentico",
        "provider": "HuggingFace",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "profile": "Aggressivo",
        "hf_token": "token_abc",
        "max_rounds": 2,
    }
    agent = agent_factory("AgenticHF", config)
    assert isinstance(agent, AgenticLLMAgent)
    assert agent.provider == "HuggingFace"
    assert agent.hf_model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    assert agent.hf_token == "token_abc"


@patch("huggingface_hub.InferenceClient")
def test_decide_with_huggingface_mock(mock_client_class, train01_env):
    """Verifica la decisione tramite mock di InferenceClient."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance

    # Mock della risposta chat completion
    mock_choice = MagicMock()
    mock_choice.message.content = '{"explanation": "Azione preventiva", "plan": {"maintenance": {"Ponte_1": "ispezione_straordinaria"}}}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_instance.chat.completions.create.return_value = mock_response

    agent = LLMAgent("TestHF", "Bilanciato", "HuggingFace", model="Qwen/Qwen2.5-72B-Instruct", hf_token="test_tok")
    plan, rationale = agent.decide(train01_env)

    assert plan == {"maintenance": {"Ponte_1": "ispezione_straordinaria"}}
    assert rationale[1][0]["explanation"] == "Azione preventiva"


def test_arena_single_simulation_heuristic():
    """Verifica che run_single_simulation esegua correttamente un'intera simulazione per l'agente euristico."""
    from infralab.lab.spec import resolve_scenario
    heuristic_comp = {"name": "Test_Euristico", "type": "Euristico"}
    result = run_single_simulation(heuristic_comp, scenario_spec=resolve_scenario("PAPERA-SHORT-A"),
                                   seed=1, scenario_id="PAPERA-SHORT-A", horizon_quarters=8)

    assert result.status == "SUCCESS", f"Errore: {result.error_message}"
    assert result.quarters_simulated == 8
    assert result.final_health_index > 0.0
    assert result.min_beta > 0.0
    assert result.final_net_worth > 0.0
    # Regressione: il piano deve arrivare al motore con il wrapper "maintenance",
    # altrimenti nessuna azione viene eseguita e la spesa resta 0.
    assert result.total_spent > 0.0
    assert result.trace is not None and len(result.trace) == 8
    assert {"quarter", "beta_net", "health_index", "budget", "spent"} <= set(result.trace[0])


def test_arena_cli_list_and_dry_run(tmp_path):
    """CLI: --list esce 0; dry run senza LLM produce CSV + JSON con provenance."""
    from infralab.lab.arena import main
    import json as _json
    assert main(["--list"]) == 0
    out = tmp_path / "arena"
    assert main(["--no-llm", "--competitors", "Euristico_Baseline", "--scenarios", "WALL-DEMO",
                 "--seeds", "1", "--horizon", "4", "--output-dir", str(out)]) == 0
    jsons = list(out.glob("arena_runs_*.json"))
    assert len(jsons) == 1 and (out / "arena_paper_summary.csv").exists()
    payload = _json.loads(jsons[0].read_text(encoding="utf-8"))
    assert "git_commit" in payload["provenance"] and len(payload["runs"]) == 1
    assert payload["runs"][0]["status"] == "SUCCESS"


def test_arena_cli_comma_lists_and_competitors_file(tmp_path):
    """Liste separate da virgola e --competitors-file con subset per nome."""
    from infralab.lab.arena import main, _split_list
    import json as _json
    assert _split_list(["A,B", "C"]) == ["A", "B", "C"]
    comp_file = tmp_path / "comps.json"
    comp_file.write_text(_json.dumps([
        {"name": "h1", "type": "Euristico"},
        {"name": "llm_x", "type": "LLM", "provider": "HuggingFace", "model": "org/m"},
    ]), encoding="utf-8")
    out = tmp_path / "arena"
    assert main(["--competitors-file", str(comp_file), "--competitors", "h1",
                 "--scenarios", "WALL-DEMO,TRAIN-01", "--seeds", "1", "--horizon", "2",
                 "--output-dir", str(out)]) == 0
    payload = _json.loads(next(out.glob("arena_runs_*.json")).read_text(encoding="utf-8"))
    assert {r["scenario_id"] for r in payload["runs"]} == {"WALL-DEMO", "TRAIN-01"}
    assert {r["competitor_name"] for r in payload["runs"]} == {"h1"}
    assert "campaign_start_utc" in payload["provenance"] and "hf_model_revisions" in payload["provenance"]
