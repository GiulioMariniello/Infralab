import { useEffect, useState } from "react";
import { EDITOR_AGENT_TYPES, pluginApi, researchApi, type AgentSpec, type LlmStatus, type PluginInfo } from "./api";
import { Button, C, Label } from "./ui";

const TYPE_HINTS: Record<string, string> = {
  Umano: "decide manualmente dalla Console (piani per trimestre)",
  Euristico: "policy a regole: ispezioni programmate + soglie di condizione/β",
  "GA Orizzonte Variabile": "ottimizzazione genetica su orizzonte mobile (lento)",
  "Algoritmo Memetico": "GA + ricerca locale (lento)",
  MCTS: "Monte Carlo Tree Search sulle azioni (lento)",
  "RL Q-Learning": "Q-learning tabellare addestrato online (lento)",
  LLM: "modello linguistico: decide dal prompt di stato (API o locale)",
  "LLM Agentico": "LLM che scrive/esegue codice: clona la simulazione, testa piani, ottimizza, poi decide",
  Custom: "agente definito da te in Python (tab Codice): decide(obs) → piano",
};

const ALL_TYPES = [...EDITOR_AGENT_TYPES, "LLM", "LLM Agentico", "Custom"];

export default function EditorView({
  agents,
  setAgents,
}: {
  agents: AgentSpec[];
  setAgents: (a: AgentSpec[]) => void;
}) {
  const [llm, setLlm] = useState<LlmStatus | null>(null);
  const [agentPlugins, setAgentPlugins] = useState<PluginInfo[]>([]);

  useEffect(() => {
    researchApi.llmStatus().then(setLlm).catch(() => setLlm(null));
    pluginApi.list("agent").then(setAgentPlugins).catch(() => setAgentPlugins([]));
  }, []);

  function update(i: number, patch: Partial<AgentSpec>) {
    setAgents(agents.map((a, k) => (k === i ? { ...a, ...patch } : a)));
  }
  function add() {
    setAgents([...agents, { name: `Manager ${agents.length + 1}`, type: "Euristico" }]);
  }
  function remove(i: number) {
    setAgents(agents.filter((_, k) => k !== i));
  }

  const names = agents.map((a) => a.name.trim());
  const duplicates = new Set(names.filter((n, i) => names.indexOf(n) !== i));
  const humans = agents.filter((a) => a.type === "Umano").length;

  function providerOptions(): { value: string; label: string; disabled: boolean }[] {
    if (!llm) return [{ value: "Locale", label: "Locale", disabled: false }];
    return [
      { value: "Locale", label: `Locale · ${llm.locale.model} @ ${llm.locale.base_url}`, disabled: false },
      { value: "OpenAI", label: `OpenAI · ${llm.openai.model}${llm.openai.configured ? "" : " (chiave assente)"}`, disabled: !llm.openai.configured },
      { value: "Google", label: `Google · ${llm.google.model}${llm.google.configured ? "" : " (chiave assente)"}`, disabled: !llm.google.configured },
    ];
  }

  return (
    <div style={{ marginTop: 4, maxWidth: 980 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14, flexWrap: "wrap" }}>
        <Label>Editor agenti — composizione della partita</Label>
        <span className="mono" style={{ fontSize: 12, color: C.dim }}>
          {agents.length} manager · {humans} uman{humans === 1 ? "o" : "i"} · {agents.length - humans} AI
        </span>
        <Button onClick={add} disabled={agents.length >= 6}>+ Aggiungi manager</Button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {agents.map((a, i) => (
          <div
            key={i}
            style={{
              display: "flex", alignItems: "flex-end", gap: 14, flexWrap: "wrap",
              background: C.panel, border: `1px solid ${duplicates.has(a.name.trim()) ? C.red : C.line2}`,
              borderRadius: 8, padding: "13px 16px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)",
            }}
          >
            <span className="mono" style={{ fontSize: 11, color: C.faint, width: 24, marginBottom: 8 }}>{String(i + 1).padStart(2, "0")}</span>
            <div>
              <Label>Nome</Label>
              <input value={a.name} onChange={(e) => update(i, { name: e.target.value })} className="mono" style={{ ...input, width: 160, marginTop: 4 }} />
            </div>
            <div>
              <Label>Tipo</Label>
              <select
                value={a.type}
                onChange={(e) => {
                  const t = e.target.value;
                  const isLlm = t.startsWith("LLM");
                  update(i, {
                    type: t,
                    provider: isLlm ? a.provider ?? "Locale" : undefined,
                    profile: isLlm ? a.profile ?? "Bilanciato" : undefined,
                    plugin: t === "Custom" ? a.plugin ?? agentPlugins[0]?.name : undefined,
                  });
                }}
                className="mono"
                style={{ ...input, width: 200, marginTop: 4 }}
              >
                {ALL_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            {a.type === "Custom" && (
              <div>
                <Label>Plugin agente</Label>
                {agentPlugins.length ? (
                  <select value={a.plugin ?? ""} onChange={(e) => update(i, { plugin: e.target.value })} className="mono" style={{ ...input, width: 200, marginTop: 4 }}>
                    <option value="">— scegli —</option>
                    {agentPlugins.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
                  </select>
                ) : (
                  <div className="mono" style={{ fontSize: 11.5, color: C.red, marginTop: 8, width: 200 }}>nessun plugin: crealo nel tab Codice</div>
                )}
              </div>
            )}
            {a.type.startsWith("LLM") && (
              <>
                <div>
                  <Label>Provider</Label>
                  <select value={a.provider ?? "Locale"} onChange={(e) => update(i, { provider: e.target.value })} className="mono" style={{ ...input, width: 280, marginTop: 4 }}>
                    {providerOptions().map((o) => (
                      <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <Label>Profilo</Label>
                  <select value={a.profile ?? "Bilanciato"} onChange={(e) => update(i, { profile: e.target.value })} className="mono" style={{ ...input, width: 160, marginTop: 4 }}>
                    {(llm?.profiles ?? ["Bilanciato"]).map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <div>
                  <Label>Modello (opz.)</Label>
                  <input value={a.model ?? ""} onChange={(e) => update(i, { model: e.target.value || undefined })} placeholder="default" className="mono" style={{ ...input, width: 150, marginTop: 4 }} />
                </div>
              </>
            )}
            <span
              className="mono"
              style={{
                fontSize: 10.5, padding: "4px 8px", borderRadius: 4, marginBottom: 6,
                background: a.type === "Umano" ? C.tint : C.panel2,
                color: a.type === "Umano" ? C.blueDeep : C.dim,
                border: `1px solid ${a.type === "Umano" ? C.celeste : C.line2}`,
              }}
            >
              {a.type === "Umano" ? "MANUALE" : a.type === "Custom" ? "CODICE" : a.type.startsWith("LLM") ? (a.type === "LLM Agentico" ? "LLM+CODE" : "LLM") : "AI"}
            </span>
            <span style={{ fontSize: 12, color: C.faint, flex: 1, minWidth: 160, marginBottom: 7 }}>{TYPE_HINTS[a.type] ?? ""}</span>
            <Button onClick={() => remove(i)} disabled={agents.length <= 1}>Rimuovi</Button>
          </div>
        ))}
      </div>

      {duplicates.size > 0 && (
        <div style={{ marginTop: 12, fontSize: 12.5, color: C.red }}>Nomi duplicati: correggi prima di avviare.</div>
      )}

      <div style={{ marginTop: 18, fontSize: 12.5, color: C.faint, lineHeight: 1.6, maxWidth: 720 }}>
        Gli asset dello scenario vengono assegnati <b style={{ color: C.dim }}>round-robin</b> tra i manager
        (salvo assegnazione esplicita nello scenario). Ogni manager ha il proprio budget. Gli agenti
        <b style={{ color: C.dim }}> AI/LLM decidono da soli</b> ad ogni trimestre; per gli <b style={{ color: C.dim }}>umani</b> imposti
        i piani dalla Console (modalità Manuale). LLM <b style={{ color: C.dim }}>Locale</b> usa un server OpenAI-compatibile
        (Ollama, LM Studio, vLLM) su {llm?.locale.base_url ?? "http://localhost:11434/v1"} — configurabile con
        INFRALAB_LLM_BASE_URL / INFRALAB_LLM_MODEL.
      </div>
    </div>
  );
}

const input: React.CSSProperties = {
  background: C.bg, color: C.ink, border: `1px solid ${C.line2}`,
  borderRadius: 4, padding: "8px 10px", fontSize: 13,
};
