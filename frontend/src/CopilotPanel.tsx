import { useEffect, useRef, useState } from "react";
import { api, type CopilotMessage } from "./api";
import { Button, C, ChartCard } from "./ui";

// Pannello di dialogo col copilota (condizione Uomo+LLM). Il copilota discute e
// propone piani; "Applica al piano" riempie i selettori — la decisione resta
// all'umano. Condiviso da Console (admin) e Plancia (giocatore di ricerca).
export default function CopilotPanel({ sid, agents, onApply }: {
  sid: string; agents: string[]; onApply: (plan: Record<string, string>) => void;
}) {
  const [messages, setMessages] = useState<CopilotMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [provider, setProvider] = useState("Locale");
  const [proposal, setProposal] = useState<Record<string, string> | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = draft.trim();
    if (!text || sending) return;
    setDraft("");
    setSending(true);
    setError(null);
    setMessages((m) => [...m, { role: "user", content: text }]);
    try {
      const turn = await api.copilot(sid, text, provider, agents[0]);
      setMessages(turn.history);
      setProposal(turn.proposed_plan);
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }

  return (
    <ChartCard title={`Copilota AI — decidete insieme${agents[0] ? " · " + agents[0] : ""}`} style={{ marginTop: 16 }}>
      <div style={{ fontSize: 12, color: C.faint, marginBottom: 10, lineHeight: 1.5 }}>
        Il copilota conosce lo stato tecnico dei tuoi asset (β, condizioni, budget, proiezioni) ma <b style={{ color: C.dim }}>non decide da solo</b>:
        discute le ipotesi con te, motiva i trade-off e propone piani che puoi applicare, modificare o ignorare.
      </div>
      <div ref={scrollRef} style={{ maxHeight: 260, overflowY: "auto", display: "flex", flexDirection: "column", gap: 8, padding: "4px 2px" }}>
        {messages.length === 0 && (
          <div className="mono" style={{ fontSize: 11.5, color: C.faint, padding: "12px 4px" }}>
            Presentati: il copilota si calibra sul tuo profilo (propensione al rischio, priorità budget/sicurezza/CO₂).
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "78%",
            background: m.role === "user" ? C.tint : C.panel2, border: `1px solid ${m.role === "user" ? C.celeste : C.line2}`,
            borderRadius: 8, padding: "8px 12px", fontSize: 13, color: C.ink, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
            {m.content}
          </div>
        ))}
        {sending && <div className="mono" style={{ fontSize: 11.5, color: C.faint }}>il copilota sta ragionando…</div>}
      </div>
      {proposal && Object.keys(proposal).length > 0 && (
        <div style={{ marginTop: 10, background: C.tint, border: `1px solid ${C.celeste}`, borderRadius: 8, padding: "10px 14px",
          display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <span className="mono" style={{ fontSize: 11, letterSpacing: ".08em", textTransform: "uppercase", color: C.blueDeep, fontWeight: 600 }}>Proposta</span>
          <span className="mono" style={{ fontSize: 12, color: C.dim, flex: 1 }}>
            {Object.entries(proposal).map(([aid, act]) => `asset ${aid} → ${act}`).join(" · ")}
          </span>
          <Button kind="primary" onClick={() => { onApply(proposal); setProposal(null); }}>Applica al piano</Button>
          <Button onClick={() => setProposal(null)}>Ignora</Button>
        </div>
      )}
      {error && <div className="mono" style={{ fontSize: 11.5, color: C.red, marginTop: 8 }}>{error}</div>}
      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <select value={provider} onChange={(e) => setProvider(e.target.value)} className="mono"
          style={{ background: C.bg, color: C.dim, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "8px 8px", fontSize: 12 }}>
          {["Locale", "OpenAI", "Google"].map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") send(); }}
          placeholder="Discuti col copilota: ipotesi, dubbi, richieste di proposta…"
          style={{ flex: 1, background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "8px 12px", fontSize: 13 }}
        />
        <Button kind="primary" onClick={send} disabled={sending || !draft.trim()}>Invia</Button>
      </div>
    </ChartCard>
  );
}
