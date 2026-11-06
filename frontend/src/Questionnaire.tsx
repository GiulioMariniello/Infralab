import { useEffect, useState } from "react";
import { studyApi, type QItem, type QSection } from "./api";
import { Button, C } from "./ui";

// Questionario post-partita (schema-driven dal backend): profilo, NASA-TLX,
// fiducia nell'automazione, strategia. Mostrato al giocatore a fine partita;
// i punteggi (TLX, fiducia) sono calcolati e salvati lato server.
export default function Questionnaire({ sessionId, onDone, only, title, intro }: {
  sessionId: string; onDone: () => void; only?: string[]; title?: string; intro?: string;
}) {
  const [sections, setSections] = useState<QSection[]>([]);
  const [resp, setResp] = useState<Record<string, unknown>>({});
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    studyApi.questionnaireSchema().then((d) => {
      const secs = only ? d.sections.filter((s) => only.includes(s.id)) : d.sections;
      setSections(secs);
      const init: Record<string, unknown> = {};
      for (const s of secs) for (const it of s.items) if (it.default !== undefined) init[it.key] = it.default;
      setResp(init);
    }).catch(() => setSections([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function set(key: string, v: unknown) {
    setResp((r) => ({ ...r, [key]: v }));
  }

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      await studyApi.submitQuestionnaire(sessionId, resp);
      setDone(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <div style={{ maxWidth: 640, margin: "40px auto", textAlign: "center" }}>
        <div className="serif" style={{ fontSize: 26, color: C.blueDeep, fontWeight: 600 }}>Grazie!</div>
        <div style={{ marginTop: 12, color: C.dim, fontSize: 15 }}>Le tue risposte sono state registrate a fini di ricerca.</div>
        <div style={{ marginTop: 22 }}><Button kind="primary" onClick={onDone}>Continua →</Button></div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 760, margin: "16px auto 60px" }}>
      <div className="serif" style={{ fontSize: 24, color: C.blueDeep, fontWeight: 600 }}>{title ?? "Questionario di fine simulazione"}</div>
      <div style={{ color: C.faint, fontSize: 13, marginTop: 6, marginBottom: 18 }}>
        {intro ?? "Circa 2 minuti. Dati anonimi, usati solo per la ricerca. La sezione «Fiducia nel supporto AI» è facoltativa se non hai usato il copilota."}
      </div>
      {sections.map((s) => (
        <div key={s.id} style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 10, padding: "16px 20px", marginBottom: 14 }}>
          <div className="mono" style={{ fontSize: 11.5, letterSpacing: ".08em", textTransform: "uppercase", color: C.blueDeep, fontWeight: 600 }}>
            {s.title}{s.optional && <span style={{ color: C.faint, fontWeight: 400 }}> · facoltativa</span>}
          </div>
          {s.hint && <div style={{ fontSize: 12, color: C.faint, marginTop: 4 }}>{s.hint}</div>}
          <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 14 }}>
            {s.items.map((it) => <Field key={it.key} item={it} value={resp[it.key]} onChange={(v) => set(it.key, v)} />)}
          </div>
        </div>
      ))}
      {error && <div className="mono" style={{ color: C.red, fontSize: 12.5, marginBottom: 10 }}>{error}</div>}
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <Button kind="primary" onClick={submit} disabled={busy}>Invia questionario →</Button>
        <button onClick={onDone} className="mono" style={{ border: "none", background: "transparent", color: C.faint, fontSize: 12.5, cursor: "pointer" }}>salta</button>
      </div>
    </div>
  );
}

function Field({ item, value, onChange }: { item: QItem; value: unknown; onChange: (v: unknown) => void }) {
  return (
    <div>
      <div style={{ fontSize: 13.5, color: C.ink, marginBottom: 6 }}>{item.label}</div>
      {item.type === "number" && (
        <input type="number" min={item.min} max={item.max} value={value === undefined ? "" : Number(value)}
          onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))} className="mono" style={input} />
      )}
      {item.type === "text" && (
        <textarea value={String(value ?? "")} maxLength={item.max_chars ?? 800} onChange={(e) => onChange(e.target.value)}
          style={{ ...input, width: "100%", height: 70, resize: "vertical" }} />
      )}
      {item.type === "select" && (
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} className="mono" style={{ ...input, minWidth: 320 }}>
          <option value="">—</option>
          {item.options!.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      )}
      {item.type === "radio" && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {item.options!.map((o) => (
            <button key={o} onClick={() => onChange(o)} className="mono"
              style={{ fontSize: 12, padding: "7px 12px", borderRadius: 6, cursor: "pointer",
                border: `1px solid ${value === o ? C.celeste : C.line2}`, background: value === o ? C.tint : C.bg, color: value === o ? C.blueDeep : C.dim }}>
              {o}
            </button>
          ))}
        </div>
      )}
      {(item.type === "likert" || item.type === "slider") && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="mono" style={{ fontSize: 10.5, color: C.faint, width: 70, textAlign: "right" }}>{item.low ?? item.min}</span>
            <input type="range" min={item.min} max={item.max} step={item.step ?? 1}
              value={value === undefined ? Math.round(((item.min ?? 0) + (item.max ?? 100)) / 2) : Number(value)}
              onChange={(e) => onChange(Number(e.target.value))} style={{ flex: 1, accentColor: C.blue }} />
            <span className="mono" style={{ fontSize: 10.5, color: C.faint, width: 70 }}>{item.high ?? item.max}</span>
            <span className="mono" style={{ fontSize: 13, color: C.blueDeep, fontWeight: 600, width: 34, textAlign: "right" }}>
              {value === undefined ? "—" : Number(value)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

const input: React.CSSProperties = {
  background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 5, padding: "8px 10px", fontSize: 13,
};
