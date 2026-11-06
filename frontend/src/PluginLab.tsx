import { useCallback, useEffect, useState } from "react";
import { pluginApi, type PluginInfo } from "./api";
import { Button, C, Label } from "./ui";

type Kind = "agent" | "degradation";

const KIND_INFO: Record<Kind, { label: string; entry: string; blurb: string }> = {
  agent: {
    label: "Agente",
    entry: "decide(obs)",
    blurb: "Definisci decide(obs) → dict {id_asset: azione}. L'agente diventa selezionabile nell'Editor come tipo «Custom».",
  },
  degradation: {
    label: "Modello di degrado",
    entry: "degrade(ctx)",
    blurb: "Definisci degrade(ctx) → nuova condizione [0,1]. Il modello compare in Config come «custom:<nome>» per il parametro Modello di degrado.",
  },
};

// Laboratorio codice: scrivi in Python i tuoi agenti e modelli di degrado.
// Il contratto input/output è nel template; valida (esecuzione di prova in
// sandbox) e salva. I plugin salvati alimentano Editor e Config.
export default function PluginLab() {
  const [kind, setKind] = useState<Kind>("agent");
  const [templates, setTemplates] = useState<{ agent: string; degradation: string } | null>(null);
  const [list, setList] = useState<PluginInfo[]>([]);
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(() => {
    pluginApi.list(kind).then(setList).catch(() => setList([]));
  }, [kind]);

  useEffect(() => {
    pluginApi.templates().then(setTemplates).catch(() => setTemplates(null));
  }, []);

  useEffect(() => {
    reload();
    setStatus(null);
    if (templates) setSource(templates[kind]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, templates]);

  async function validate() {
    setBusy(true);
    try {
      const r = await pluginApi.validate(kind, source);
      setStatus({ ok: r.ok, msg: r.message });
    } catch (e) {
      setStatus({ ok: false, msg: String(e) });
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!name.trim()) { setStatus({ ok: false, msg: "Dai un nome al plugin." }); return; }
    setBusy(true);
    try {
      const r = await pluginApi.save(kind, name.trim(), source);
      setStatus({ ok: true, msg: r.message + " — salvato come «" + r.name + "»." });
      reload();
    } catch (e) {
      setStatus({ ok: false, msg: String(e) });
    } finally {
      setBusy(false);
    }
  }

  async function loadPlugin(n: string) {
    const p = await pluginApi.get(kind, n);
    setName(p.name);
    setSource(p.source);
    setStatus(null);
  }

  async function remove(n: string) {
    await pluginApi.remove(kind, n).catch(() => undefined);
    if (name === n) resetToTemplate();
    reload();
  }

  function resetToTemplate() {
    if (templates) setSource(templates[kind]);
    setName("");
    setStatus(null);
  }

  const info = KIND_INFO[kind];

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
        <Label>Laboratorio codice</Label>
        <div style={{ display: "flex", border: `1px solid ${C.line2}`, borderRadius: 6, overflow: "hidden" }}>
          {(["agent", "degradation"] as Kind[]).map((k) => (
            <button
              key={k}
              onClick={() => setKind(k)}
              className="mono"
              style={{ border: "none", padding: "8px 16px", fontSize: 12, letterSpacing: ".05em",
                background: kind === k ? C.blue : "transparent", color: kind === k ? "#fff" : C.dim, fontWeight: kind === k ? 600 : 500 }}
            >
              {KIND_INFO[k].label}
            </button>
          ))}
        </div>
        <span className="mono" style={{ fontSize: 12, color: C.blueDeep }}>entrypoint: {info.entry}</span>
      </div>

      <div style={{ fontSize: 12.5, color: C.faint, lineHeight: 1.5, maxWidth: 820, marginBottom: 14 }}>
        {info.blurb} Il codice gira in una <b style={{ color: C.dim }}>sandbox</b> (builtins ristretti, import consentiti:
        math, statistics, random, itertools, functools, collections, heapq, json). Il contratto input/output è nei commenti del template.
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 16 }}>
        <div>
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="nome-del-plugin"
              className="mono"
              style={{ ...input, width: 240 }}
            />
            <Button onClick={validate} disabled={busy}>Valida</Button>
            <Button kind="primary" onClick={save} disabled={busy}>Salva</Button>
            <Button onClick={resetToTemplate} disabled={busy}>Template</Button>
          </div>
          <textarea
            value={source}
            onChange={(e) => setSource(e.target.value)}
            spellCheck={false}
            style={{
              width: "100%", height: 460, background: "#0f1b26", color: "#d6e6f2",
              border: `1px solid ${C.line2}`, borderRadius: 8, padding: "14px 16px",
              fontFamily: "IBM Plex Mono, monospace", fontSize: 12.5, lineHeight: 1.55, tabSize: 4, resize: "vertical",
            }}
          />
          {status && (
            <div className="mono" style={{ marginTop: 10, fontSize: 12.5, padding: "10px 14px", borderRadius: 6,
              background: status.ok ? "#eaf6ee" : "#fdecea", color: status.ok ? C.green : C.red,
              border: `1px solid ${status.ok ? C.green : C.red}` }}>
              {status.ok ? "✓ " : "✕ "}{status.msg}
            </div>
          )}
        </div>

        <div>
          <Label>{info.label} salvati</Label>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            {list.length === 0 && <span className="mono" style={{ fontSize: 12, color: C.faint }}>nessuno ancora</span>}
            {list.map((p) => (
              <div key={p.name} style={{ display: "flex", alignItems: "center", gap: 8, background: C.panel2, border: `1px solid ${C.line2}`, borderRadius: 6, padding: "8px 10px" }}>
                <span className="mono" style={{ fontSize: 12.5, color: C.blueDeep, flex: 1, cursor: "pointer" }} onClick={() => loadPlugin(p.name)}>{p.name}</span>
                <button onClick={() => remove(p.name)} className="mono" style={{ border: "none", background: "transparent", color: C.faint, cursor: "pointer", fontSize: 12 }}>✕</button>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, fontSize: 11.5, color: C.faint, lineHeight: 1.5 }}>
            {kind === "agent"
              ? "Usa l'agente in una partita: Editor → tipo «Custom» → scegli il plugin."
              : "Usa il modello: Config → Modello di degrado → «custom:<nome>»."}
          </div>
        </div>
      </div>
    </div>
  );
}

const input: React.CSSProperties = {
  background: C.bg, color: C.ink, border: `1px solid ${C.line2}`,
  borderRadius: 4, padding: "8px 10px", fontSize: 13,
};
