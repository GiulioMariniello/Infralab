import { useEffect, useMemo, useState } from "react";
import { api, type ConfigField, type ConfigSchema } from "./api";
import { Button, C, Label } from "./ui";

export default function ConfigView({
  overrides,
  setOverrides,
}: {
  overrides: Record<string, unknown>;
  setOverrides: (o: Record<string, unknown>) => void;
}) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null);

  useEffect(() => {
    api.config().then(setSchema).catch(() => setSchema(null));
  }, []);

  const groups = useMemo(() => {
    if (!schema) return [];
    const g: Record<string, ConfigField[]> = {};
    for (const f of schema.fields) (g[f.group] ??= []).push(f);
    return Object.entries(g);
  }, [schema]);

  if (!schema) return <div style={{ textAlign: "center", padding: 70, color: C.faint }} className="mono">Caricamento configurazione…</div>;

  const modified = Object.keys(overrides).length;

  function setField(key: string, value: unknown, isDefault: boolean) {
    const next = { ...overrides };
    if (isDefault) delete next[key];
    else next[key] = value;
    setOverrides(next);
  }

  function valueOf(f: ConfigField) {
    return f.key in overrides ? overrides[f.key] : schema!.defaults[f.key];
  }

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16, flexWrap: "wrap" }}>
        <Label>Configurazione simulazione</Label>
        <span className="mono" style={{ fontSize: 12, color: modified ? C.blue : C.faint }}>
          {modified ? `${modified} parametri modificati rispetto al default` : "tutti i parametri al default"}
        </span>
        <Button onClick={() => setOverrides({})} disabled={!modified}>Ripristina default</Button>
        <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>
          I parametri si applicano premendo “Avvia / Ricrea sessione”.
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(360px, 1fr))", gap: 16 }}>
        {groups.map(([group, fields]) => (
          <div key={group} style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "16px 18px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
            <div className="mono" style={{ fontSize: 11.5, letterSpacing: ".12em", textTransform: "uppercase", color: C.blue, marginBottom: 12 }}>{group}</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
              {fields.map((f) => {
                const v = valueOf(f);
                const isMod = f.key in overrides;
                return (
                  <div key={f.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                    <span style={{ fontSize: 13, color: isMod ? C.blueDeep : C.dim, fontWeight: isMod ? 600 : 400 }}>
                      {f.label}
                      {isMod && <span style={{ color: C.celeste }}> ●</span>}
                    </span>
                    {f.type === "bool" ? (
                      <input type="checkbox" checked={Boolean(v)} onChange={(e) => setField(f.key, e.target.checked, e.target.checked === schema.defaults[f.key])} />
                    ) : f.type === "text" ? (
                      <input type="text" value={String(v ?? "")} onChange={(e) => setField(f.key, e.target.value, e.target.value === schema.defaults[f.key])} className="mono" style={sel} />
                    ) : f.type === "json" ? (
                      <textarea
                        defaultValue={JSON.stringify(v ?? null)}
                        onBlur={(e) => {
                          try {
                            const parsed = JSON.parse(e.target.value || "null");
                            setField(f.key, parsed, JSON.stringify(parsed) === JSON.stringify(schema.defaults[f.key]));
                          } catch {
                            /* JSON non valido: il valore precedente resta */
                          }
                        }}
                        className="mono"
                        rows={2}
                        style={{ ...sel, width: 220 }}
                      />
                    ) : f.type === "select" ? (
                      <select value={String(v)} onChange={(e) => setField(f.key, e.target.value, e.target.value === schema.defaults[f.key])} className="mono" style={sel}>
                        {f.options!.map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    ) : (
                      <input
                        type="number"
                        value={Number(v)}
                        min={f.min}
                        max={f.max}
                        step={f.step}
                        onChange={(e) => {
                          const num = Number(e.target.value);
                          setField(f.key, num, num === Number(schema.defaults[f.key]));
                        }}
                        className="mono"
                        style={{ ...sel, width: 130, textAlign: "right" }}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const sel: React.CSSProperties = { background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "6px 8px", fontSize: 12.5, minWidth: 130 };
