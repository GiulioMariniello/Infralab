import { useCallback, useEffect, useState } from "react";
import { api, builderApi, type AssetType, type AssetTypeSchema, type CustomScenario, type PropertyField } from "./api";
import { Button, C, ChartCard, Label } from "./ui";

type AssetRow = Record<string, unknown>;

// Il form è generato dallo schema del tipo: i nuovi asset type ereditano
// il costruttore gratis (nessun campo hardcoded qui).
export default function BuilderView({ onSaved }: { onSaved: () => void }) {
  const [types, setTypes] = useState<AssetType[]>([]);
  const [typeName, setTypeName] = useState("bridge");
  const [schema, setSchema] = useState<AssetTypeSchema | null>(null);
  const [customs, setCustoms] = useState<CustomScenario[]>([]);
  const [sid, setSid] = useState("MIO-SCENARIO-01");
  const [name, setName] = useState("");
  const [budget, setBudget] = useState(600000);
  const [assets, setAssets] = useState<AssetRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);

  const reloadCustoms = useCallback(() => {
    builderApi.list().then(setCustoms).catch(() => setCustoms([]));
  }, []);

  useEffect(() => {
    api.assetTypes().then(setTypes).catch(() => setTypes([]));
    reloadCustoms();
  }, [reloadCustoms]);

  useEffect(() => {
    builderApi.typeSchema(typeName).then(setSchema).catch(() => setSchema(null));
  }, [typeName]);

  function defaultsFromSchema(s: AssetTypeSchema): AssetRow {
    const row: AssetRow = {};
    for (const [key, meta] of Object.entries(s.property_schema)) {
      if (meta.default !== undefined) row[key] = meta.default;
    }
    return row;
  }

  function addAsset() {
    if (!schema) return;
    const nextId = assets.reduce((m, a) => Math.max(m, Number(a.id ?? 0)), 0) + 1;
    setAssets([...assets, { id: nextId, name: `${schema.display_name} ${nextId}`, ...defaultsFromSchema(schema) }]);
  }

  function updateAsset(i: number, key: string, value: unknown) {
    setAssets(assets.map((a, k) => (k === i ? { ...a, [key]: value } : a)));
  }

  function removeAsset(i: number) {
    setAssets(assets.filter((_, k) => k !== i));
  }

  function switchType(t: string) {
    setTypeName(t);
    setAssets([]);   // schema diverso: il parco riparte
    setMsg(null);
    setError(null);
  }

  function loadCustom(s: CustomScenario) {
    setTypeName(s.asset_type);
    setSid(s.scenario_id);
    setName(s.name ?? "");
    setBudget(Number(s.params?.initial_budget ?? 600000));
    setAssets(s.bridges as AssetRow[]);
    setMsg(null);
    setError(null);
  }

  async function save() {
    setBusy(true);
    setMsg(null);
    setError(null);
    try {
      const res = await builderApi.save({
        scenario_id: sid.trim(),
        name: name.trim() || sid.trim(),
        description: `Scenario custom (${schema?.display_name ?? typeName}) creato dal costruttore asset.`,
        asset_type: typeName,
        params: { initial_budget: budget },
        bridges: assets,
      });
      setMsg(
        res.warnings.length
          ? `Salvato con avvisi: ${res.warnings.join(" · ")}`
          : `Scenario '${res.scenario_id}' salvato: ora è selezionabile per le sessioni e le campagne.`
      );
      reloadCustoms();
      onSaved();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function importExcel(file: File) {
    setImporting(true);
    setMsg(null);
    setError(null);
    try {
      const res = await builderApi.importFile(typeName, file);
      if (!res.assets.length) {
        setError(`Nessun asset importato. ${res.errors.join(" · ") || ""}`);
      } else {
        setAssets(res.assets as AssetRow[]);
        const bits = [`${res.assets.length} asset importati`];
        if (res.warnings.length) bits.push(`${res.warnings.length} avvisi`);
        if (res.errors.length) bits.push(`${res.errors.length} righe saltate`);
        if (res.ignored_columns.length) bits.push(`colonne ignorate: ${res.ignored_columns.join(", ")}`);
        setMsg(bits.join(" · ") + ". Rivedi e salva.");
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setImporting(false);
    }
  }

  async function removeScenario(scenarioId: string) {
    try {
      await builderApi.remove(scenarioId);
      reloadCustoms();
      onSaved();
    } catch (e) {
      setError(String(e));
    }
  }

  const fields: [string, PropertyField][] = schema ? Object.entries(schema.property_schema) : [];

  return (
    <ChartCard title="Parco asset — costruttore scenari" style={{ marginTop: 22 }}>
      <div style={{ fontSize: 12.5, color: C.faint, lineHeight: 1.5, maxWidth: 760, marginBottom: 14 }}>
        Componi manualmente il parco: scegli la tipologia, aggiungi asset con le loro proprietà
        (il form segue lo schema della tipologia) e salva come <b style={{ color: C.dim }}>scenario custom</b> —
        giocabile in sessione, usabile nelle campagne e nei link giocatore come uno scenario base.
      </div>

      <div style={{ display: "flex", gap: 14, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 14 }}>
        <div>
          <Label>Tipologia</Label>
          <select value={typeName} onChange={(e) => switchType(e.target.value)} className="mono" style={{ ...input, width: 240, marginTop: 4 }}>
            {types.map((t) => (
              <option key={t.name} value={t.name}>
                {t.ui_meta?.icon ? `${t.ui_meta.icon} ` : ""}{t.display_name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <Label>ID scenario</Label>
          <input value={sid} onChange={(e) => setSid(e.target.value)} className="mono" style={{ ...input, width: 190, marginTop: 4 }} />
        </div>
        <div>
          <Label>Nome</Label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder={sid} className="mono" style={{ ...input, width: 220, marginTop: 4 }} />
        </div>
        <div>
          <Label>Budget iniziale (€)</Label>
          <input type="number" value={budget} onChange={(e) => setBudget(Number(e.target.value))} className="mono" style={{ ...input, width: 140, marginTop: 4 }} />
        </div>
        <Button onClick={addAsset} disabled={!schema || assets.length >= 50}>+ Aggiungi asset</Button>
        <Button kind="primary" onClick={save} disabled={busy || !assets.length || !sid.trim()}>
          Salva scenario ({assets.length} asset)
        </Button>
      </div>

      {/* Import da Excel */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 14, padding: "12px 14px", background: C.panel2, border: `1px solid ${C.line2}`, borderRadius: 8 }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: ".08em", textTransform: "uppercase", color: C.blueDeep, fontWeight: 600 }}>Da Excel</span>
        <a href={builderApi.templateUrl(typeName)} className="mono" style={{ fontSize: 12.5, color: C.blue, textDecoration: "none", border: `1px solid ${C.celeste}`, borderRadius: 5, padding: "7px 12px", background: C.tint }}>
          ⬇ Scarica modello ({typeName})
        </a>
        <label className="mono" style={{ fontSize: 12.5, color: C.dim, border: `1px solid ${C.line2}`, borderRadius: 5, padding: "7px 12px", cursor: "pointer", background: C.bg }}>
          {importing ? "importazione…" : "⬆ Importa file compilato"}
          <input
            type="file"
            accept=".xlsx,.xls,.csv"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) importExcel(f); e.target.value = ""; }}
          />
        </label>
        <span style={{ fontSize: 11.5, color: C.faint }}>compila il modello (colonne = proprietà del tipo) e ricaricalo: gli asset riempiono la tabella.</span>
      </div>

      {schema?.ui_meta?.demand_proxy_note && (
        <div className="mono" style={{ fontSize: 11, color: C.faint, marginBottom: 10 }}>ⓘ {schema.ui_meta.demand_proxy_note}</div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {assets.map((a, i) => (
          <div key={i} style={{ background: C.panel2, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
              <span className="mono" style={{ fontSize: 11, color: C.faint, width: 26, marginBottom: 8 }}>#{String(a.id)}</span>
              <div>
                <Label>Nome</Label>
                <input value={String(a.name ?? "")} onChange={(e) => updateAsset(i, "name", e.target.value)} className="mono" style={{ ...input, width: 180, marginTop: 4 }} />
              </div>
              {fields.map(([key, meta]) => (
                <div key={key}>
                  <Label>{meta.label}{meta.unit && meta.unit !== "-" ? ` (${meta.unit})` : ""}</Label>
                  {meta.type === "select" ? (
                    <select
                      value={String(a[key] ?? meta.default ?? "")}
                      onChange={(e) => updateAsset(i, key, e.target.value)}
                      className="mono"
                      style={{ ...input, width: 110, marginTop: 4 }}
                    >
                      {(meta.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : (
                    <input
                      type="number"
                      value={a[key] === undefined || a[key] === null ? "" : Number(a[key])}
                      placeholder={meta.default !== undefined ? String(meta.default) : ""}
                      min={meta.min}
                      max={meta.max}
                      onChange={(e) => updateAsset(i, key, e.target.value === "" ? undefined : Number(e.target.value))}
                      className="mono"
                      style={{ ...input, width: 112, marginTop: 4 }}
                    />
                  )}
                </div>
              ))}
              <div>
                <Label>Lat / Lon (opz.)</Label>
                <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                  <input type="number" value={a.lat === undefined || a.lat === null ? "" : Number(a.lat)} placeholder="lat" onChange={(e) => updateAsset(i, "lat", e.target.value === "" ? undefined : Number(e.target.value))} className="mono" style={{ ...input, width: 86 }} />
                  <input type="number" value={a.lon === undefined || a.lon === null ? "" : Number(a.lon)} placeholder="lon" onChange={(e) => updateAsset(i, "lon", e.target.value === "" ? undefined : Number(e.target.value))} className="mono" style={{ ...input, width: 86 }} />
                </div>
              </div>
              <Button onClick={() => removeAsset(i)}>Rimuovi</Button>
            </div>
          </div>
        ))}
        {!assets.length && (
          <div className="mono" style={{ fontSize: 12, color: C.faint, padding: "16px 4px" }}>
            Nessun asset nel parco: “+ Aggiungi asset” per iniziare, oppure carica uno scenario custom qui sotto.
          </div>
        )}
      </div>

      {msg && <div className="mono" style={{ fontSize: 12, color: C.green, marginTop: 12 }}>{msg}</div>}
      {error && <div className="mono" style={{ fontSize: 12, color: C.red, marginTop: 12 }}>{error}</div>}

      {customs.length > 0 && (
        <div style={{ marginTop: 18 }}>
          <Label>Scenari custom salvati</Label>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            {customs.map((s) => (
              <div key={s.scenario_id} className="mono" style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 12.5, color: C.dim, background: C.panel2, border: `1px solid ${C.line2}`, borderRadius: 6, padding: "8px 12px" }}>
                <b style={{ color: C.blueDeep }}>{s.scenario_id}</b>
                <span>{s.asset_type} · {s.bridges.length} asset</span>
                <span style={{ flex: 1 }} />
                <Button onClick={() => loadCustom(s)}>Modifica</Button>
                <Button onClick={() => removeScenario(s.scenario_id)}>Elimina</Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </ChartCard>
  );
}

const input: React.CSSProperties = {
  background: C.bg, color: C.ink, border: `1px solid ${C.line2}`,
  borderRadius: 4, padding: "8px 10px", fontSize: 13,
};
