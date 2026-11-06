import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type AssetDetail, type SessionState } from "./api";
import { axisTick, betaColor, C, ChartCard, Label, tooltipStyle } from "./ui";
import { fmt, pct, sci } from "./useSession";

const COMP_COLORS = ["#1f6fb2", "#4aa3df", "#2e9e6b", "#e0902f", "#8a5bd6", "#d1544a"];

export default function AnalysisView({ state }: { state: SessionState | null }) {
  const [assetId, setAssetId] = useState<number | null>(null);
  const [detail, setDetail] = useState<AssetDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (state && assetId == null && state.assets.length) setAssetId(state.assets[0].id);
  }, [state, assetId]);

  useEffect(() => {
    if (!state || assetId == null) return;
    setLoading(true);
    api
      .assetDetail(state.session_id, assetId)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [state?.session_id, assetId, state?.kpis.quarter, state]);

  const degradation = useMemo(() => {
    if (!detail) return { rows: [] as Record<string, number | null>[], comps: [] as string[] };
    const comps = detail.components.map((c) => c.name);
    const byYear = new Map<number, Record<string, number | null>>();
    for (const c of detail.components) {
      for (const p of c.degradation) {
        const row = byYear.get(p.year) ?? { year: p.year };
        row[c.name] = p.condition;
        byYear.set(p.year, row);
      }
    }
    return { rows: [...byYear.values()].sort((a, b) => (a.year as number) - (b.year as number)), comps };
  }, [detail]);

  if (!state) return <div style={{ textAlign: "center", padding: 70, color: C.faint }} className="mono">Avvia una sessione per l'analisi.</div>;

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 16 }}>
        <div>
          <Label>Asset</Label>
          <select value={assetId ?? ""} onChange={(e) => setAssetId(Number(e.target.value))} className="mono" style={{ marginTop: 5, background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "8px 10px", fontSize: 13, minWidth: 300 }}>
            {state.assets.map((a) => (
              <option key={a.id} value={a.id}>{a.name}</option>
            ))}
          </select>
        </div>
        {loading && <span className="mono" style={{ fontSize: 12, color: C.faint }}>caricamento…</span>}
      </div>

      {detail && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginTop: 16 }}>
            <Tile label="β annuo (prior)" value={fmt(detail.beta_annual, 2)} color={betaColor(detail.beta_annual)} />
            <Tile label="β annuo (posterior)" value={fmt(detail.posterior_beta_annual, 2)} color={betaColor(detail.posterior_beta_annual)} />
            <Tile label="Target β annuo" value={fmt(detail.target_beta_annual, 2)} color={C.dim} />
            <Tile label="p_f annuo" value={sci(detail.annual_pf)} color={C.blue} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 18 }}>
            <ChartCard title="Proiezione affidabilità β (orizzonte)">
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={detail.projection} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
                  <CartesianGrid stroke={C.line} />
                  <XAxis dataKey="year" tick={axisTick} stroke={C.line2} label={{ value: "anni", position: "insideBottom", offset: -2, fontSize: 10, fill: C.faint }} />
                  <YAxis tick={axisTick} stroke={C.line2} domain={["auto", "auto"]} />
                  <Tooltip contentStyle={tooltipStyle} />
                  {detail.target_beta_annual != null && (
                    <ReferenceLine y={detail.target_beta_annual} stroke={C.amber} strokeDasharray="4 3" label={{ value: `target ${fmt(detail.target_beta_annual, 1)}`, fontSize: 10, fill: C.amber }} />
                  )}
                  <Line type="monotone" dataKey="beta" name="β" stroke={C.blue} strokeWidth={2.2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Curve di degrado per componente (condizione)">
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={degradation.rows} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
                  <CartesianGrid stroke={C.line} />
                  <XAxis dataKey="year" tick={axisTick} stroke={C.line2} label={{ value: "anni", position: "insideBottom", offset: -2, fontSize: 10, fill: C.faint }} />
                  <YAxis tick={axisTick} stroke={C.line2} domain={[0, 1]} />
                  <Tooltip contentStyle={tooltipStyle} />
                  {degradation.comps.map((c, i) => (
                    <Line key={c} type="monotone" dataKey={c} name={c} stroke={COMP_COLORS[i % COMP_COLORS.length]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 8 }}>
                {degradation.comps.map((c, i) => (
                  <span key={c} className="mono" style={{ fontSize: 11, color: C.dim, display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <span style={{ width: 10, height: 3, background: COMP_COLORS[i % COMP_COLORS.length] }} /> {c}
                  </span>
                ))}
              </div>
            </ChartCard>
          </div>

          <ChartCard title="Componenti — stato limite governante e margini β" style={{ marginTop: 16 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr>{["Componente", "Cond.", "β annuo", "Governante", "Stati limite (β annuo · margine)"].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {detail.components.map((c) => (
                  <tr key={c.name} style={{ borderTop: `1px solid ${C.line}` }}>
                    <td style={{ ...td, color: C.ink, fontWeight: 500 }}>{c.name}</td>
                    <td style={tdN}>{pct(c.condition)}</td>
                    <td style={{ ...tdN, color: betaColor(c.beta_annual) }}>{fmt(c.beta_annual, 2)}</td>
                    <td style={{ ...td, fontFamily: "IBM Plex Mono", color: C.blueDeep }}>{c.governing_limit_state ?? "—"}</td>
                    <td style={td}>
                      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                        {Object.entries(c.limit_states).map(([n, ls]) => (
                          <span key={n} className="mono" style={{ fontSize: 11.5, color: C.dim }}>
                            {n}: <b style={{ color: betaColor(ls.beta_annual) }}>{fmt(ls.beta_annual, 2)}</b>
                            <span style={{ color: (ls.margin_annual ?? 0) < 0 ? C.red : C.green }}> ({(ls.margin_annual ?? 0) >= 0 ? "+" : ""}{fmt(ls.margin_annual, 2)})</span>
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ChartCard>
        </>
      )}
    </div>
  );
}

function Tile({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "14px 16px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
      <Label>{label}</Label>
      <div className="mono" style={{ fontSize: 24, fontWeight: 600, color, marginTop: 8 }}>{value}</div>
    </div>
  );
}

const th: React.CSSProperties = { textAlign: "left", padding: "6px 10px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "8px 10px", color: C.dim };
const tdN: React.CSSProperties = { ...td, fontFamily: "IBM Plex Mono", textAlign: "right", fontVariantNumeric: "tabular-nums" };
