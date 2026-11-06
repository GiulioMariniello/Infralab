import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type AssetDetail, type AssetView } from "./api";
import { axisTick, betaColor, C, tooltipStyle } from "./ui";
import { eur, fmt, pct, sci } from "./useSession";

const COMP_COLORS = ["#1f6fb2", "#4aa3df", "#2e9e6b", "#e0902f", "#8a5bd6", "#d1544a"];

// Gemello digitale di un asset: pannello a scomparsa con lo stato "vivo"
// (β prior/posterior/target, proiezione, curve di degrado per componente,
// stati limite governanti). Riusa l'endpoint assetDetail. Mostrato come drawer
// nella plancia giocatore e apribile ovunque serva il dettaglio.
export default function DigitalTwin({
  sessionId,
  asset,
  quarter,
  icon = "",
  onClose,
}: {
  sessionId: string;
  asset: AssetView;
  quarter: number;
  icon?: string;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<AssetDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.assetDetail(sessionId, asset.id).then(setDetail).catch(() => setDetail(null)).finally(() => setLoading(false));
  }, [sessionId, asset.id, quarter]);

  const degRows: Record<string, number | null>[] = [];
  const comps = detail?.components.map((c) => c.name) ?? [];
  if (detail) {
    const byYear = new Map<number, Record<string, number | null>>();
    for (const c of detail.components) {
      for (const p of c.degradation) {
        const row = byYear.get(p.year) ?? { year: p.year };
        row[c.name] = p.condition;
        byYear.set(p.year, row);
      }
    }
    degRows.push(...[...byYear.values()].sort((a, b) => (a.year as number) - (b.year as number)));
  }

  return (
    <div style={overlay} onClick={onClose}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
          <div>
            <div className="serif" style={{ fontSize: 21, color: C.blueDeep, fontWeight: 600 }}>{icon ? icon + " " : ""}{asset.name}</div>
            <div className="mono" style={{ fontSize: 11.5, color: C.faint, marginTop: 2 }}>
              gemello digitale · {asset.manager} · {asset.status}
            </div>
          </div>
          <button onClick={onClose} className="mono" style={closeBtn}>chiudi ✕</button>
        </div>

        {loading && <div className="mono" style={{ color: C.faint, padding: 20 }}>caricamento gemello…</div>}

        {detail && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginTop: 10 }}>
              <Tile label="β annuo" value={fmt(detail.beta_annual, 2)} color={betaColor(detail.beta_annual)} />
              <Tile label="β post-prova" value={fmt(detail.posterior_beta_annual, 2)} color={betaColor(detail.posterior_beta_annual)} />
              <Tile label="Target β" value={fmt(detail.target_beta_annual, 2)} color={C.dim} />
              <Tile label="p_f annuo" value={sci(detail.annual_pf)} color={C.blue} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 14 }}>
              <Card title="Proiezione β (orizzonte)">
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={detail.projection} margin={{ top: 6, right: 14, bottom: 2, left: -10 }}>
                    <CartesianGrid stroke={C.line} />
                    <XAxis dataKey="year" tick={axisTick} stroke={C.line2} />
                    <YAxis tick={axisTick} stroke={C.line2} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={tooltipStyle} />
                    {detail.target_beta_annual != null && (
                      <ReferenceLine y={detail.target_beta_annual} stroke={C.amber} strokeDasharray="4 3" />
                    )}
                    <Line type="monotone" dataKey="beta" name="β" stroke={C.blue} strokeWidth={2.2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Degrado per componente (condizione)">
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={degRows} margin={{ top: 6, right: 14, bottom: 2, left: -10 }}>
                    <CartesianGrid stroke={C.line} />
                    <XAxis dataKey="year" tick={axisTick} stroke={C.line2} />
                    <YAxis tick={axisTick} stroke={C.line2} domain={[0, 1]} />
                    <Tooltip contentStyle={tooltipStyle} />
                    {comps.map((c, i) => (
                      <Line key={c} type="monotone" dataKey={c} name={c} stroke={COMP_COLORS[i % COMP_COLORS.length]} strokeWidth={2} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6 }}>
                  {comps.map((c, i) => (
                    <span key={c} className="mono" style={{ fontSize: 10.5, color: C.dim, display: "inline-flex", alignItems: "center", gap: 5 }}>
                      <span style={{ width: 10, height: 3, background: COMP_COLORS[i % COMP_COLORS.length] }} /> {c}
                    </span>
                  ))}
                </div>
              </Card>
            </div>

            <Card title="Componenti — stato limite governante" style={{ marginTop: 12 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                <thead>
                  <tr>{["Componente", "Cond.", "β annuo", "Governante"].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {detail.components.map((c) => (
                    <tr key={c.name} style={{ borderTop: `1px solid ${C.line}` }}>
                      <td style={{ ...td, color: C.ink, fontWeight: 500 }}>{c.name}</td>
                      <td style={tdN}>
                        <CondBar v={c.condition} />
                      </td>
                      <td style={{ ...tdN, color: betaColor(c.beta_annual) }}>{fmt(c.beta_annual, 2)}</td>
                      <td style={{ ...td, fontFamily: "IBM Plex Mono", color: C.blueDeep }}>{c.governing_limit_state ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <div className="mono" style={{ fontSize: 11.5, color: C.faint, marginTop: 12 }}>
              NAV {eur(asset.net_asset_value)} · health {pct(asset.health_index)}
              {asset.seismic?.enabled && asset.seismic.failure_probability != null && (
                <> · p_f sismico {sci(asset.seismic.failure_probability)}</>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function CondBar({ v }: { v: number }) {
  const w = Math.max(3, Math.min(100, v * 100));
  const col = v >= 0.6 ? C.green : v >= 0.35 ? C.amber : C.red;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 7, justifyContent: "flex-end", width: "100%" }}>
      <span style={{ color: C.dim, fontSize: 11.5 }}>{pct(v)}</span>
      <span style={{ width: 64, height: 6, background: C.line, borderRadius: 3, overflow: "hidden" }}>
        <span style={{ display: "block", width: `${w}%`, height: "100%", background: col }} />
      </span>
    </span>
  );
}

function Tile({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: C.panel2, border: `1px solid ${C.line2}`, borderRadius: 7, padding: "10px 12px" }}>
      <div className="mono" style={{ fontSize: 9.5, letterSpacing: ".08em", textTransform: "uppercase", color: C.faint }}>{label}</div>
      <div className="mono" style={{ fontSize: 20, fontWeight: 600, color, marginTop: 5 }}>{value}</div>
    </div>
  );
}

function Card({ title, children, style }: { title: string; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "12px 14px", ...style }}>
      <div className="mono" style={{ fontSize: 10.5, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, marginBottom: 8 }}>{title}</div>
      {children}
    </div>
  );
}

const overlay: React.CSSProperties = {
  position: "fixed", inset: 0, background: "rgba(11,39,64,0.32)", backdropFilter: "blur(2px)",
  display: "flex", justifyContent: "flex-end", zIndex: 50,
};
const panel: React.CSSProperties = {
  width: "min(720px, 94vw)", height: "100%", overflowY: "auto", background: C.bg,
  borderLeft: `1px solid ${C.line2}`, padding: "22px 24px", boxShadow: "-8px 0 32px rgba(20,60,110,0.14)",
};
const closeBtn: React.CSSProperties = {
  border: `1px solid ${C.line2}`, background: C.panel, color: C.dim, borderRadius: 5, padding: "6px 12px", fontSize: 12, cursor: "pointer",
};
const th: React.CSSProperties = { textAlign: "left", padding: "5px 8px", fontSize: 10, letterSpacing: ".08em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "7px 8px", color: C.dim };
const tdN: React.CSSProperties = { ...td, fontFamily: "IBM Plex Mono", textAlign: "right", fontVariantNumeric: "tabular-nums" };
