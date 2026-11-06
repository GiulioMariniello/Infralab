import { useEffect, useState } from "react";
import { api, type AssetType, type SessionState } from "./api";
import AssetMap from "./AssetMap";
import { betaColor, C, ChartCard } from "./ui";
import { eur, fmt, pct } from "./useSession";

export default function NetworkView({ state }: { state: SessionState | null }) {
  const [icon, setIcon] = useState("");
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    if (!state) return;
    api.assetTypes()
      .then((ts: AssetType[]) => setIcon(ts.find((t) => t.name === state.asset_type)?.ui_meta?.icon ?? ""))
      .catch(() => setIcon(""));
  }, [state?.asset_type]);

  if (!state) return <Empty />;
  const byRisk = [...state.assets].sort((x, y) => y.risk_priority_index - x.risk_priority_index);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16, marginTop: 4 }}>
      <ChartCard title={`Mappa della rete — ${state.asset_type} · colore = β, raggio = importanza`}>
        <AssetMap assets={state.assets} height={430} icon={icon} selectedId={selected} onSelect={setSelected} />
        <div style={{ display: "flex", gap: 18, marginTop: 10 }}>
          <LegendDot color={C.green} label="β ≥ 3.8" />
          <LegendDot color={C.amber} label="3.0 – 3.8" />
          <LegendDot color={C.red} label="β < 3.0" />
        </div>
      </ChartCard>

      <ChartCard title="Priorità di rischio (rete)">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {["Asset", "β", "Rischio", "Imp.", "Cond."].map((h) => (
                <th key={h} className="mono" style={th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {byRisk.map((a) => (
              <tr
                key={a.id}
                onClick={() => setSelected(a.id)}
                style={{ borderTop: `1px solid ${C.line}`, cursor: "pointer", background: a.id === selected ? C.tint : "transparent" }}
              >
                <td style={{ ...td, color: C.ink, fontWeight: 500 }}>{a.name}</td>
                <td style={{ ...tdN, color: betaColor(a.reliability_index) }}>{fmt(a.reliability_index, 2)}</td>
                <td style={tdN}>
                  <RiskBar v={a.risk_priority_index} max={byRisk[0].risk_priority_index || 1} />
                </td>
                <td style={tdN}>{fmt(a.network_importance, 2)}</td>
                <td style={tdN}>{pct(a.min_condition)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: 12, fontSize: 12, color: C.faint }}>
          Rischio = importanza di rete × p_f × severità della condizione. NAV totale:{" "}
          <b className="mono" style={{ color: C.blueDeep }}>{eur(state.kpis.net_asset_value)}</b>
        </div>
      </ChartCard>
    </div>
  );
}

function RiskBar({ v, max }: { v: number; max: number }) {
  const w = Math.max(2, Math.min(100, (v / (max || 1)) * 100));
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 6, width: "100%", justifyContent: "flex-end" }}>
      <div style={{ flex: 1, height: 6, background: C.line, borderRadius: 3, overflow: "hidden", maxWidth: 90 }}>
        <div style={{ width: `${w}%`, height: "100%", background: w > 66 ? C.red : w > 33 ? C.amber : C.celeste }} />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="mono" style={{ fontSize: 11, color: C.dim, display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span style={{ width: 10, height: 10, borderRadius: "50%", background: color }} /> {label}
    </span>
  );
}

function Empty() {
  return <div style={{ textAlign: "center", padding: "70px", color: C.faint }} className="mono">Avvia una sessione per vedere la rete.</div>;
}

const th: React.CSSProperties = { textAlign: "left", padding: "6px 8px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "7px 8px", color: C.dim, textAlign: "right" };
const tdN: React.CSSProperties = { ...td, fontFamily: "IBM Plex Mono", fontVariantNumeric: "tabular-nums" };
