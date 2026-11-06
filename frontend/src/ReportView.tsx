import type { SessionState } from "./api";
import { downloadText, toCsv } from "./download";
import { betaColor, Button, C, ChartCard, Label } from "./ui";
import { eur, fmt, pct, sci } from "./useSession";

export default function ReportView({ state }: { state: SessionState | null }) {
  if (!state) return <div style={{ textAlign: "center", padding: 70, color: C.faint }} className="mono">Avvia una sessione per il report.</div>;

  const stamp = `${state.scenario_id}_T${state.kpis.quarter}`;

  function exportHistory() {
    downloadText(`history_${stamp}.csv`, toCsv(state!.history as unknown as Record<string, unknown>[]), "text/csv");
  }
  function exportAssets() {
    const rows = state!.assets.map((a) => ({
      id: a.id, name: a.name, manager: a.manager, status: a.status, min_condition: a.min_condition, health_index: a.health_index,
      reliability_index: a.reliability_index, annual_failure_probability: a.annual_failure_probability,
      net_asset_value: a.net_asset_value, risk_priority_index: a.risk_priority_index, cda: a.cda,
      site_pga: a.seismic?.site_pga ?? "", seismic_pf: a.seismic?.failure_probability ?? "",
    }));
    downloadText(`assets_${stamp}.csv`, toCsv(rows), "text/csv");
  }
  function exportJson() {
    downloadText(`session_${stamp}.json`, JSON.stringify(state, null, 2), "application/json");
  }

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <Label>Esporta</Label>
        <Button kind="primary" onClick={exportHistory}>Storia (CSV)</Button>
        <Button onClick={exportAssets}>Asset (CSV)</Button>
        <Button onClick={exportJson}>Sessione (JSON)</Button>
        <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>
          {state.scenario_id} · {state.asset_type} · {state.agents.map((a) => `${a.name} (${a.is_human ? "umano" : a.type})`).join(", ")} · T{state.kpis.quarter}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 14 }}>
        <Recap label="Trimestri simulati" value={String(state.kpis.quarter)} />
        <Recap label="β minimo" value={fmt(state.kpis.min_reliability_index, 2)} color={betaColor(state.kpis.min_reliability_index)} />
        <Recap label="Health finale" value={pct(state.kpis.health_index)} color={C.green} />
        <Recap label="NAV di rete" value={eur(state.kpis.net_asset_value)} color={C.blueDeep} />
        <Recap label="Emissioni (tCO₂e)" value={fmt(state.kpis.emissions_tco2 ?? 0, 1)} color={C.amber} />
      </div>

      <ChartCard title="Riepilogo asset" style={{ marginTop: 16 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13.5 }}>
          <thead>
            <tr>{["ID", "Nome", "Stato", "Cond.", "Health", "β", "p_f", "NAV", "CdA"].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {state.assets.map((a) => (
              <tr key={a.id} style={{ borderTop: `1px solid ${C.line}` }}>
                <td style={td}>{a.id}</td>
                <td style={{ ...td, color: C.ink, fontWeight: 500 }}>{a.name}</td>
                <td style={td}>{a.status}</td>
                <td style={tdN}>{fmt(a.min_condition, 3)}</td>
                <td style={tdN}>{pct(a.health_index)}</td>
                <td style={{ ...tdN, color: betaColor(a.reliability_index) }}>{fmt(a.reliability_index, 2)}</td>
                <td style={tdN}>{sci(a.annual_failure_probability)}</td>
                <td style={tdN}>{eur(a.net_asset_value)}</td>
                <td style={td}>{a.cda ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ChartCard>
    </div>
  );
}

function Recap({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "14px 16px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
      <Label>{label}</Label>
      <div className="mono" style={{ fontSize: 24, fontWeight: 600, color: color ?? C.ink, marginTop: 8 }}>{value}</div>
    </div>
  );
}

const th: React.CSSProperties = { textAlign: "left", padding: "6px 10px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "8px 10px", color: C.dim };
const tdN: React.CSSProperties = { ...td, fontFamily: "IBM Plex Mono", textAlign: "right", fontVariantNumeric: "tabular-nums" };
