import { useState } from "react";
import { Area, Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SessionState } from "./api";
import { axisTick, C, ChartCard, Label, tooltipStyle } from "./ui";

type MetricKey = "budget" | "health_index" | "net_asset_value" | "min_reliability_index" | "max_annual_pf";
const METRICS: { key: MetricKey; label: string }[] = [
  { key: "min_reliability_index", label: "β minimo" },
  { key: "health_index", label: "Health index" },
  { key: "budget", label: "Budget" },
  { key: "net_asset_value", label: "NAV di rete" },
  { key: "max_annual_pf", label: "p_f annuo (max)" },
];

export default function PlottingView({ state }: { state: SessionState | null }) {
  const [yKey, setYKey] = useState<MetricKey>("min_reliability_index");
  const [y2Key, setY2Key] = useState<MetricKey | "">("");
  const [kind, setKind] = useState<"line" | "area" | "bar">("line");

  if (!state) return <div style={{ textAlign: "center", padding: 70, color: C.faint }} className="mono">Avvia una sessione per il plotting.</div>;

  const data = state.history;
  const yLabel = METRICS.find((m) => m.key === yKey)!.label;
  const y2Label = y2Key ? METRICS.find((m) => m.key === y2Key)!.label : "";

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", gap: 22, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
        <Picker label="Metrica (asse sx)" value={yKey} onChange={(v) => setYKey(v as MetricKey)} options={METRICS.map((m) => [m.key, m.label])} />
        <Picker label="Tipo" value={kind} onChange={(v) => setKind(v as typeof kind)} options={[["line", "Linea"], ["area", "Area"], ["bar", "Barre"]]} />
        <Picker label="Seconda serie (asse dx)" value={y2Key} onChange={(v) => setY2Key(v as MetricKey | "")} options={[["", "— nessuna —"], ...METRICS.filter((m) => m.key !== yKey).map((m) => [m.key, m.label] as [string, string])]} />
      </div>

      <ChartCard title={`${yLabel}${y2Label ? " · " + y2Label : ""} — nel tempo (trimestri)`}>
        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={data} margin={{ top: 12, right: 24, bottom: 8, left: 4 }}>
            <CartesianGrid stroke={C.line} />
            <XAxis dataKey="quarter" tick={axisTick} stroke={C.line2} label={{ value: "trimestre", position: "insideBottom", offset: -2, fontSize: 10, fill: C.faint }} />
            <YAxis yAxisId="l" tick={axisTick} stroke={C.line2} domain={["auto", "auto"]} />
            {y2Key && <YAxis yAxisId="r" orientation="right" tick={axisTick} stroke={C.line2} domain={["auto", "auto"]} />}
            <Tooltip contentStyle={tooltipStyle} />
            {kind === "line" && <Line yAxisId="l" type="monotone" dataKey={yKey} name={yLabel} stroke={C.blue} strokeWidth={2.2} dot={false} />}
            {kind === "area" && (
              <>
                <defs>
                  <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={C.celeste} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={C.celeste} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <Area yAxisId="l" type="monotone" dataKey={yKey} name={yLabel} stroke={C.blue} strokeWidth={2} fill="url(#pg)" />
              </>
            )}
            {kind === "bar" && <Bar yAxisId="l" dataKey={yKey} name={yLabel} fill={C.celeste} />}
            {y2Key && <Line yAxisId="r" type="monotone" dataKey={y2Key} name={y2Label} stroke={C.amber} strokeWidth={2} strokeDasharray="5 3" dot={false} />}
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ marginTop: 8, fontSize: 12, color: C.faint }}>
          Sorgente: log di sessione ({data.length} punti). Le serie con scale diverse usano un secondo asse a destra.
        </div>
      </ChartCard>
    </div>
  );
}

function Picker({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: [string, string][] }) {
  return (
    <div>
      <Label>{label}</Label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="mono" style={{ marginTop: 5, background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "8px 10px", fontSize: 13, minWidth: 190 }}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>{l}</option>
        ))}
      </select>
    </div>
  );
}
