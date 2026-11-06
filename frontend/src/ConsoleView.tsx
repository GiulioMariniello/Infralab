import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { type AssetView, type Kpis } from "./api";
import CopilotPanel from "./CopilotPanel";
import { axisTick, betaColor, Button, C, ChartCard, Label, tooltipStyle } from "./ui";
import { eur, fmt, pct, sci, type SessionController } from "./useSession";

const AGENT_COLORS = ["#1f6fb2", "#2e9e6b", "#e0902f", "#8a5bd6", "#d1544a", "#4aa3df"];

export default function ConsoleView({ ctrl, manualOnly = false }: { ctrl: SessionController; manualOnly?: boolean }) {
  const { state, busy } = ctrl;
  const [mode, setMode] = useState<"auto" | "manual">(manualOnly ? "manual" : "auto");
  const [plan, setPlan] = useState<Record<string, string>>({});
  const [scope, setScope] = useState<string>("__net__");

  const planCost = useMemo(() => {
    if (!state) return 0;
    let tot = 0;
    for (const a of state.assets) {
      const chosen = plan[String(a.id)];
      if (chosen) tot += a.actions.find((x) => x.action === chosen)?.cost ?? 0;
    }
    return tot;
  }, [plan, state]);

  if (!state) return <Placeholder />;

  const multi = state.agents.length > 1;
  const scopeAgent = state.agents.find((a) => a.name === scope);
  const kpis: Kpis = scopeAgent ? scopeAgent.kpis : state.kpis;
  const visibleAssets = scopeAgent ? state.assets.filter((a) => a.manager === scope) : state.assets;

  async function applyPlan() {
    await ctrl.act(plan);
    setPlan({});
  }

  return (
    <>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 4, flexWrap: "wrap" }}>
        <Label>Modalità</Label>
        <div style={{ display: "flex", border: `1px solid ${C.line2}`, borderRadius: 5, overflow: "hidden" }}>
          {!manualOnly && <ModeBtn active={mode === "auto"} onClick={() => setMode("auto")}>Automatico</ModeBtn>}
          <ModeBtn active={mode === "manual"} onClick={() => setMode("manual")}>Manuale</ModeBtn>
        </div>
        {mode === "auto" ? (
          <>
            <Button onClick={() => ctrl.step(1)} disabled={busy || !state.kpis.active}>+1 trimestre</Button>
            <Button onClick={() => ctrl.step(4)} disabled={busy || !state.kpis.active}>+1 anno</Button>
            <Button onClick={() => ctrl.step(20)} disabled={busy || !state.kpis.active}>+20 trimestri</Button>
          </>
        ) : (
          <>
            <Button kind="primary" onClick={applyPlan} disabled={busy || !state.kpis.active}>Applica e avanza →</Button>
            <span className="mono" style={{ fontSize: 12, color: planCost > state.kpis.budget ? C.red : C.dim }}>costo piano {eur(planCost)}</span>
          </>
        )}
        <span className="mono" style={{ fontSize: 12, color: C.faint, marginLeft: 4 }}>
          T{state.kpis.quarter}
          {!state.kpis.active && <b style={{ color: C.red }}> · TERMINATO</b>}
        </span>
      </div>

      {/* Selettore scope: rete o singolo manager */}
      {multi && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
          <Label>Vista</Label>
          <ScopeChip active={scope === "__net__"} color={C.blueDeep} onClick={() => setScope("__net__")}>
            Rete (tutti)
          </ScopeChip>
          {state.agents.map((a, i) => (
            <ScopeChip key={a.name} active={scope === a.name} color={AGENT_COLORS[i % AGENT_COLORS.length]} onClick={() => setScope(a.name)}>
              {a.name} · {a.is_human ? "umano" : a.type}
              {!a.kpis.active && " ✕"}
            </ScopeChip>
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginTop: 16 }}>
        <Kpi label="Reliability β (min)" value={fmt(kpis.min_reliability_index, 2)} tone="blue" sub="target CC2 · 3.80" />
        <Kpi label="p_f annuo (max)" value={sci(kpis.max_annual_pf)} tone="celeste" sub="series system" />
        <Kpi label="Health index" value={pct(kpis.health_index)} tone="green" sub="ADT-weighted" />
        <Kpi label="Budget" value={eur(kpis.budget)} tone="ink" sub={`NAV ${eur(kpis.net_asset_value)}`} />
      </div>

      {/* Leaderboard multi-agente */}
      {multi && (
        <ChartCard title="Manager a confronto" style={{ marginTop: 16 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr>{["Manager", "Tipo", "Asset", "Budget", "Health", "β min", "CO₂ (t)", "Stato"].map((h) => <th key={h} style={thL}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {state.agents.map((a, i) => (
                <tr key={a.name} style={{ borderTop: `1px solid ${C.line}` }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>
                    <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: "50%", background: AGENT_COLORS[i % AGENT_COLORS.length], marginRight: 8 }} />
                    {a.name}
                  </td>
                  <td style={tdStyle}>{a.is_human ? "Umano" : a.type}</td>
                  <td style={tdNum}>{a.kpis.asset_count ?? "—"}</td>
                  <td style={tdNum}>{eur(a.kpis.budget)}</td>
                  <td style={tdNum}>{pct(a.kpis.health_index)}</td>
                  <td style={{ ...tdNum, color: betaColor(a.kpis.min_reliability_index) }}>{fmt(a.kpis.min_reliability_index, 2)}</td>
                  <td style={tdNum}>{fmt(a.kpis.emissions_tco2, 1)}</td>
                  <td style={{ ...tdStyle, color: a.kpis.active ? C.green : C.red }}>{a.kpis.active ? "attivo" : "fallito"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ChartCard>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 18 }}>
        <ChartCard title={multi ? "Affidabilità β per manager" : "Affidabilità β nel tempo"}>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={state.history} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
              <CartesianGrid stroke={C.line} />
              <XAxis dataKey="quarter" tick={axisTick} stroke={C.line2} />
              <YAxis tick={axisTick} stroke={C.line2} domain={["auto", "auto"]} />
              <Tooltip contentStyle={tooltipStyle} />
              <ReferenceLine y={3.8} stroke={C.amber} strokeDasharray="4 3" label={{ value: "target 3.8", fontSize: 10, fill: C.amber }} />
              {multi ? (
                <>
                  {state.agents.map((a, i) => (
                    <Line
                      key={a.name}
                      type="monotone"
                      dataKey={(row: Record<string, unknown>) => {
                        const ag = (row as { agents?: Record<string, Kpis> }).agents;
                        return ag?.[a.name]?.min_reliability_index ?? null;
                      }}
                      name={a.name}
                      stroke={AGENT_COLORS[i % AGENT_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                  <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11 }} />
                </>
              ) : (
                <Line type="monotone" dataKey="min_reliability_index" name="β min" stroke={C.blue} strokeWidth={2.2} dot={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Health index nel tempo (rete)">
          <ResponsiveContainer width="100%" height={230}>
            <AreaChart data={state.history} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
              <defs>
                <linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.celeste} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={C.celeste} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={C.line} />
              <XAxis dataKey="quarter" tick={axisTick} stroke={C.line2} />
              <YAxis tick={axisTick} stroke={C.line2} domain={[0, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="health_index" name="health" stroke={C.blue} strokeWidth={2} fill="url(#hg)" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {mode === "manual" && (
        <CopilotPanel
          sid={state.session_id}
          agents={state.agents.filter((a) => a.is_human).map((a) => a.name)}
          onApply={(p) => {
            // La proposta riempie i selettori: l'umano rivede e poi applica lui.
            const valid: Record<string, string> = {};
            for (const [aid, action] of Object.entries(p)) {
              const asset = state.assets.find((x) => String(x.id) === String(aid));
              if (asset && asset.actions.some((x) => x.action === action && x.feasible)) valid[String(aid)] = action;
            }
            setPlan((prev) => ({ ...prev, ...valid }));
          }}
        />
      )}

      <ChartCard title={`Asset — ${state.asset_type}${scopeAgent ? " · " + scopeAgent.name : ""}`} style={{ marginTop: 16 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13.5 }}>
          <thead>
            <tr>
              {["ID", "Nome", ...(multi ? ["Manager"] : []), "Stato", "Cond. min", "Health", "β", "p_f annuo", "NAV", mode === "manual" ? "Intervento" : ""].map((h, i) => (
                <th key={i} className="mono" style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleAssets.map((a) => (
              <tr key={a.id} style={{ borderTop: `1px solid ${C.line}` }}>
                <td style={tdStyle}>{a.id}</td>
                <td style={{ ...tdStyle, color: C.ink, fontWeight: 500 }}>{a.name}</td>
                {multi && <td style={tdStyle}>{a.manager}</td>}
                <td style={tdStyle}>{a.status}</td>
                <td style={tdNum}>{fmt(a.min_condition, 3)}</td>
                <td style={tdNum}>{pct(a.health_index)}</td>
                <td style={{ ...tdNum, color: betaColor(a.reliability_index) }}>{fmt(a.reliability_index, 2)}</td>
                <td style={tdNum}>{sci(a.annual_failure_probability)}</td>
                <td style={tdNum}>{eur(a.net_asset_value)}</td>
                {mode === "manual" && (
                  <td style={{ padding: "5px 10px" }}>
                    <ActionSelect asset={a} value={plan[String(a.id)] ?? ""} onChange={(v) => setPlan((p) => ({ ...p, [String(a.id)]: v }))} />
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </ChartCard>
    </>
  );
}

function Placeholder() {
  return (
    <div style={{ textAlign: "center", padding: "70px 20px", color: C.faint }}>
      <div className="mono" style={{ fontSize: 13, letterSpacing: ".1em" }}>
        Avvia una sessione per iniziare — scenario e manager qui sopra, composizione nell'Editor.
      </div>
    </div>
  );
}

function ScopeChip({ active, color, onClick, children }: { active: boolean; color: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className="mono"
      style={{
        fontSize: 11.5, padding: "6px 12px", borderRadius: 14, border: `1px solid ${active ? color : C.line2}`,
        background: active ? color : C.panel, color: active ? "#fff" : C.dim, fontWeight: active ? 600 : 500,
      }}
    >
      {children}
    </button>
  );
}

function ModeBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} style={{ border: "none", padding: "9px 16px", fontSize: 12, letterSpacing: ".06em", textTransform: "uppercase", background: active ? C.blue : "transparent", color: active ? "#fff" : C.dim, fontWeight: active ? 600 : 500 }}>
      {children}
    </button>
  );
}

function ActionSelect({ asset, value, onChange }: { asset: AssetView; value: string; onChange: (v: string) => void }) {
  if (asset.busy) return <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>in corso…</span>;
  const feasible = asset.actions.filter((x) => x.feasible);
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="mono" style={{ background: value ? C.tint : C.bg, color: value ? C.blueDeep : C.dim, border: `1px solid ${value ? C.celeste : C.line2}`, borderRadius: 4, padding: "6px 8px", fontSize: 12, minWidth: 190 }}>
      <option value="">— nessun intervento —</option>
      {feasible.map((x) => (
        <option key={x.action} value={x.action}>
          {x.action} · €{Math.round(x.cost).toLocaleString("it-IT")}
        </option>
      ))}
    </select>
  );
}

function Kpi({ label, value, sub, tone }: { label: string; value: string; sub: string; tone: "blue" | "celeste" | "green" | "ink" }) {
  const color = { blue: C.blueDeep, celeste: C.blue, green: C.green, ink: C.ink }[tone];
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "14px 16px", position: "relative", overflow: "hidden", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: 3, height: "100%", background: C.celeste }} />
      <Label>{label}</Label>
      <div className="mono" style={{ fontSize: 26, fontWeight: 600, color, marginTop: 8, letterSpacing: "-.01em" }}>{value}</div>
      <div className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 5 }}>{sub}</div>
    </div>
  );
}

const thStyle: React.CSSProperties = { textAlign: "left", padding: "6px 10px", fontSize: 10.5, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const thL = thStyle;
const tdStyle: React.CSSProperties = { padding: "8px 10px", color: C.dim };
const tdNum: React.CSSProperties = { ...tdStyle, fontFamily: "IBM Plex Mono", textAlign: "right", fontVariantNumeric: "tabular-nums" };
