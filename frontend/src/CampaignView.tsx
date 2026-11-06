import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AGENT_TYPES, CAMPAIGN_SWEEP_PARAMS, campaignApi, api, pluginApi, type CampaignAgentObj, type CampaignResult, type CodeFrameworkInfo, type PluginInfo, type Scenario } from "./api";
import { axisTick, betaColor, Button, C, ChartCard, Label, tooltipStyle } from "./ui";

const LLM_AGENTS = ["LLM", "LLM Agentico"];

type Metric = "mean_beta_margin" | "mean_health" | "mean_reward" | "failure_rate" | "mean_emissions_tco2";
const METRICS: { key: Metric; label: string }[] = [
  { key: "mean_beta_margin", label: "Margine β medio" },
  { key: "mean_health", label: "Health medio" },
  { key: "mean_reward", label: "Reward medio" },
  { key: "failure_rate", label: "Tasso di fallimento" },
  { key: "mean_emissions_tco2", label: "Emissioni medie (tCO₂e)" },
];

export default function CampaignView() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [agents, setAgents] = useState<string[]>(["Euristico"]);
  const [llmAgents, setLlmAgents] = useState<string[]>([]);
  const [llmProvider, setLlmProvider] = useState("Locale");
  const [customAgents, setCustomAgents] = useState<string[]>([]);
  const [agentPlugins, setAgentPlugins] = useState<PluginInfo[]>([]);
  const [replicates, setReplicates] = useState(3);
  const [horizon, setHorizon] = useState(12);
  const [gridParam, setGridParam] = useState("");
  const [gridValues, setGridValues] = useState("");
  const [frameworks, setFrameworks] = useState<CodeFrameworkInfo[]>([]);
  const [compareFw, setCompareFw] = useState<string[]>([]);
  const [metric, setMetric] = useState<Metric>("mean_beta_margin");

  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [result, setResult] = useState<CampaignResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const poll = useRef<number | null>(null);

  useEffect(() => {
    api.scenarios().then((s) => {
      setScenarios(s);
      if (s.length) setScenarioId(s[0].scenario_id);
    });
    api.codes().then((c) => setFrameworks(c.filter((f) => f.name !== "custom"))).catch(() => undefined);
    pluginApi.list("agent").then(setAgentPlugins).catch(() => undefined);
    return () => { if (poll.current) window.clearInterval(poll.current); };
  }, []);

  function toggleIn(setter: React.Dispatch<React.SetStateAction<string[]>>, a: string) {
    setter((prev) => (prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]));
  }
  function toggleAgent(a: string) { toggleIn(setAgents, a); }

  // Costruisce la lista agenti mista (stringhe built-in + spec LLM/Custom).
  function buildAgentSpecs(): (string | CampaignAgentObj)[] {
    const out: (string | CampaignAgentObj)[] = [...agents];
    for (const t of llmAgents) out.push({ id: `${t} · ${llmProvider}`, type: t, provider: llmProvider, profile: "Bilanciato" });
    for (const p of customAgents) out.push({ id: p, type: "Custom", plugin: p });
    return out;
  }
  const totalAgents = agents.length + llmAgents.length + customAgents.length;
  function toggleFw(f: string) {
    setCompareFw((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]));
  }

  async function launch() {
    setError(null);
    setResult(null);
    const values = gridValues.split(",").map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
    try {
      const { job_id, run_count } = await campaignApi.start({
        scenario_id: scenarioId,
        agents: buildAgentSpecs(),
        horizon_quarters: horizon,
        replicates,
        grid_param: compareFw.length ? null : gridParam || null,
        grid_values: compareFw.length || !gridParam ? [] : values,
        compare_frameworks: compareFw,
      });
      setStatus("running");
      setProgress({ done: 0, total: run_count });
      if (poll.current) window.clearInterval(poll.current);
      poll.current = window.setInterval(async () => {
        const st = await campaignApi.status(job_id);
        setProgress({ done: st.done, total: st.total });
        if (st.status === "done") {
          window.clearInterval(poll.current!);
          setStatus("done");
          setResult(st.result);
        } else if (st.status === "error") {
          window.clearInterval(poll.current!);
          setStatus("error");
          setError(st.error);
        }
      }, 1200);
    } catch (e) {
      setError(String(e));
    }
  }

  const chartData = (result?.aggregates ?? []).map((a) => ({
    label: a.grid_value != null ? `${a.agent} · ${a.grid_value}` : a.agent,
    value: a[metric] ?? 0,
    agent: a.agent,
  }));
  const metricLabel = METRICS.find((m) => m.key === metric)!.label;
  const running = status === "running";

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "18px 20px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
        <div style={{ display: "flex", gap: 22, flexWrap: "wrap", alignItems: "flex-start" }}>
          <div>
            <Label>Scenario</Label>
            <select value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} className="mono" style={{ ...input, width: 250, marginTop: 5 }}>
              {scenarios.map((s) => <option key={s.scenario_id} value={s.scenario_id}>{s.name} · {s.asset_type}</option>)}
            </select>
          </div>
          <div>
            <Label>Agenti a confronto</Label>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
              {AGENT_TYPES.map((a) => (
                <label key={a} className="mono" style={{ fontSize: 12, color: C.dim, display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={agents.includes(a)} onChange={() => toggleAgent(a)} />
                  {a}
                  {a !== "Euristico" && <span style={{ color: C.amber, fontSize: 10 }}>lento</span>}
                </label>
              ))}
            </div>
          </div>
          <div>
            <Label>Agenti LLM / codice</Label>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
              {LLM_AGENTS.map((a) => (
                <label key={a} className="mono" style={{ fontSize: 12, color: C.dim, display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={llmAgents.includes(a)} onChange={() => toggleIn(setLlmAgents, a)} />
                  {a} <span style={{ color: C.amber, fontSize: 10 }}>lento/costo</span>
                </label>
              ))}
              {llmAgents.length > 0 && (
                <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)} className="mono" style={{ ...input, width: 130, marginTop: 2, fontSize: 11.5, padding: "5px 8px" }}>
                  {["Locale", "OpenAI", "Google"].map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              )}
              {agentPlugins.map((p) => (
                <label key={p.name} className="mono" style={{ fontSize: 12, color: C.dim, display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={customAgents.includes(p.name)} onChange={() => toggleIn(setCustomAgents, p.name)} />
                  {p.name} <span style={{ color: C.celeste, fontSize: 10 }}>custom</span>
                </label>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div><Label>Repliche</Label><input type="number" value={replicates} min={1} max={15} onChange={(e) => setReplicates(Number(e.target.value))} className="mono" style={{ ...input, width: 90, marginTop: 5 }} /></div>
            <div><Label>Orizzonte (trim.)</Label><input type="number" value={horizon} min={2} max={60} onChange={(e) => setHorizon(Number(e.target.value))} className="mono" style={{ ...input, width: 90, marginTop: 5 }} /></div>
          </div>
          <div>
            <Label>Sweep parametro (opzionale)</Label>
            <select value={gridParam} onChange={(e) => setGridParam(e.target.value)} disabled={compareFw.length > 0} className="mono" style={{ ...input, width: 250, marginTop: 5, opacity: compareFw.length ? 0.5 : 1 }}>
              <option value="">— nessuno —</option>
              {CAMPAIGN_SWEEP_PARAMS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            {gridParam && !compareFw.length && (
              <input placeholder="valori, es. 0.08, 0.16" value={gridValues} onChange={(e) => setGridValues(e.target.value)} className="mono" style={{ ...input, width: 250, marginTop: 8, display: "block" }} />
            )}
          </div>
          <div>
            <Label>Confronta normative</Label>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
              {frameworks.map((f) => (
                <label key={f.name} className="mono" style={{ fontSize: 12, color: C.dim, display: "flex", gap: 7, alignItems: "center", cursor: "pointer" }}>
                  <input type="checkbox" checked={compareFw.includes(f.name)} onChange={() => toggleFw(f.name)} />
                  {f.name}
                  <span style={{ color: C.faint, fontSize: 10 }}>β{f.beta_reference_by_cc?.CC2 ?? "?"}·{f.reference_period_years}a</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 16 }}>
          <Button kind="primary" onClick={launch} disabled={running || !totalAgents || !scenarioId}>
            {running ? "In corso…" : "Lancia campagna"}
          </Button>
          {progress && (
            <span className="mono" style={{ fontSize: 12, color: C.dim }}>
              {progress.done}/{progress.total} run
            </span>
          )}
          {running && (
            <div style={{ flex: 1, maxWidth: 240, height: 6, background: C.line, borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${progress ? (progress.done / Math.max(1, progress.total)) * 100 : 0}%`, height: "100%", background: C.blue, transition: "width .3s" }} />
            </div>
          )}
          <span className="mono" style={{ fontSize: 11, color: C.faint }}>
            {totalAgents} agenti × {compareFw.length ? compareFw.length : gridParam && gridValues ? gridValues.split(",").filter((v) => v.trim()).length : 1} {compareFw.length ? "normative" : "valori"} × {replicates} repliche
          </span>
        </div>
      </div>

      {error && <div style={{ marginTop: 14, padding: "12px 16px", background: "#fdecea", border: `1px solid ${C.red}`, borderRadius: 6, color: "#7a2019", fontSize: 12.5, whiteSpace: "pre-wrap", maxHeight: 200, overflow: "auto" }}>{error}</div>}

      {result && (
        <>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 16, marginTop: 20 }}>
            <div>
              <Label>Metrica di confronto</Label>
              <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)} className="mono" style={{ ...input, width: 220, marginTop: 5 }}>
                {METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </div>
            <span className="mono" style={{ fontSize: 12, color: C.faint }}>{result.completed} run · {result.failed} falliti</span>
          </div>

          <ChartCard title={`${metricLabel} per agente${result.grid_param ? " · " + result.grid_param : ""}`} style={{ marginTop: 12 }}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} margin={{ top: 20, right: 16, bottom: 30, left: -4 }}>
                <CartesianGrid stroke={C.line} vertical={false} />
                <XAxis dataKey="label" tick={{ ...axisTick, fontSize: 10 }} stroke={C.line2} angle={-12} textAnchor="end" interval={0} height={50} />
                <YAxis tick={axisTick} stroke={C.line2} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={metric === "mean_beta_margin" ? (d.value >= 0 ? C.green : C.red) : metric === "failure_rate" ? C.red : C.blue} />
                  ))}
                  <LabelList dataKey="value" position="top" formatter={(v: unknown) => Number(v).toFixed(2)} style={{ fontFamily: "IBM Plex Mono", fontSize: 10, fill: C.dim }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Aggregati" style={{ marginTop: 16 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr>{["Agente", result.grid_param ?? "", "Run", "Reward", "Health", "β margin", "± β", "CO₂ t", "Trim.", "Fail"].map((h, i) => <th key={i} style={th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {result.aggregates.map((a, i) => (
                  <tr key={i} style={{ borderTop: `1px solid ${C.line}` }}>
                    <td style={{ ...td, color: C.ink, fontWeight: 500, textAlign: "left" }}>{a.agent}</td>
                    <td style={tdN}>{a.grid_value ?? "—"}</td>
                    <td style={tdN}>{a.runs}</td>
                    <td style={tdN}>{a.mean_reward?.toFixed(2) ?? "—"}</td>
                    <td style={tdN}>{a.mean_health != null ? (a.mean_health * 100).toFixed(1) + "%" : "—"}</td>
                    <td style={{ ...tdN, color: betaColor((a.mean_beta_margin ?? 0) + 3.8) }}>{a.mean_beta_margin?.toFixed(2) ?? "—"}</td>
                    <td style={tdN}>{a.std_beta_margin.toFixed(2)}</td>
                    <td style={tdN}>{a.mean_emissions_tco2?.toFixed(1) ?? "—"}</td>
                    <td style={tdN}>{a.mean_quarters?.toFixed(1) ?? "—"}</td>
                    <td style={{ ...tdN, color: a.failure_rate > 0 ? C.red : C.dim }}>{(a.failure_rate * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ChartCard>

          {result.stats && result.stats.metrics.length > 0 && (
            <ChartCard title="Confronto statistico fra condizioni — CI bootstrap 95%, Mann-Whitney, effect size" style={{ marginTop: 16 }}>
              {result.stats.metrics.map((m) => (
                <div key={m.key} style={{ marginBottom: 18 }}>
                  <div className="mono" style={{ fontSize: 12, color: C.blueDeep, fontWeight: 600, marginBottom: 6 }}>{m.label}</div>
                  <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 8 }}>
                    {Object.entries(m.groups).map(([name, g]) => g && (
                      <span key={name} className="mono" style={{ fontSize: 11.5, color: C.dim }}>
                        <b style={{ color: C.ink }}>{name}</b>: {g.mean.toFixed(2)} <span style={{ color: C.faint }}>[{g.lo.toFixed(2)}, {g.hi.toFixed(2)}] n={g.n}</span>
                      </span>
                    ))}
                  </div>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                    <thead>
                      <tr>{["Confronto", "Δ media", "p (MWU)", "Cohen d", "Cliff δ", "Effetto"].map((h, i) => <th key={i} style={{ ...th, textAlign: i === 0 ? "left" : "right" }}>{h}</th>)}</tr>
                    </thead>
                    <tbody>
                      {m.comparisons.map((c, i) => (
                        <tr key={i} style={{ borderTop: `1px solid ${C.line}` }}>
                          <td style={{ ...td, textAlign: "left", color: C.ink }}>{c.a} vs {c.b}</td>
                          <td style={tdN}>{c.mean_diff.toFixed(2)}</td>
                          <td style={{ ...tdN, color: c.significant ? C.green : C.dim, fontWeight: c.significant ? 700 : 400 }}>
                            {c.p_value == null ? "—" : c.p_value.toFixed(4)}{c.significant ? " *" : ""}
                          </td>
                          <td style={tdN}>{c.cohens_d ?? "—"}</td>
                          <td style={tdN}>{c.cliffs_delta ?? "—"}</td>
                          <td style={{ ...td, textAlign: "right", color: C.dim }}>{c.effect_magnitude ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
              <div className="mono" style={{ fontSize: 11, color: C.faint }}>
                * differenza significativa a α=0.05 (test di Mann-Whitney bilaterale). Cliff δ: effetto non parametrico.
              </div>
            </ChartCard>
          )}
        </>
      )}
    </div>
  );
}

const input: React.CSSProperties = { background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "8px 10px", fontSize: 13 };
const th: React.CSSProperties = { textAlign: "right", padding: "6px 8px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "7px 8px", color: C.dim, textAlign: "right" };
const tdN: React.CSSProperties = { ...td, fontFamily: "IBM Plex Mono", fontVariantNumeric: "tabular-nums" };
