import { useEffect, useState } from "react";
import { api, type AgentSpec, type Scenario } from "./api";
import { Button, C, Label } from "./ui";
import type { SessionController } from "./useSession";

export default function SessionBar({
  ctrl,
  overrides,
  agents,
  refresh = 0,
  onOpenConfig,
  onOpenEditor,
}: {
  ctrl: SessionController;
  overrides: Record<string, unknown>;
  agents: AgentSpec[];
  refresh?: number;
  onOpenConfig?: () => void;
  onOpenEditor?: () => void;
}) {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [seed, setSeed] = useState(42);
  const overrideCount = Object.keys(overrides).length;
  const humans = agents.filter((a) => a.type === "Umano").length;
  const namesOk = new Set(agents.map((a) => a.name.trim())).size === agents.length && agents.every((a) => a.name.trim());

  useEffect(() => {
    api
      .scenarios()
      .then((s) => {
        setScenarios(s);
        // primo caricamento o scenario selezionato sparito (custom eliminato)
        setScenarioId((cur) => (s.some((x) => x.scenario_id === cur) ? cur : s[0]?.scenario_id ?? ""));
      })
      .catch(() => undefined);
  }, [refresh]);

  const scenario = scenarios.find((s) => s.scenario_id === scenarioId);

  return (
    <div>
      <div
        style={{
          display: "flex",
          gap: 18,
          alignItems: "flex-end",
          flexWrap: "wrap",
          background: C.panel,
          border: `1px solid ${C.line2}`,
          borderRadius: 8,
          padding: "16px 20px",
          boxShadow: "0 1px 3px rgba(20,60,110,0.06)",
        }}
      >
        <Field label="Scenario">
          <select value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} className="mono" style={{ ...inputStyle, width: 270 }}>
            {scenarios.map((s) => (
              <option key={s.scenario_id} value={s.scenario_id}>
                {s.name} · {s.asset_type} ({s.asset_count}){s.source === "custom" ? " · custom" : ""}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Manager">
          <div
            onClick={onOpenEditor}
            className="mono"
            style={{ ...inputStyle, width: 230, cursor: onOpenEditor ? "pointer" : "default", display: "flex", alignItems: "center", gap: 8 }}
            title="Apri l'Editor agenti"
          >
            <span style={{ color: C.blueDeep, fontWeight: 600 }}>{agents.length}</span>
            <span style={{ color: C.dim, fontSize: 12 }}>
              agent{agents.length === 1 ? "e" : "i"} · {humans} uman{humans === 1 ? "o" : "i"} · {agents.length - humans} AI
            </span>
            <span style={{ marginLeft: "auto", color: C.blue, fontSize: 11 }}>Editor →</span>
          </div>
        </Field>
        <Field label="Seed">
          <input className="mono" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} style={inputStyle} />
        </Field>
        <Button kind="primary" onClick={() => ctrl.start(scenarioId, agents, seed, overrides)} disabled={ctrl.busy || !scenarioId || !namesOk}>
          {ctrl.state ? "Ricrea sessione" : "Avvia sessione"}
        </Button>
        {overrideCount > 0 && (
          <span
            className="mono"
            onClick={onOpenConfig}
            style={{ fontSize: 11, color: C.blue, background: C.tint, border: `1px solid ${C.celeste}`, borderRadius: 4, padding: "5px 9px", cursor: onOpenConfig ? "pointer" : "default" }}
          >
            {overrideCount} parametr{overrideCount === 1 ? "o" : "i"} modificat{overrideCount === 1 ? "o" : "i"}
          </span>
        )}
        {scenario && <div style={{ fontSize: 12.5, color: C.dim, maxWidth: 300, lineHeight: 1.4 }}>{scenario.description}</div>}
      </div>
      {ctrl.error && (
        <div style={{ marginTop: 14, padding: "12px 16px", background: "#fdecea", border: `1px solid ${C.red}`, borderRadius: 6, color: "#7a2019", fontSize: 13 }}>
          {ctrl.error}
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <Label>{label}</Label>
      <div style={{ marginTop: 5 }}>{children}</div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  background: C.bg,
  color: C.ink,
  border: `1px solid ${C.line2}`,
  borderRadius: 4,
  padding: "8px 10px",
  fontSize: 13,
  width: 110,
};
