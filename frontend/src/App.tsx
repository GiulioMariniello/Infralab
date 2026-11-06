import { useEffect, useState } from "react";
import AnalysisView from "./AnalysisView";
import BuilderView from "./BuilderView";
import CampaignView from "./CampaignView";
import ConfigView from "./ConfigView";
import ConsoleView from "./ConsoleView";
import EditorView from "./EditorView";
import Landing from "./Landing";
import NetworkView from "./NetworkView";
import PlayerView from "./PlayerView";
import PluginLab from "./PluginLab";
import PlottingView from "./PlottingView";
import ReportView from "./ReportView";
import ResearchView from "./ResearchView";
import SeismicView from "./SeismicView";
import SessionBar from "./SessionBar";
import { researchApi, type AgentSpec } from "./api";
import { C, Mark, Wordmark } from "./ui";
import { useSession } from "./useSession";

type View = "console" | "editor" | "config" | "network" | "analysis" | "seismic" | "plotting" | "report" | "campaign" | "code" | "research";
const TABS: { id: View; label: string }[] = [
  { id: "console", label: "Console" },
  { id: "editor", label: "Editor" },
  { id: "config", label: "Config" },
  { id: "network", label: "Rete" },
  { id: "analysis", label: "Analisi" },
  { id: "seismic", label: "Sismica" },
  { id: "plotting", label: "Plotting" },
  { id: "report", label: "Report" },
  { id: "campaign", label: "Campagne" },
  { id: "code", label: "Codice" },
  { id: "research", label: "Ricerca" },
];

// Il giocatore ha un'esperienza semplificata: la Plancia (PlayerView, con mappa
// e gemelli digitali) come schermata principale, più il Report.
const PLAYER_TABS: View[] = ["console", "report"];
const PLAYER_TAB_LABEL: Partial<Record<View, string>> = { console: "Plancia" };

export default function App() {
  const [entered, setEntered] = useState(false);
  const [view, setView] = useState<View>("console");
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [agents, setAgents] = useState<AgentSpec[]>([{ name: "Manager 1", type: "Euristico" }]);
  const [adminToken, setAdminToken] = useState<string | null>(null);
  const [scenarioRefresh, setScenarioRefresh] = useState(0);
  const [playerName, setPlayerName] = useState<string | null>(null);
  const [consentGiven, setConsentGiven] = useState(false);
  const [ptoken, setPtoken] = useState<string | null>(null);
  const [joinError, setJoinError] = useState<string | null>(null);
  const ctrl = useSession();

  // Ingresso giocatore via link con token: ?ptoken=...
  useEffect(() => {
    const ptoken = new URLSearchParams(window.location.search).get("ptoken");
    if (!ptoken) return;
    setPtoken(ptoken);
    researchApi
      .playerJoin(ptoken)
      .then((state) => {
        ctrl.adopt(state);
        setPlayerName(state.player_name);
        setConsentGiven(Boolean(state.consent_given));
        setEntered(true);
        setView("console");
      })
      .catch(() => setJoinError("Link di gioco non valido o revocato. Contatta lo sperimentatore."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (joinError) {
    return (
      <div style={{ maxWidth: 480, margin: "120px auto", textAlign: "center" }}>
        <Mark size={40} />
        <div style={{ marginTop: 18, color: C.red, fontSize: 15 }}>{joinError}</div>
      </div>
    );
  }

  if (!entered) return <Landing onEnter={() => setEntered(true)} />;

  const isPlayer = playerName != null;
  const tabs = isPlayer ? TABS.filter((t) => PLAYER_TABS.includes(t.id)) : TABS;

  return (
    <div style={{ minHeight: "100vh" }}>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 40px",
          background: "rgba(245,248,252,0.9)",
          backdropFilter: "blur(8px)",
          borderBottom: `1px solid ${C.line2}`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: isPlayer ? "default" : "pointer" }} onClick={() => !isPlayer && setEntered(false)}>
            <Mark size={26} />
            <Wordmark size={18} />
          </div>
          <nav style={{ display: "flex", gap: 4 }}>
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setView(t.id)}
                style={{
                  border: "none",
                  background: "transparent",
                  padding: "8px 14px",
                  fontSize: 12,
                  letterSpacing: ".1em",
                  textTransform: "uppercase",
                  color: view === t.id ? C.blueDeep : C.faint,
                  borderBottom: `2px solid ${view === t.id ? C.blue : "transparent"}`,
                  fontWeight: view === t.id ? 600 : 500,
                }}
              >
                {(isPlayer && PLAYER_TAB_LABEL[t.id]) || t.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="mono" style={{ fontSize: 11, letterSpacing: ".14em", textTransform: "uppercase", color: C.faint }}>
          {isPlayer
            ? `Giocatore: ${playerName}${ctrl.state ? ` · T${ctrl.state.kpis.quarter}` : ""}`
            : ctrl.state
              ? `${ctrl.state.scenario_id} · T${ctrl.state.kpis.quarter}`
              : "Reliability-based AI"}
        </div>
      </header>

      <main style={{ maxWidth: 1180, margin: "0 auto", padding: "20px 40px 80px" }}>
        {!isPlayer && view !== "research" && (
          <SessionBar ctrl={ctrl} overrides={overrides} agents={agents} refresh={scenarioRefresh} onOpenConfig={() => setView("config")} onOpenEditor={() => setView("editor")} />
        )}
        {isPlayer && (
          <div
            className="mono"
            style={{ fontSize: 12, color: C.blueDeep, background: C.tint, border: `1px solid ${C.celeste}`, borderRadius: 6, padding: "10px 14px" }}
          >
            Sessione di ricerca · {ctrl.state?.scenario_id} — le tue decisioni vengono registrate.
            Nella <b>Plancia</b> scegli gli interventi sulle card, apri il <b>gemello digitale</b> di ogni asset, poi “Applica e avanza”.
          </div>
        )}
        <div style={{ marginTop: 20 }}>
          {view === "console" && (isPlayer ? <PlayerView ctrl={ctrl} consentGiven={consentGiven} onConsent={() => setConsentGiven(true)} ptoken={ptoken} /> : <ConsoleView ctrl={ctrl} />)}
          {view === "editor" && !isPlayer && (
            <>
              <EditorView agents={agents} setAgents={setAgents} />
              <BuilderView onSaved={() => setScenarioRefresh((v) => v + 1)} />
            </>
          )}
          {view === "config" && !isPlayer && <ConfigView overrides={overrides} setOverrides={setOverrides} />}
          {view === "network" && <NetworkView state={ctrl.state} />}
          {view === "analysis" && <AnalysisView state={ctrl.state} />}
          {view === "seismic" && !isPlayer && <SeismicView ctrl={ctrl} />}
          {view === "plotting" && !isPlayer && <PlottingView state={ctrl.state} />}
          {view === "report" && <ReportView state={ctrl.state} />}
          {view === "campaign" && !isPlayer && <CampaignView />}
          {view === "code" && !isPlayer && <PluginLab />}
          {view === "research" && !isPlayer && <ResearchView adminToken={adminToken} setAdminToken={setAdminToken} />}
        </div>
      </main>
    </div>
  );
}
