import { useEffect, useState } from "react";
import { api, researchApi, type BatteryAnalysis, type ExperimentAnalysis, type PlayerLink, type ResearchSession, type Scenario } from "./api";
import { Button, C, ChartCard, Label } from "./ui";

export default function ResearchView({
  adminToken,
  setAdminToken,
}: {
  adminToken: string | null;
  setAdminToken: (t: string | null) => void;
}) {
  if (!adminToken) return <Login onToken={setAdminToken} />;
  return <AdminPanel token={adminToken} onLogout={() => setAdminToken(null)} />;
}

function Login({ onToken }: { onToken: (t: string) => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    try {
      const { token } = await researchApi.adminLogin(password);
      onToken(token);
    } catch {
      setError("Password errata.");
    }
  }

  return (
    <div style={{ maxWidth: 420, margin: "60px auto 0" }}>
      <ChartCard title="Accesso amministratore — modalità ricerca">
        <div style={{ fontSize: 13, color: C.dim, lineHeight: 1.5, marginBottom: 14 }}>
          Gestione esperimenti con soggetti umani: partecipanti, link-partita con token,
          monitoraggio delle sessioni persistite.
        </div>
        <Label>Password</Label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          className="mono"
          style={{ ...input, width: "100%", marginTop: 6 }}
          autoFocus
        />
        {error && <div style={{ color: C.red, fontSize: 12.5, marginTop: 8 }}>{error}</div>}
        <div style={{ marginTop: 14 }}>
          <Button kind="primary" onClick={submit} style={{ width: "100%" }}>Accedi</Button>
        </div>
      </ChartCard>
    </div>
  );
}

function AdminPanel({ token, onLogout }: { token: string; onLogout: () => void }) {
  const [participantsText, setParticipantsText] = useState("");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [links, setLinks] = useState<PlayerLink[]>([]);
  const [condition, setCondition] = useState("Uomo");
  const [conditions, setConditions] = useState<string[]>(["Uomo", "Uomo+LLM"]);
  const [sessions, setSessions] = useState<ResearchSession[]>([]);
  const [analysis, setAnalysis] = useState<ExperimentAnalysis | null>(null);
  const [batteryAnalysis, setBatteryAnalysis] = useState<BatteryAnalysis | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  async function refresh() {
    try {
      const [p, s] = await Promise.all([researchApi.participants(token), researchApi.researchSessions(token)]);
      setParticipantsText(p.participants.join("\n"));
      setSessions(s);
    } catch {
      onLogout();
    }
  }

  useEffect(() => {
    refresh();
    api.scenarios().then((s) => {
      setScenarios(s);
      if (s.length) setScenarioId(s[0].scenario_id);
    });
    researchApi.conditions(token).then((c) => setConditions(c.human_conditions)).catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadAnalysis() {
    try {
      setAnalysis(await researchApi.experiment(token, scenarioId));
    } catch (e) {
      setMessage(String(e));
    }
  }

  async function saveParticipants() {
    const list = participantsText.split("\n").map((s) => s.trim()).filter(Boolean);
    const res = await researchApi.saveParticipants(token, list);
    setParticipantsText(res.participants.join("\n"));
    setMessage(`${res.participants.length} partecipanti salvati.`);
  }

  async function generateLinks() {
    const list = participantsText.split("\n").map((s) => s.trim()).filter(Boolean);
    const res = await researchApi.playerLinks(token, scenarioId, list, condition);
    setLinks(res.links);
  }

  async function generateBattery() {
    const list = participantsText.split("\n").map((s) => s.trim()).filter(Boolean);
    try {
      const res = await researchApi.batteryLinks(token, list);
      setLinks(res.links);
      setMessage(`Batteria 2×2 generata per ${res.n_participants} partecipanti (controbilanciata).`);
    } catch (e) {
      setMessage(String(e));
    }
  }

  async function loadBatteryAnalysis() {
    try {
      setBatteryAnalysis(await researchApi.experimentBattery(token));
    } catch (e) {
      setMessage(String(e));
    }
  }

  function copyLink(link: PlayerLink) {
    const url = window.location.origin + link.path;
    navigator.clipboard?.writeText(url);
    setCopied(link.token);
    setTimeout(() => setCopied(null), 1500);
  }

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
        <Label>Amministrazione esperimenti</Label>
        {message && <span className="mono" style={{ fontSize: 12, color: C.green }}>{message}</span>}
        <span style={{ flex: 1 }} />
        <Button onClick={refresh}>Aggiorna</Button>
        <Button onClick={onLogout}>Esci</Button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 16 }}>
        <ChartCard title="Partecipanti (uno per riga)">
          <textarea
            value={participantsText}
            onChange={(e) => setParticipantsText(e.target.value)}
            className="mono"
            style={{ ...input, width: "100%", height: 180, resize: "vertical", lineHeight: 1.6 }}
            placeholder={"Mario Rossi\nAnna Bianchi"}
          />
          <div style={{ marginTop: 10 }}>
            <Button kind="primary" onClick={saveParticipants}>Salva partecipanti</Button>
          </div>
        </ChartCard>

        <ChartCard title="Link-partita (token per giocatore)">
          <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div>
              <Label>Scenario</Label>
              <select value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} className="mono" style={{ ...input, width: 240, marginTop: 5 }}>
                {scenarios.map((s) => <option key={s.scenario_id} value={s.scenario_id}>{s.name} · {s.asset_type}</option>)}
              </select>
            </div>
            <div>
              <Label>Condizione</Label>
              <select value={condition} onChange={(e) => setCondition(e.target.value)} className="mono" style={{ ...input, width: 140, marginTop: 5 }}>
                {conditions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <Button kind="primary" onClick={generateLinks} disabled={!scenarioId}>Genera link</Button>
            <Button onClick={generateBattery}>Genera batteria 2×2</Button>
          </div>
          {links.length > 0 && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, marginTop: 14 }}>
              <thead>
                <tr>{["Partecipante", "Cond.", "Link", ""].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {links.map((l) => (
                  <tr key={l.token} style={{ borderTop: `1px solid ${C.line}` }}>
                    <td style={{ ...td, fontWeight: 600, color: C.ink }}>{l.participant}</td>
                    <td style={{ ...td, fontFamily: "IBM Plex Mono", fontSize: 11 }}>{l.condition ?? "—"}</td>
                    <td style={{ ...td, fontFamily: "IBM Plex Mono", fontSize: 11.5, color: C.blue }}>{l.path}</td>
                    <td style={td}>
                      <Button onClick={() => copyLink(l)}>{copied === l.token ? "Copiato ✓" : "Copia URL"}</Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </ChartCard>
      </div>

      <ChartCard title="Sessioni di ricerca (persistite su disco)" style={{ marginTop: 16 }}>
        {sessions.length === 0 ? (
          <div style={{ color: C.faint, fontSize: 13 }}>Nessuna sessione ancora.</div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr>{["Sessione", "Scenario", "Fase", "Trimestre", "Giocatore", "Stato"].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.session_id} style={{ borderTop: `1px solid ${C.line}` }}>
                  <td style={{ ...td, fontFamily: "IBM Plex Mono", fontSize: 11.5 }}>{s.session_id}</td>
                  <td style={td}>{s.scenario_id ?? "—"}</td>
                  <td style={td}>{s.phase ?? "—"}</td>
                  <td style={{ ...td, fontFamily: "IBM Plex Mono", textAlign: "right" }}>{s.quarter ?? "—"}</td>
                  <td style={td}>{(s.agents ?? []).join(", ")}</td>
                  <td style={{ ...td, color: s.archived ? C.faint : (s.active_agents ?? []).length ? C.green : C.red }}>
                    {s.archived ? "archiviata" : (s.active_agents ?? []).length ? "attiva" : "terminata"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </ChartCard>

      <ChartCard title="Analisi esperimento — esiti umani per condizione" style={{ marginTop: 16 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <Button kind="primary" onClick={loadAnalysis} disabled={!scenarioId}>Analizza «{scenarioId}»</Button>
          {analysis && <a href={researchApi.experimentCsvUrl(scenarioId)} className="mono" style={{ fontSize: 12.5, color: C.blue, textDecoration: "none", border: `1px solid ${C.celeste}`, borderRadius: 5, padding: "7px 12px", background: C.tint }}>⬇ Esporta CSV</a>}
          <span style={{ fontSize: 12, color: C.faint }}>Le condizioni computazionali (LLM, Euristico…) si aggiungono via Campagne sullo stesso scenario.</span>
        </div>
        {analysis && (
          <div style={{ marginTop: 14 }}>
            <div className="mono" style={{ fontSize: 12, color: C.dim, marginBottom: 10 }}>
              {analysis.n_sessions} sessioni · condizioni: {analysis.conditions.join(", ") || "—"}
            </div>
            {analysis.metrics.map((m) => (
              <div key={m.key} style={{ marginBottom: 16 }}>
                <div className="mono" style={{ fontSize: 12, color: C.blueDeep, fontWeight: 600, marginBottom: 6 }}>{m.label}</div>
                <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 6 }}>
                  {Object.entries(m.groups).map(([name, g]) => g && (
                    <span key={name} className="mono" style={{ fontSize: 11.5, color: C.dim }}>
                      <b style={{ color: C.ink }}>{name}</b>: {g.mean.toFixed(2)} <span style={{ color: C.faint }}>[{g.lo.toFixed(2)}, {g.hi.toFixed(2)}] n={g.n}</span>
                    </span>
                  ))}
                </div>
                {m.comparisons.length > 0 && (
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead><tr>{["Confronto", "Δ", "p (MWU)", "Cliff δ", "Effetto"].map((h, i) => <th key={i} style={{ ...th, textAlign: i === 0 ? "left" : "right" }}>{h}</th>)}</tr></thead>
                    <tbody>
                      {m.comparisons.map((c, i) => (
                        <tr key={i} style={{ borderTop: `1px solid ${C.line}` }}>
                          <td style={{ ...td, color: C.ink }}>{c.a} vs {c.b}</td>
                          <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono" }}>{c.mean_diff.toFixed(2)}</td>
                          <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono", color: c.significant ? C.green : C.dim, fontWeight: c.significant ? 700 : 400 }}>{c.p_value == null ? "—" : c.p_value.toFixed(4)}{c.significant ? " *" : ""}</td>
                          <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono" }}>{c.cliffs_delta ?? "—"}</td>
                          <td style={{ ...td, textAlign: "right", color: C.dim }}>{c.effect_magnitude ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ))}
          </div>
        )}
      </ChartCard>

      <ChartCard title="Analisi appaiata — batteria within-subjects (Δ con − senza copilota)" style={{ marginTop: 16 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <Button kind="primary" onClick={loadBatteryAnalysis}>Analizza batterie</Button>
          {batteryAnalysis && <a href={researchApi.experimentBatteryCsvUrl()} className="mono" style={{ fontSize: 12.5, color: C.blue, textDecoration: "none", border: `1px solid ${C.celeste}`, borderRadius: 5, padding: "7px 12px", background: C.tint }}>⬇ Esporta CSV (per partita)</a>}
          <span style={{ fontSize: 12, color: C.faint }}>Wilcoxon signed-rank sui soggetti che hanno completato entrambe le condizioni, per classe di lunghezza.</span>
        </div>
        {batteryAnalysis && (
          <div style={{ marginTop: 14 }}>
            <div className="mono" style={{ fontSize: 12, color: C.dim, marginBottom: 10 }}>
              {batteryAnalysis.n_participants} partecipanti · {batteryAnalysis.n_games} partite · lunghezze: {batteryAnalysis.length_classes.join(", ") || "—"}
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
              <thead>
                <tr>{["Esito", "Lunghezza", "n coppie", "Δ media", "p (Wilcoxon)", "effect r"].map((h, i) => <th key={i} style={{ ...th, textAlign: i < 2 ? "left" : "right" }}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {batteryAnalysis.metrics.flatMap((m) =>
                  Object.entries(m.by_length).map(([lc, d]) => (
                    <tr key={m.key + lc} style={{ borderTop: `1px solid ${C.line}` }}>
                      <td style={{ ...td, color: C.ink }}>{m.label}</td>
                      <td style={td}>{lc}</td>
                      <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono" }}>{d?.n_pairs ?? 0}</td>
                      <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono" }}>{d?.mean_diff != null ? d.mean_diff.toFixed(3) : "—"}</td>
                      <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono", color: d?.significant ? C.green : C.dim, fontWeight: d?.significant ? 700 : 400 }}>
                        {d?.p_value != null ? d.p_value.toFixed(4) : "—"}{d?.significant ? " *" : ""}
                      </td>
                      <td style={{ ...td, textAlign: "right", fontFamily: "IBM Plex Mono" }}>{d?.effect_r ?? "—"}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
            <div className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 8 }}>
              Δ &gt; 0 = il copilota migliora l'esito. * significativo a α=0.05. Le condizioni computazionali si aggiungono via Campagne sugli stessi scenari.
            </div>
          </div>
        )}
      </ChartCard>
    </div>
  );
}

const input: React.CSSProperties = { background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 4, padding: "8px 10px", fontSize: 13 };
const th: React.CSSProperties = { textAlign: "left", padding: "6px 8px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "7px 8px", color: C.dim };
