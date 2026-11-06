import { useEffect, useMemo, useState } from "react";
import { api, type AssetType, type AssetView } from "./api";
import AssetMap from "./AssetMap";
import CopilotPanel from "./CopilotPanel";
import DigitalTwin from "./DigitalTwin";
import Questionnaire from "./Questionnaire";
import Tutorial from "./Tutorial";
import { betaColor, Button, C } from "./ui";
import { eur, fmt, pct, type SessionController } from "./useSession";

// Plancia del giocatore: una schermata sola, semplice e ricca — mappa,
// stato in parole, card asset con azioni inline, gemello digitale a scomparsa.
// Copilota disponibile SOLO in condizione Uomo+LLM; consenso richiesto prima.
export default function PlayerView({ ctrl, consentGiven, onConsent, ptoken }: {
  ctrl: SessionController; consentGiven: boolean; onConsent: () => void; ptoken: string | null;
}) {
  const { state, busy } = ctrl;
  const [plan, setPlan] = useState<Record<string, string>>({});
  const [rationale, setRationale] = useState("");
  const [twinId, setTwinId] = useState<number | null>(null);
  const [icon, setIcon] = useState("");
  const [showQ, setShowQ] = useState(false);
  const [consenting, setConsenting] = useState(false);
  const [tutorialSeen, setTutorialSeen] = useState(false);

  useEffect(() => {
    if (!state) return;
    api.assetTypes()
      .then((ts: AssetType[]) => setIcon(ts.find((t) => t.name === state.asset_type)?.ui_meta?.icon ?? ""))
      .catch(() => setIcon(""));
  }, [state?.asset_type]);

  const planCost = useMemo(() => {
    if (!state) return 0;
    let tot = 0;
    for (const a of state.assets) {
      const chosen = plan[String(a.id)];
      if (chosen) tot += a.actions.find((x) => x.action === chosen)?.cost ?? 0;
    }
    return tot;
  }, [plan, state]);

  if (!state) return <div className="mono" style={{ textAlign: "center", padding: 80, color: C.faint }}>Caricamento della tua rete…</div>;

  // Gate consenso informato prima di iniziare a giocare.
  if (!consentGiven) {
    return (
      <div style={{ maxWidth: 640, margin: "36px auto" }}>
        <div className="serif" style={{ fontSize: 24, color: C.blueDeep, fontWeight: 600 }}>Consenso informato</div>
        <div style={{ marginTop: 14, color: C.dim, fontSize: 14, lineHeight: 1.6 }}>
          Partecipi a uno studio di ricerca sulla gestione dell'affidabilità delle infrastrutture.
          Giocherai una simulazione di gestione di un parco di asset e, al termine, compilerai un breve
          questionario. Le tue <b>decisioni e risposte sono raccolte in forma anonima</b> e usate solo a
          fini di ricerca scientifica. La partecipazione è volontaria e puoi interromperla in qualsiasi momento.
          {state.condition && <> Condizione assegnata: <b className="mono" style={{ color: C.blueDeep }}>{state.condition}</b>.</>}
        </div>
        <div style={{ marginTop: 22, display: "flex", gap: 12, alignItems: "center" }}>
          <Button kind="primary" disabled={consenting} onClick={async () => {
            setConsenting(true);
            try { await api.consent(state.session_id); onConsent(); } finally { setConsenting(false); }
          }}>Acconsento e inizio →</Button>
        </div>
      </div>
    );
  }

  const battery = state.battery;
  const currentGame = battery?.games[battery.current_index];
  const isPractice = currentGame?.length_class === "practice";

  async function apply() {
    await ctrl.act(plan, rationale.trim() || undefined);
    setPlan({});
    setRationale("");
  }

  // --- Flusso a batteria (più partite con/senza copilota) ---
  // Tutorial una sola volta, prima della partita di pratica.
  if (battery && !tutorialSeen && battery.current_index === 0 && !state.finished) {
    return <Tutorial onDone={() => setTutorialSeen(true)} />;
  }
  if (battery?.complete) {
    // Fine batteria: questionario finale (profilo/fiducia/strategia).
    return <Questionnaire sessionId={state.session_id} title="Questionario finale"
      intro="Ultimo passo: qualche domanda sul tuo profilo, sulla strategia e — se hai usato il copilota — sulla fiducia. ~2 minuti."
      onDone={() => setShowQ(true)} />;
  }
  if (battery && state.finished && isPractice) {
    // La pratica non ha mini-TLX: transizione esplicita alle partite vere.
    return (
      <div style={{ maxWidth: 560, margin: "60px auto", textAlign: "center" }}>
        <div style={{ fontSize: 40 }}>✅</div>
        <div className="serif" style={{ fontSize: 23, color: C.blueDeep, fontWeight: 600, marginTop: 8 }}>Pratica conclusa</div>
        <div style={{ color: C.dim, fontSize: 15, marginTop: 12, lineHeight: 1.6 }}>
          Hai preso confidenza con l'interfaccia. Ora iniziano le <b>partite vere</b> (queste vengono registrate).
        </div>
        <div style={{ marginTop: 24 }}>
          <Button kind="primary" onClick={() => { if (ptoken) ctrl.advanceBattery(ptoken); }}>Inizia le partite vere →</Button>
        </div>
      </div>
    );
  }
  if (battery && state.finished) {
    // Fine partita vera: mini-TLX, poi avanza alla successiva.
    const gameNo = battery.current_index; // le partite vere partono dopo la pratica (indice 1)
    return <Questionnaire sessionId={state.session_id} only={["nasa_tlx"]}
      title={`Partita ${gameNo} di ${battery.total - 1} — carico di lavoro`}
      intro="Valuta rapidamente il carico di lavoro di questa partita (~30 secondi), poi si passa alla prossima."
      onDone={() => { if (ptoken) ctrl.advanceBattery(ptoken); }} />;
  }

  const k = state.kpis;
  const betaLow = k.min_reliability_index != null && k.min_reliability_index < 3.8;
  const twinAsset = state.assets.find((a) => a.id === twinId) ?? null;
  const overBudget = planCost > k.budget;
  const planCount = Object.values(plan).filter(Boolean).length;

  // Questionario finale per i link singoli (non batteria).
  if (showQ) {
    return <Questionnaire sessionId={state.session_id} onDone={() => setShowQ(false)} />;
  }

  const annual = (state.cadence_quarters ?? 1) >= 4;
  const yearNo = Math.ceil((k.quarter - 1) / (state.cadence_quarters ?? 1)) + (state.finished ? 0 : 1);
  const totalYears = Math.round((state.horizon_quarters ?? 0) / (state.cadence_quarters ?? 1));

  return (
    <div style={{ marginTop: 4 }}>
      {/* Progresso batteria */}
      {battery && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          {battery.games.map((g) => (
            <span key={g.index} title={`${g.length_class} · ${g.condition}`}
              style={{ flex: 1, height: 6, borderRadius: 3,
                background: g.index < battery.current_index ? C.green : g.index === battery.current_index ? C.blue : C.line }} />
          ))}
          <span className="mono" style={{ fontSize: 11, color: C.faint, whiteSpace: "nowrap" }}>
            {isPractice ? "prova (non conta)" : `partita ${battery.current_index}/${battery.total - 1}`} · {state.copilot_enabled ? "con copilota" : "senza copilota"}
          </span>
        </div>
      )}

      {/* Stato in una frase */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
        <span style={{ fontSize: 16, color: C.ink }}>
          {state.finished ? (
            <b style={{ color: k.active ? C.blueDeep : C.red }}>{k.active ? "Orizzonte raggiunto — partita conclusa." : "Partita terminata (fallimento)."}</b>
          ) : betaLow ? (
            <>C'è un asset <b style={{ color: C.amber }}>sotto la soglia di sicurezza</b> (β target 3.8): valuta un intervento.</>
          ) : (
            <>La tua rete è <b style={{ color: C.green }}>in salute</b>. Pianifica o avanza {annual ? "l'anno" : "il trimestre"}.</>
          )}
        </span>
        <span className="mono" style={{ fontSize: 12, color: C.faint }}>
          {annual ? `Anno ${Math.min(yearNo, totalYears)}/${totalYears}` : `Trimestre T${k.quarter}`}
        </span>
      </div>

      {/* KPI grandi e semplici */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14 }}>
        <BigKpi label="Sicurezza minima (β)" value={fmt(k.min_reliability_index, 2)} color={betaColor(k.min_reliability_index)} hint="target 3.80" />
        <BigKpi label="Salute della rete" value={pct(k.health_index)} color={C.green} hint="media pesata" />
        <BigKpi label="Budget disponibile" value={eur(k.budget)} color={C.blueDeep} hint={`valore rete ${eur(k.net_asset_value)}`} />
        <BigKpi label="Asset gestiti" value={String(state.assets.length)} color={C.ink} hint={`${k.active ? "operativa" : "terminata"}`} />
      </div>

      {/* Mappa */}
      <div style={{ marginTop: 18 }}>
        <SectionTitle>La tua rete sul territorio</SectionTitle>
        <AssetMap assets={state.assets} height={380} icon={icon} selectedId={twinId} onSelect={setTwinId} />
        <div className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 6 }}>
          Clicca un asset sulla mappa o su una card per aprirne il <b style={{ color: C.dim }}>gemello digitale</b>.
        </div>
      </div>

      {/* Copilota AI — solo condizione Uomo+LLM */}
      {state.copilot_enabled && (
        <div style={{ marginTop: 20 }}>
          <SectionTitle>Copilota AI — decidete insieme</SectionTitle>
          <CopilotPanel
            sid={state.session_id}
            agents={state.agents.filter((a) => a.is_human).map((a) => a.name)}
            onApply={(p) => {
              const valid: Record<string, string> = {};
              for (const [aid, action] of Object.entries(p)) {
                const asset = state.assets.find((x) => String(x.id) === String(aid));
                if (asset && asset.actions.some((x) => x.action === action && x.feasible)) valid[String(aid)] = action;
              }
              setPlan((prev) => ({ ...prev, ...valid }));
            }}
          />
        </div>
      )}

      {/* Card asset con azione inline */}
      <div style={{ marginTop: 20 }}>
        <SectionTitle>I tuoi asset — scegli gli interventi</SectionTitle>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
          {state.assets.map((a) => (
            <AssetCard
              key={a.id}
              asset={a}
              icon={icon}
              value={plan[String(a.id)] ?? ""}
              onAction={(v) => setPlan((p) => ({ ...p, [String(a.id)]: v }))}
              onTwin={() => setTwinId(a.id)}
            />
          ))}
        </div>
      </div>

      {/* Barra azione sticky */}
      <div style={sticky}>
        <input
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          placeholder="perché questa scelta? (facoltativo, per la ricerca)"
          style={{ flex: 1, minWidth: 180, background: C.bg, color: C.ink, border: `1px solid ${C.line2}`, borderRadius: 5, padding: "8px 12px", fontSize: 12.5 }}
        />
        <span className="mono" style={{ fontSize: 12.5, color: C.dim, whiteSpace: "nowrap" }}>
          {planCount ? `${planCount} intervent${planCount === 1 ? "o" : "i"}` : "nessun intervento"}
          {" · "}<b style={{ color: overBudget ? C.red : C.blueDeep }}>{eur(planCost)}</b>
        </span>
        {overBudget && <span className="mono" style={{ fontSize: 12, color: C.red }}>oltre il budget</span>}
        <Button kind="primary" onClick={apply} disabled={busy || overBudget}>
          Applica e avanza {annual ? "l'anno" : ""}→
        </Button>
        {!battery && (
          <button onClick={() => setShowQ(true)} className="mono" style={{ border: `1px solid ${C.line2}`, background: C.panel2, color: C.blueDeep, borderRadius: 5, padding: "8px 12px", fontSize: 11.5, cursor: "pointer", whiteSpace: "nowrap" }}>
            Concludi
          </button>
        )}
      </div>

      {twinAsset && (
        <DigitalTwin sessionId={state.session_id} asset={twinAsset} quarter={k.quarter} icon={icon} onClose={() => setTwinId(null)} />
      )}
    </div>
  );
}

function AssetCard({ asset, icon, value, onAction, onTwin }: {
  asset: AssetView; icon: string; value: string; onAction: (v: string) => void; onTwin: () => void;
}) {
  const feasible = asset.actions.filter((x) => x.feasible);
  const cond = asset.min_condition;
  const condCol = cond >= 0.6 ? C.green : cond >= 0.35 ? C.amber : C.red;
  return (
    <div style={{ background: C.panel, border: `1px solid ${value ? C.celeste : C.line2}`, borderRadius: 10, padding: "14px 16px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
        <div style={{ cursor: "pointer" }} onClick={onTwin}>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>{icon ? icon + " " : ""}{asset.name}</div>
          <div className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 2 }}>{asset.status}{asset.busy ? " · in intervento" : ""}</div>
        </div>
        <span className="mono" style={{ fontSize: 13, fontWeight: 700, color: betaColor(asset.reliability_index), whiteSpace: "nowrap" }}>
          β {fmt(asset.reliability_index, 2)}
        </span>
      </div>

      <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 8 }}>
        <span className="mono" style={{ fontSize: 10.5, color: C.faint, width: 58 }}>condizione</span>
        <span style={{ flex: 1, height: 8, background: C.line, borderRadius: 4, overflow: "hidden" }}>
          <span style={{ display: "block", width: `${Math.max(3, cond * 100)}%`, height: "100%", background: condCol }} />
        </span>
        <span className="mono" style={{ fontSize: 11.5, color: C.dim, width: 42, textAlign: "right" }}>{pct(cond)}</span>
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
        {asset.busy ? (
          <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>intervento in corso…</span>
        ) : (
          <select
            value={value}
            onChange={(e) => onAction(e.target.value)}
            className="mono"
            style={{ flex: 1, background: value ? C.tint : C.bg, color: value ? C.blueDeep : C.dim, border: `1px solid ${value ? C.celeste : C.line2}`, borderRadius: 5, padding: "8px 10px", fontSize: 12 }}
          >
            <option value="">— nessun intervento —</option>
            {feasible.map((x) => (
              <option key={x.action} value={x.action}>{x.action} · {eur(x.cost)}</option>
            ))}
          </select>
        )}
        <button onClick={onTwin} className="mono" style={{ border: `1px solid ${C.line2}`, background: C.panel2, color: C.blueDeep, borderRadius: 5, padding: "8px 10px", fontSize: 11.5, cursor: "pointer", whiteSpace: "nowrap" }}>
          Gemello →
        </button>
      </div>
    </div>
  );
}

function BigKpi({ label, value, color, hint }: { label: string; value: string; color: string; hint: string }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 10, padding: "16px 18px", position: "relative", overflow: "hidden", boxShadow: "0 1px 3px rgba(20,60,110,0.06)" }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: 4, height: "100%", background: color }} />
      <div className="mono" style={{ fontSize: 10.5, letterSpacing: ".08em", textTransform: "uppercase", color: C.faint }}>{label}</div>
      <div className="mono" style={{ fontSize: 30, fontWeight: 600, color, marginTop: 8, letterSpacing: "-.01em" }}>{value}</div>
      <div className="mono" style={{ fontSize: 11, color: C.faint, marginTop: 5 }}>{hint}</div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="mono" style={{ fontSize: 11, letterSpacing: ".12em", textTransform: "uppercase", color: C.blueDeep, marginBottom: 10 }}>{children}</div>;
}

const sticky: React.CSSProperties = {
  position: "sticky", bottom: 0, marginTop: 20, display: "flex", alignItems: "center", gap: 12,
  background: "rgba(245,248,252,0.94)", backdropFilter: "blur(8px)", borderTop: `1px solid ${C.line2}`,
  padding: "14px 18px", borderRadius: "10px 10px 0 0",
};
