import { useState } from "react";
import { Button, C } from "./ui";

// Tutorial di onboarding (fuori dai 30 min di gioco): insegna gli elementi
// essenziali prima della partita di pratica. Carosello di pochi passi.
const STEPS: { icon: string; title: string; body: string }[] = [
  {
    icon: "🎯",
    title: "Il tuo compito",
    body: "Gestisci un piccolo parco di infrastrutture per alcuni anni. Ogni anno decidi quali interventi fare per mantenerle sicure ed efficienti, con un budget limitato.",
  },
  {
    icon: "📊",
    title: "La sicurezza: l'indice β",
    body: "Ogni asset ha un indice di affidabilità β. Il valore obiettivo è 3.8: sopra è sicuro (verde), sotto è a rischio (ambra/rosso). Tieni gli asset sopra il target.",
  },
  {
    icon: "⚖️",
    title: "Degrado e budget",
    body: "Gli asset si degradano nel tempo. Il budget non basta per riparare tutto: dovrai scegliere le priorità (l'asset più critico? il più economico da sistemare?). È qui la decisione.",
  },
  {
    icon: "🔧",
    title: "Come si decide",
    body: "Su ogni card scegli un intervento dal menu (ispezione, ripristino, sostituzione…). Poi premi «Applica e avanza l'anno»: passa un anno e vedi gli effetti.",
  },
  {
    icon: "🛰️",
    title: "Il gemello digitale",
    body: "Clicca un asset (sulla mappa o sulla card, pulsante «Gemello») per aprirne il dettaglio: β, proiezione nel tempo, condizione dei singoli componenti.",
  },
  {
    icon: "🤝",
    title: "Il copilota AI (in alcune partite)",
    body: "In alcune partite avrai un copilota AI con cui discutere: conosce lo stato tecnico e propone piani motivati, ma non decide da solo — la scelta resta sempre tua.",
  },
  {
    icon: "▶️",
    title: "Si comincia con una prova",
    body: "La prima partita è di PRATICA e non viene conteggiata: serve solo a prendere confidenza con l'interfaccia. Poi iniziano le partite vere. Buon lavoro!",
  },
];

export default function Tutorial({ onDone }: { onDone: () => void }) {
  const [i, setI] = useState(0);
  const step = STEPS[i];
  const last = i === STEPS.length - 1;
  return (
    <div style={{ maxWidth: 620, margin: "48px auto" }}>
      <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 12, padding: "30px 34px", boxShadow: "0 2px 10px rgba(20,60,110,0.08)" }}>
        <div style={{ fontSize: 44 }}>{step.icon}</div>
        <div className="serif" style={{ fontSize: 23, color: C.blueDeep, fontWeight: 600, marginTop: 10 }}>{step.title}</div>
        <div style={{ fontSize: 15, color: C.dim, lineHeight: 1.65, marginTop: 12 }}>{step.body}</div>

        <div style={{ display: "flex", gap: 6, marginTop: 26 }}>
          {STEPS.map((_, k) => (
            <span key={k} style={{ flex: 1, height: 4, borderRadius: 2, background: k <= i ? C.blue : C.line }} />
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 22 }}>
          <button onClick={() => setI((x) => Math.max(0, x - 1))} disabled={i === 0}
            className="mono" style={{ border: `1px solid ${C.line2}`, background: C.panel2, color: i === 0 ? C.faint : C.dim, borderRadius: 6, padding: "9px 14px", fontSize: 13, cursor: i === 0 ? "default" : "pointer" }}>
            ← Indietro
          </button>
          <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>{i + 1} / {STEPS.length}</span>
          <span style={{ flex: 1 }} />
          {last ? (
            <Button kind="primary" onClick={onDone}>Inizia la pratica →</Button>
          ) : (
            <Button kind="primary" onClick={() => setI((x) => x + 1)}>Avanti →</Button>
          )}
          {!last && <button onClick={onDone} className="mono" style={{ border: "none", background: "transparent", color: C.faint, fontSize: 12, cursor: "pointer" }}>salta</button>}
        </div>
      </div>
    </div>
  );
}
