import { Button, C, Eyebrow, Mark, Wordmark } from "./ui";

export default function Landing({ onEnter }: { onEnter: () => void }) {
  return (
    <div style={{ maxWidth: 1140, margin: "0 auto", padding: "40px 40px 80px" }}>
      {/* barra strumento */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: `1px solid ${C.line2}`,
          paddingBottom: 16,
          marginBottom: 46,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Mark />
          <Wordmark />
        </div>
        <div className="mono" style={{ fontSize: 11, letterSpacing: ".14em", textTransform: "uppercase", color: C.faint }}>
          REL · <b style={{ color: C.blue }}>β / p_f</b> &nbsp;·&nbsp; ENGINE <b style={{ color: C.blue }}>v0.1.0</b> &nbsp;·&nbsp;{" "}
          <b style={{ color: C.blue }}>UNINA</b> · STRUCT-AI
        </div>
      </div>

      <Eyebrow>Laboratorio virtuale · Gestione di asset infrastrutturali</Eyebrow>
      <h1
        className="serif"
        style={{ fontSize: "clamp(38px,6vw,64px)", fontWeight: 600, lineHeight: 1.02, letterSpacing: "-.015em", color: C.blueDeep, margin: "20px 0 0" }}
      >
        Un banco di prova per
        <br />
        decisioni di sicurezza <em style={{ color: C.celeste, fontWeight: 500 }}>affidabili</em>.
      </h1>
      <p style={{ maxWidth: 660, marginTop: 20, fontSize: 16, lineHeight: 1.62, color: C.dim, fontWeight: 400 }}>
        InfraLab simula il ciclo di vita di reti infrastrutturali — ponti, muri di sostegno e oltre — accoppiando{" "}
        <b style={{ color: C.ink }}>affidabilità strutturale FOSM/bayesiana</b>, degrado stocastico e agenti decisionali in un
        unico banco di prova riproducibile per <b style={{ color: C.ink }}>nuovi paper e nuovi test</b>.
      </p>

      <div style={{ height: 1, background: C.line2, margin: "36px 0" }} />

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <FeatureCard n="01" title="Motore di affidabilità" body="Indici β / p_f per stato limite, aggiornamento bayesiano da prove di carico, sistema serie a livello di rete." />
        <FeatureCard n="02" title="Multi-asset" body="Registry estensibile: ponti e muri di sostegno oggi, pale eoliche / edifici / stazioni domani, stesso motore." />
        <FeatureCard n="03" title="Agenti & campagne" body="Euristica, GA, MCTS, RL, LLM confrontabili su politiche reliability-constrained; campagne riproducibili." />
      </div>

      <div style={{ marginTop: 40, display: "flex", alignItems: "center", gap: 18 }}>
        <Button kind="primary" onClick={onEnter} style={{ padding: "13px 26px", fontSize: 13 }}>
          Apri la console →
        </Button>
        <span className="mono" style={{ fontSize: 11.5, color: C.faint, letterSpacing: ".08em" }}>
          FASTAPI · REACT · IN TEMPO REALE
        </span>
      </div>
    </div>
  );
}

function FeatureCard({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div
      style={{
        flex: "1 1 300px",
        background: C.panel,
        border: `1px solid ${C.line2}`,
        borderRadius: 8,
        padding: "24px 24px 26px",
        boxShadow: "0 1px 3px rgba(20,60,110,0.06)",
      }}
    >
      <div className="mono" style={{ fontSize: 11.5, letterSpacing: ".12em", color: C.faint }}>
        {n} / MODULO
      </div>
      <div className="serif" style={{ fontSize: 21, fontWeight: 500, color: C.blueDeep, margin: "12px 0 10px" }}>
        {title}
      </div>
      <div style={{ fontSize: 14, lineHeight: 1.55, color: C.dim }}>{body}</div>
    </div>
  );
}
