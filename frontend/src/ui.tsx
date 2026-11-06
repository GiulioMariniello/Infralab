// Primitivi di UI e marchio del design system InfraLab.
import type { CSSProperties, ReactNode } from "react";

export const C = {
  bg: "#f5f8fc",
  panel: "#ffffff",
  panel2: "#eef4fb",
  ink: "#132638",
  dim: "#4a5c6e",
  faint: "#8496a8",
  line: "rgba(20,60,110,0.10)",
  line2: "rgba(20,60,110,0.16)",
  blue: "#1f6fb2",
  blueDeep: "#0b4f86",
  celeste: "#4aa3df",
  celesteSoft: "#7cc4e8",
  tint: "#eaf3fb",
  green: "#2e9e6b",
  amber: "#e0902f",
  red: "#d1544a",
};

export function Mark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 34 34" fill="none">
      <rect x="1" y="1" width="32" height="32" rx="3" stroke={C.blue} strokeWidth="1.3" />
      <path d="M6 24 L14 10 L20 20 L24 14 L28 24" stroke={C.blueDeep} strokeWidth="1.4" fill="none" strokeLinejoin="round" />
      <line x1="6" y1="24" x2="28" y2="24" stroke={C.celeste} strokeWidth="1.2" />
      <circle cx="14" cy="10" r="1.7" fill={C.celeste} />
    </svg>
  );
}

export function Wordmark({ size = 21 }: { size?: number }) {
  return (
    <span className="serif" style={{ fontWeight: 600, fontSize: size, letterSpacing: ".02em", color: C.blueDeep }}>
      Infra<span style={{ color: C.blue }}>Lab</span>
    </span>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div
      className="mono"
      style={{
        fontSize: 12,
        letterSpacing: ".28em",
        textTransform: "uppercase",
        color: C.blue,
        display: "flex",
        alignItems: "center",
        gap: 14,
      }}
    >
      <span style={{ width: 34, height: 1, background: C.celeste }} />
      {children}
    </div>
  );
}

export function Card({ children, style, hover }: { children: ReactNode; style?: CSSProperties; hover?: boolean }) {
  return (
    <div
      className={hover ? "il-card-hover" : undefined}
      style={{
        background: C.panel,
        border: `1px solid ${C.line2}`,
        borderRadius: 8,
        boxShadow: "0 1px 3px rgba(20,60,110,0.06)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Button({
  children,
  onClick,
  kind = "ghost",
  disabled,
  style,
}: {
  children: ReactNode;
  onClick?: () => void;
  kind?: "primary" | "ghost";
  disabled?: boolean;
  style?: CSSProperties;
}) {
  const base: CSSProperties = {
    fontSize: 12.5,
    letterSpacing: ".08em",
    textTransform: "uppercase",
    fontWeight: kind === "primary" ? 600 : 500,
    padding: "10px 18px",
    borderRadius: 5,
    transition: "all .16s",
    opacity: disabled ? 0.5 : 1,
    pointerEvents: disabled ? "none" : "auto",
    ...style,
  };
  const skin: CSSProperties =
    kind === "primary"
      ? { background: C.blue, color: "#fff", border: `1px solid ${C.blue}`, boxShadow: "0 2px 6px rgba(31,111,178,.25)" }
      : { background: C.panel, color: C.blue, border: `1px solid ${C.line2}` };
  return (
    <button onClick={onClick} disabled={disabled} style={{ ...base, ...skin }}>
      {children}
    </button>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <div className="mono" style={{ fontSize: 10.5, letterSpacing: ".14em", textTransform: "uppercase", color: C.faint }}>
      {children}
    </div>
  );
}

export function ChartCard({ title, children, style }: { title: string; children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.line2}`, borderRadius: 8, padding: "16px 18px", boxShadow: "0 1px 3px rgba(20,60,110,0.06)", ...style }}>
      <div className="mono" style={{ fontSize: 11.5, letterSpacing: ".12em", textTransform: "uppercase", color: C.faint, marginBottom: 12 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

export const betaColor = (v: number | null | undefined) =>
  v == null ? C.faint : v >= 3.8 ? C.green : v >= 3.0 ? C.amber : C.red;

export const axisTick = { fontSize: 11, fill: C.faint, fontFamily: "IBM Plex Mono" } as const;
export const tooltipStyle: CSSProperties = { background: "#fff", border: `1px solid ${C.line2}`, borderRadius: 6, fontFamily: "IBM Plex Mono", fontSize: 12 };
