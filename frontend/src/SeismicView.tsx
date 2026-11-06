import { useEffect, useMemo, useState } from "react";
import { api, type SeismicState } from "./api";
import { Button, C, ChartCard, Label } from "./ui";
import { sci, type SessionController } from "./useSession";

// Scala colore PGA: da celeste chiaro (bassa) ad ambra a rosso (alta).
function pgaColor(pga: number, far: number, peak: number): string {
  const t = Math.max(0, Math.min(1, (pga - far) / Math.max(1e-6, peak - far)));
  const stops = [
    [232, 244, 252], // celeste chiarissimo
    [120, 196, 232], // celeste
    [224, 144, 47], // ambra
    [209, 84, 74], // rosso
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export default function SeismicView({ ctrl }: { ctrl: SessionController }) {
  const [sz, setSz] = useState<SeismicState | null>(null);
  const sid = ctrl.state?.session_id;
  const quarter = ctrl.state?.kpis.quarter;

  useEffect(() => {
    if (!sid) return;
    api.seismic(sid).then(setSz).catch(() => setSz(null));
  }, [sid, quarter, ctrl.state?.assets]);

  const map = useMemo(() => {
    if (!sz) return null;
    const W = 720, H = 460, n = sz.grid.length;
    const { lat0, lon0, lat1, lon1 } = sz.bounds;
    const px = (lon: number) => ((lon - lon0) / (lon1 - lon0 || 1)) * W;
    const py = (lat: number) => (1 - (lat - lat0) / (lat1 - lat0 || 1)) * H;
    return { W, H, n, px, py };
  }, [sz]);

  if (!ctrl.state) return <div style={{ textAlign: "center", padding: 70, color: C.faint }} className="mono">Avvia una sessione per la sismica.</div>;

  const enabled = sz?.enabled ?? false;

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
        <Label>Sismica</Label>
        <span
          className="mono"
          style={{ fontSize: 12, padding: "5px 10px", borderRadius: 4, background: enabled ? "#eaf6ee" : C.panel2, color: enabled ? C.green : C.faint, border: `1px solid ${enabled ? C.green : C.line2}` }}
        >
          {enabled ? "ATTIVA · incide su β" : "DISATTIVATA"}
        </span>
        <Button kind={enabled ? "ghost" : "primary"} onClick={() => ctrl.toggleSeismic(!enabled)} disabled={ctrl.busy}>
          {enabled ? "Disattiva sismica" : "Attiva sismica"}
        </Button>
        {sz && (
          <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>
            centro pericolosità {sz.hazard_center.lat.toFixed(2)}, {sz.hazard_center.lon.toFixed(2)} · PGA {sz.far_pga}–{sz.peak_pga} g · decad. {sz.decay_km} km
          </span>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 16 }}>
        <ChartCard title="Mappa di pericolosità sismica — PGA di sito (475 anni)">
          {sz && map ? (
            <svg viewBox={`0 0 ${map.W} ${map.H}`} style={{ width: "100%", height: "auto", borderRadius: 6 }}>
              {sz.grid.map((row, i) =>
                row.map((pga, j) => (
                  <rect
                    key={`${i}-${j}`}
                    x={(j / map.n) * map.W}
                    y={((map.n - 1 - i) / map.n) * map.H}
                    width={map.W / map.n + 0.6}
                    height={map.H / map.n + 0.6}
                    fill={pgaColor(pga, sz.far_pga, sz.peak_pga)}
                  />
                )),
              )}
              {/* centro pericolosità */}
              <g>
                <circle cx={map.px(sz.hazard_center.lon)} cy={map.py(sz.hazard_center.lat)} r={7} fill="none" stroke="#7a1f18" strokeWidth={2} />
                <text x={map.px(sz.hazard_center.lon)} y={map.py(sz.hazard_center.lat) - 11} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="10" fill="#7a1f18">
                  zona sismogenetica
                </text>
              </g>
              {/* asset */}
              {sz.assets.filter((a) => a.lat != null).map((a) => (
                <g key={a.id}>
                  <circle cx={map.px(a.lon!)} cy={map.py(a.lat!)} r={6} fill="#fff" stroke={C.blueDeep} strokeWidth={2} />
                  <text x={map.px(a.lon!)} y={map.py(a.lat!) - 9} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="9.5" fill={C.ink}>
                    {a.name.length > 16 ? a.name.slice(0, 15) + "…" : a.name}
                  </text>
                </g>
              ))}
            </svg>
          ) : (
            <div style={{ padding: 30, color: C.faint }}>Nessun dato di pericolosità.</div>
          )}
          {sz && <PgaLegend far={sz.far_pga} peak={sz.peak_pga} />}
        </ChartCard>

        <ChartCard title="Fragilità sismica per asset">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr>{["Asset", "PGA sito", "Dist. km", "p_f sismico"].map((h) => <th key={h} style={th}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {(sz?.assets ?? []).map((a) => (
                <tr key={a.id} style={{ borderTop: `1px solid ${C.line}` }}>
                  <td style={{ ...td, color: C.ink, fontWeight: 500, textAlign: "left" }}>{a.name}</td>
                  <td style={tdN}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, justifyContent: "flex-end" }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: sz ? pgaColor(a.site_pga ?? 0, sz.far_pga, sz.peak_pga) : "#ccc" }} />
                      {a.site_pga != null ? `${a.site_pga.toFixed(3)} g` : "—"}
                    </span>
                  </td>
                  <td style={tdN}>{a.distance_to_hazard_km != null ? a.distance_to_hazard_km.toFixed(0) : "—"}</td>
                  <td style={{ ...tdN, color: (a.failure_probability ?? 0) > 0.01 ? C.red : C.dim }}>{sci(a.failure_probability)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12, fontSize: 12, color: C.faint, lineHeight: 1.5 }}>
            La PGA di sito dipende dalla <b>distanza dalla zona sismogenetica</b>: più vicino = pericolosità maggiore.
            Con la sismica attiva, il p_f sismico si somma (sistema serie) al p_f strutturale e riduce β.
          </div>
        </ChartCard>
      </div>
    </div>
  );
}

function PgaLegend({ far, peak }: { far: number; peak: number }) {
  const stops = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10 }}>
      <span className="mono" style={{ fontSize: 10.5, color: C.faint }}>{far.toFixed(2)} g</span>
      <div style={{ display: "flex", flex: 1, maxWidth: 260, height: 10, borderRadius: 3, overflow: "hidden" }}>
        {stops.map((t) => (
          <div key={t} style={{ flex: 1, background: pgaColor(far + t * (peak - far), far, peak) }} />
        ))}
      </div>
      <span className="mono" style={{ fontSize: 10.5, color: C.faint }}>{peak.toFixed(2)} g</span>
    </div>
  );
}

const th: React.CSSProperties = { textAlign: "left", padding: "6px 8px", fontSize: 10, letterSpacing: ".1em", textTransform: "uppercase", color: C.faint, fontWeight: 500 };
const td: React.CSSProperties = { padding: "7px 8px", color: C.dim, textAlign: "right" };
const tdN: React.CSSProperties = { ...td, fontFamily: "IBM Plex Mono", fontVariantNumeric: "tabular-nums" };
