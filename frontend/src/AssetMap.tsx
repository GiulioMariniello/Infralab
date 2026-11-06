import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { AssetView } from "./api";
import { betaColor, C } from "./ui";
import { eur, fmt, pct } from "./useSession";

// Mappa geografica degli asset (Leaflet, marker vettoriali colorati per β).
// Niente immagini-marker esterne (circleMarker), tile OSM: se la rete manca i
// marker restano visibili su sfondo neutro. Riusata da rete e plancia giocatore.
export default function AssetMap({
  assets,
  height = 420,
  selectedId = null,
  onSelect,
  icon = "",
}: {
  assets: AssetView[];
  height?: number;
  selectedId?: number | null;
  onSelect?: (id: number) => void;
  icon?: string;
}) {
  const elRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const markers = useRef<Map<number, L.CircleMarker>>(new Map());

  const geo = assets.filter((a) => a.lat != null && a.lon != null);

  useEffect(() => {
    if (!elRef.current || mapRef.current) return;
    const map = L.map(elRef.current, { zoomControl: true, attributionControl: true, scrollWheelZoom: false });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap",
    }).addTo(map);
    map.setView([42.0, 13.0], 5);
    layerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Ridisegna i marker quando gli asset cambiano.
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer) return;
    layer.clearLayers();
    markers.current.clear();
    if (!geo.length) return;
    const pts: [number, number][] = [];
    for (const a of geo) {
      const lat = a.lat as number;
      const lon = a.lon as number;
      pts.push([lat, lon]);
      const r = 7 + 10 * Math.min(1, a.network_importance ?? 0);
      const m = L.circleMarker([lat, lon], {
        radius: r,
        color: "#ffffff",
        weight: 2,
        fillColor: betaColor(a.reliability_index),
        fillOpacity: 0.9,
      });
      m.bindTooltip(a.name, { direction: "top", offset: [0, -4] });
      m.bindPopup(popupHtml(a, icon), { closeButton: false });
      if (onSelect) m.on("click", () => onSelect(a.id));
      m.addTo(layer);
      markers.current.set(a.id, m);
    }
    map.fitBounds(L.latLngBounds(pts).pad(0.25), { maxZoom: 13 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(geo.map((a) => [a.id, a.lat, a.lon, a.reliability_index, a.network_importance])), icon]);

  // Evidenzia il marker selezionato.
  useEffect(() => {
    markers.current.forEach((m, id) => {
      m.setStyle({ weight: id === selectedId ? 4 : 2, color: id === selectedId ? C.blueDeep : "#ffffff" });
      if (id === selectedId) m.openPopup();
    });
  }, [selectedId]);

  if (!geo.length) {
    return (
      <div style={{ height, display: "flex", alignItems: "center", justifyContent: "center", background: C.tint, borderRadius: 8, color: C.faint }} className="mono">
        Nessuna coordinata: aggiungi lat/lon agli asset (costruttore o Excel) per vederli sulla mappa.
      </div>
    );
  }
  return <div ref={elRef} style={{ height, width: "100%", borderRadius: 8, overflow: "hidden", border: `1px solid ${C.line2}` }} />;
}

function popupHtml(a: AssetView, icon: string): string {
  const beta = fmt(a.reliability_index, 2);
  const color = betaColor(a.reliability_index);
  const seismic = a.seismic?.enabled && a.seismic.failure_probability != null
    ? `<div style="color:#6b7a86">p_f sismico ${a.seismic.failure_probability.toExponential(1)}</div>`
    : "";
  return `
    <div style="font-family:IBM Plex Mono,monospace;min-width:150px">
      <div style="font-weight:600;color:#0b2740;margin-bottom:4px">${icon ? icon + " " : ""}${escapeHtml(a.name)}</div>
      <div>β <b style="color:${color}">${beta}</b> · cond ${pct(a.min_condition)}</div>
      <div style="color:#6b7a86">${escapeHtml(a.status)} · ${escapeHtml(a.manager)}</div>
      <div style="color:#6b7a86">NAV ${eur(a.net_asset_value)}</div>
      ${seismic}
    </div>`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] as string));
}
