# gis_bim_module.py
"""
Modulo GIS e BIM per BridgeManager2025.

- render_gis_map(): Mappa interattiva dei ponti con Plotly + OpenStreetMap (no API key)
- render_bim_3d():  Visualizzazione BIM 3D della struttura del ponte con Plotly Mesh3d
"""
from typing import Dict, List, Optional, Any

import numpy as np
import plotly.graph_objects as go


def hex_to_rgba(color: Optional[str], alpha: float = 1.0) -> Optional[str]:
    """Converte '#rrggbb' in 'rgba(r,g,b,a)' (decimale). Passa attraverso rgba/rgb/None."""
    if not color:
        return None
    if not color.startswith("#"):
        return color
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return color
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Palette colori condizione strutturale
# ---------------------------------------------------------------------------

CONDITION_COLORSCALE = [
    [0.00, "#c0392b"],
    [0.20, "#e74c3c"],
    [0.40, "#e67e22"],
    [0.60, "#f39c12"],
    [0.80, "#2ecc71"],
    [1.00, "#1abc9c"],
]


def condition_to_hex(condition: float) -> str:
    """Mappa un valore di condizione [0,1] in un colore HEX."""
    if condition > 0.80:
        return "#1abc9c"
    elif condition > 0.60:
        return "#2ecc71"
    elif condition > 0.40:
        return "#f39c12"
    elif condition > 0.20:
        return "#e67e22"
    else:
        return "#c0392b"


def condition_label(condition: float) -> str:
    if condition > 0.80:
        return "Buono"
    elif condition > 0.60:
        return "Discreto"
    elif condition > 0.40:
        return "Scarso"
    elif condition > 0.20:
        return "Critico"
    else:
        return "Collasso"


# ---------------------------------------------------------------------------
# GIS Map
# ---------------------------------------------------------------------------

def render_gis_map(
    bridges_dict: Dict,
    earthquake_events: Optional[List] = None,
) -> go.Figure:
    """
    Mappa GIS interattiva dei ponti con Plotly Scattermapbox.
    Usa OpenStreetMap (gratuito, nessun API key richiesto).

    Args:
        bridges_dict: dizionario {id: Bridge}
        earthquake_events: lista di EarthquakeEvent opzionale

    Returns:
        Figura Plotly pronta per st.plotly_chart()
    """
    lats, lons, cond_vals, sizes, hover_texts, names = [], [], [], [], [], []

    for bridge in bridges_dict.values():
        lat = getattr(bridge, "lat", None)
        lon = getattr(bridge, "lon", None)
        if lat is None or lon is None:
            continue

        cond = bridge.min_condition
        # Dimensione marker: proporzionale alla gravita' (peggiore = piu' grande)
        size = 14 + (1.0 - cond) * 20

        status_icon = (
            "🔧" if bridge.active_maintenance
            else "🚨" if cond < 0.20
            else "⚠️" if cond < 0.40
            else "✅"
        )
        maint_text = (
            f"In manutenzione fino a T{bridge.active_maintenance['ends_at_quarter']}"
            if bridge.active_maintenance else "Operativo"
        )
        asset_val = getattr(bridge, "current_asset_value", 0)
        hover = (
            f"<b>{status_icon} {bridge.name}</b><br>"
            f"ID: {bridge.id} | Manager: {bridge.manager}<br>"
            f"Condizione: <b>{cond:.1%}</b> ({condition_label(cond)})<br>"
            f"CdA: {bridge.cda}<br>"
            f"Stato: {maint_text}<br>"
            f"Valore Asset: €{asset_val:,.0f}<br>"
            f"Dimensioni: {bridge.length}m × {bridge.width}m | ADT: {bridge.adt:,}<br>"
            f"Coord: ({lat:.4f}°N, {lon:.4f}°E)"
        )

        lats.append(lat)
        lons.append(lon)
        cond_vals.append(cond)
        sizes.append(size)
        hover_texts.append(hover)
        names.append(bridge.name[:14] + ("…" if len(bridge.name) > 14 else ""))

    fig = go.Figure()

    if lats:
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=cond_vals,
                colorscale=CONDITION_COLORSCALE,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(text="Condizione", side="right", font=dict(color="#132638")),
                    tickformat=".0%",
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["0% Critico", "20%", "40%", "60%", "80%", "100% Ottimo"],
                    tickfont=dict(color="#132638"),
                    thickness=16,
                    len=0.65,
                    bgcolor="rgba(20,20,40,0.8)",
                    bordercolor="rgba(20,60,110,0.16)",
                ),
                opacity=0.88,
            ),
            text=names,
            textposition="top right",
            textfont=dict(size=9, color="#132638"),
            hovertext=hover_texts,
            hoverinfo="text",
            name="Ponti",
        ))

        center_lat = float(np.mean(lats))
        center_lon = float(np.mean(lons))
        zoom = 9
    else:
        center_lat, center_lon, zoom = 41.0, 14.5, 7

    # Epicentri terremoti
    if earthquake_events:
        eq_lats = [e.epicenter_lat for e in earthquake_events if hasattr(e, "epicenter_lat")]
        eq_lons = [e.epicenter_lon for e in earthquake_events if hasattr(e, "epicenter_lon")]
        eq_hover = [
            f"<b>🌍 {e.name}</b><br>"
            f"Magnitudo: M{e.magnitude}<br>"
            f"Epicentro: ({e.epicenter_lat:.3f}°N, {e.epicenter_lon:.3f}°E)<br>"
            f"Trimestre: {e.quarter}"
            for e in earthquake_events if hasattr(e, "epicenter_lat")
        ]
        eq_sizes = [8 + getattr(e, "magnitude", 5) * 3.5 for e in earthquake_events if hasattr(e, "epicenter_lat")]

        if eq_lats:
            fig.add_trace(go.Scattermapbox(
                lat=eq_lats,
                lon=eq_lons,
                mode="markers",
                marker=dict(
                    size=eq_sizes,
                    color="#e74c3c",
                    opacity=0.65,
                ),
                hovertext=eq_hover,
                hoverinfo="text",
                name="Epicentri Sismici",
            ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=580,
        title=dict(
            text="🗺️ Mappa GIS — Rete Infrastrutturale",
            font=dict(size=20, color="#132638", family="Arial"),
            x=0.5,
        ),
        paper_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="rgba(20,60,110,0.16)",
            font=dict(color="#132638", size=11),
            x=0.01,
            y=0.99,
        ),
        hoverlabel=dict(
            bgcolor="#eaf3fb",
            bordercolor="#4a90d9",
            font=dict(color="#132638", size=12),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# BIM 3D Visualization
# ---------------------------------------------------------------------------

def _box_mesh(
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
    color: str,
    name: str,
    opacity: float = 0.88,
    showlegend: bool = True,
) -> go.Mesh3d:
    """Crea un mesh 3D a forma di parallelepipedo (scatola)."""
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 0, 4, 4, 4, 1, 1, 2, 2, 3, 3]
    j = [1, 3, 4, 5, 7, 0, 2, 5, 3, 6, 0, 7]
    k = [2, 2, 5, 6, 6, 7, 5, 6, 6, 7, 4, 4]
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.3, roughness=0.6),
        lightposition=dict(x=100, y=200, z=150),
        hovertext=name,
        hoverinfo="text",
    )


def render_bim_3d(bridge_data: Dict) -> go.Figure:
    """
    Visualizzazione BIM 3D del ponte con Plotly.

    Geometria semplificata ma realistica:
    - Spalle (abutments) alle due estremita'
    - Pile (piers) con colori condizione
    - Impalcato (deck)
    - Appoggi (bearings)

    Args:
        bridge_data: dizionario da Bridge.to_dict()

    Returns:
        Figura Plotly 3D
    """
    components = bridge_data.get("components", {})
    length = float(bridge_data.get("length", 100))
    width = float(bridge_data.get("width", 10))
    bridge_name = bridge_data.get("name", "Ponte")

    # Normalizzazione dimensioni per visualizzazione
    L = max(length / 8.0, 5.0)
    W = max(width / 8.0, 1.0)

    def comp_color(comp_name: str) -> str:
        comp = components.get(comp_name, {})
        obs_min = comp.get("observed_condition_min")
        obs_max = comp.get("observed_condition_max")
        cond_true = comp.get("current_condition", 0.5)
        if obs_min is not None and obs_max is not None:
            cond = (obs_min + obs_max) / 2.0
        else:
            cond = cond_true
        return condition_to_hex(cond)

    fig = go.Figure()

    # --- Spalle (abutment) ---
    spalle_c = comp_color("spalle")
    fig.add_trace(_box_mesh(0, W, 0, W, 0, W * 2.5, spalle_c, "Spalle (sin.)", showlegend=True))
    fig.add_trace(_box_mesh(L - W, L, 0, W, 0, W * 2.5, spalle_c, "Spalle (des.)", showlegend=False))

    # --- Pile (piers) ---
    n_piers = max(1, int(length / 25))
    pile_c = comp_color("pile")
    pier_h = W * 2.3
    for i in range(n_piers):
        px = W + (L - 2 * W) * (i + 1) / (n_piers + 1)
        fig.add_trace(_box_mesh(
            px - W * 0.15, px + W * 0.15,
            W * 0.20, W * 0.80,
            0, pier_h,
            pile_c, f"Pila {i + 1}", showlegend=(i == 0),
            opacity=0.92,
        ))

    # --- Appoggi (bearings) — piccole basi sulle pile ---
    appoggi_c = comp_color("appoggi")
    for i in range(n_piers):
        px = W + (L - 2 * W) * (i + 1) / (n_piers + 1)
        fig.add_trace(_box_mesh(
            px - W * 0.22, px + W * 0.22,
            W * 0.15, W * 0.85,
            pier_h, pier_h + W * 0.12,
            appoggi_c, f"Appoggio {i + 1}", showlegend=(i == 0),
            opacity=0.95,
        ))

    # --- Impalcato (deck) ---
    impalcato_c = comp_color("impalcato")
    deck_z0 = pier_h + W * 0.12
    deck_z1 = deck_z0 + W * 0.18
    fig.add_trace(_box_mesh(0, L, 0, W, deck_z0, deck_z1, impalcato_c, "Impalcato", opacity=0.85))

    # --- Guardrail (decorativo) ---
    rail_h = deck_z1 + W * 0.15
    rail_color = "#7f8c8d"
    fig.add_trace(_box_mesh(0, L, -W * 0.05, 0, deck_z1, rail_h, rail_color, "Guardrail", opacity=0.5, showlegend=False))
    fig.add_trace(_box_mesh(0, L, W, W * 1.05, deck_z1, rail_h, rail_color, "Guardrail R", opacity=0.5, showlegend=False))

    # --- Annotazioni condizione nel titolo ---
    comp_summary = " | ".join(
        f"{cn.capitalize()}: {condition_label(components.get(cn, {}).get('current_condition', 0.5))}"
        for cn in ["impalcato", "pile", "spalle", "appoggi"]
        if cn in components
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Lunghezza (m)",
                showgrid=True,
                gridcolor="rgba(20,60,110,0.09)",
                backgroundcolor="#ffffff",
                tickfont=dict(color="#5a6b7b"),
                title_font=dict(color="#4a5c6e"),
            ),
            yaxis=dict(
                title="Larghezza (m)",
                showgrid=True,
                gridcolor="rgba(20,60,110,0.09)",
                backgroundcolor="#ffffff",
                tickfont=dict(color="#5a6b7b"),
                title_font=dict(color="#4a5c6e"),
            ),
            zaxis=dict(
                title="Altezza (m)",
                showgrid=True,
                gridcolor="rgba(20,60,110,0.09)",
                backgroundcolor="#ffffff",
                tickfont=dict(color="#5a6b7b"),
                title_font=dict(color="#4a5c6e"),
            ),
            bgcolor="#ffffff",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            aspectmode="manual",
            aspectratio=dict(x=max(L / W, 2.0), y=1.0, z=0.6),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        height=520,
        title=dict(
            text=f"🏗️ BIM 3D — {bridge_name}<br>"
                 f"<sup style='font-size:11px;color:#5a6b7b'>{comp_summary}</sup>",
            font=dict(size=18, color="#132638", family="Arial"),
            x=0.5,
        ),
        paper_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="rgba(20,60,110,0.16)",
            font=dict(color="#132638", size=10),
            x=1.02,
            y=0.5,
        ),
        hoverlabel=dict(
            bgcolor="#eaf3fb",
            bordercolor="#4a90d9",
            font=dict(color="#132638", size=12),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Grafico KPI avanzato con Plotly
# ---------------------------------------------------------------------------

def render_kpi_timeline(history_df, agent_names: List[str]) -> go.Figure:
    """Grafico interattivo dell'andamento Health Index nel tempo."""
    import pandas as pd
    fig = go.Figure()

    colors = ["#4a90d9", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c"]

    for i, agent in enumerate(agent_names):
        agent_df = history_df[history_df["agent"] == agent].sort_values("quarter")
        if agent_df.empty:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=agent_df["quarter"],
            y=agent_df["health_index"],
            mode="lines+markers",
            name=agent,
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>{agent}</b><br>Trimestre: %{{x}}<br>Health Index: %{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="📈 Health Index nel Tempo", font=dict(size=16, color="#132638"), x=0.5),
        xaxis=dict(title="Trimestre", gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b")),
        yaxis=dict(title="Health Index", tickformat=".0%", gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(bgcolor="rgba(13,17,23,0.85)", bordercolor="rgba(20,60,110,0.16)", font=dict(color="#132638")),
        hoverlabel=dict(bgcolor="#eaf3fb", font=dict(color="#132638")),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_budget_timeline(history_df, agent_names: List[str]) -> go.Figure:
    """Grafico interattivo del budget nel tempo."""
    fig = go.Figure()
    colors = ["#4a90d9", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c"]

    for i, agent in enumerate(agent_names):
        agent_df = history_df[history_df["agent"] == agent].sort_values("quarter")
        if agent_df.empty:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=agent_df["quarter"],
            y=agent_df["budget"],
            mode="lines",
            name=agent,
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.12),
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{agent}</b><br>T%{{x}}<br>Budget: €%{{y:,.0f}}<extra></extra>",
        ))

    # Linea zero budget
    fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c", line_width=1.5, annotation_text="Budget = 0")

    fig.update_layout(
        title=dict(text="💰 Evoluzione Budget", font=dict(size=16, color="#132638"), x=0.5),
        xaxis=dict(title="Trimestre", gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b")),
        yaxis=dict(title="Budget (€)", tickformat=",.0f", gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(bgcolor="rgba(13,17,23,0.85)", bordercolor="rgba(20,60,110,0.16)", font=dict(color="#132638")),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_fragility_curves(bridge_type: str = "standard", highlight_pga: float = None) -> go.Figure:
    """
    Visualizza le curve di fragilita' lognormali per 4 stati di danno.

    Args:
        bridge_type: 'standard', 'vecchio', 'moderno'
        highlight_pga: PGA specifico da evidenziare (linea verticale)
    """
    import numpy as np
    from infralab.core.hazards.earthquake import FRAGILITY_PARAMS, FRAGILITY_COLORS, fragility_lognormal

    pga_vals = np.linspace(0.0, 1.2, 300)
    params = FRAGILITY_PARAMS.get(bridge_type, FRAGILITY_PARAMS["standard"])

    fig = go.Figure()

    for state, p in params.items():
        probs = [fragility_lognormal(float(pga), p["median_pga"], p["beta"]) for pga in pga_vals]
        color = FRAGILITY_COLORS.get(state, "#888888")
        fig.add_trace(go.Scatter(
            x=list(pga_vals),
            y=probs,
            mode="lines",
            name=f"P(≥{state})",
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>P(≥{state})</b><br>"
                "PGA: %{x:.3f} g<br>"
                "Prob.: %{y:.1%}<extra></extra>"
            ),
            fill="tozeroy" if state == "Lieve" else None,
            fillcolor=hex_to_rgba(color, 0.06),
        ))

    # Linea verticale del PGA evidenziato
    if highlight_pga is not None and highlight_pga > 0:
        fig.add_vline(
            x=highlight_pga,
            line_dash="dash",
            line_color="#4a90d9",
            line_width=2,
            annotation_text=f"PGA = {highlight_pga:.3f}g",
            annotation_font_color="#4a90d9",
        )
        # Punti di intersezione
        for state, p in params.items():
            prob_at = fragility_lognormal(highlight_pga, p["median_pga"], p["beta"])
            color = FRAGILITY_COLORS.get(state, "#888888")
            fig.add_trace(go.Scatter(
                x=[highlight_pga],
                y=[prob_at],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle-open", line=dict(width=2, color=color)),
                showlegend=False,
                hovertemplate=f"<b>{state}</b>: {prob_at:.1%}<extra></extra>",
            ))

    # Zone di rischio sfondo
    fig.add_vrect(x0=0.0, x1=0.05, fillcolor="rgba(46,204,113,0.06)", line_width=0, annotation_text="Bassa sismicità", annotation_position="top left")
    fig.add_vrect(x0=0.20, x1=0.40, fillcolor="rgba(231,76,60,0.06)", line_width=0, annotation_text="Alta sismicità", annotation_position="top right")

    type_labels = {"standard": "Standard", "vecchio": "Pre-normativa (Vecchio)", "moderno": "Post-2003 (Moderno)"}
    fig.update_layout(
        title=dict(
            text=f"📉 Curve di Fragilità — Ponti {type_labels.get(bridge_type, bridge_type)}<br>"
                 "<sup style='color:#5a6b7b'>Modello Lognormale (HAZUS-MH MR5) — 4 stati di danno</sup>",
            font=dict(size=17, color="#132638", family="Arial"),
            x=0.5,
        ),
        xaxis=dict(
            title="PGA (g)", range=[0, 1.2],
            gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b"),
            title_font=dict(color="#4a5c6e"),
        ),
        yaxis=dict(
            title="Probabilità di superamento", tickformat=".0%", range=[0, 1.05],
            gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b"),
            title_font=dict(color="#4a5c6e"),
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)", bordercolor="rgba(20,60,110,0.16)",
            font=dict(color="#132638"), title=dict(text="Stato di Danno", font=dict(color="#132638")),
        ),
        hoverlabel=dict(bgcolor="#eaf3fb", font=dict(color="#132638")),
        height=430,
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def render_seismic_timeline(earthquake_events: List[Dict]) -> go.Figure:
    """
    Timeline interattiva degli eventi sismici con magnitudine e ponti danneggiati.

    Args:
        earthquake_events: lista di dict (formato EarthquakeEvent.to_dict())
    """
    if not earthquake_events:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="🌍 Nessun evento sismico registrato", font=dict(color="#132638"), x=0.5),
            paper_bgcolor="#ffffff", height=200,
        )
        return fig

    quarters = [ev.get("quarter", 0) for ev in earthquake_events]
    mags = [ev.get("magnitude", 5.0) for ev in earthquake_events]
    n_damaged = [len(ev.get("damage_reports", [])) for ev in earthquake_events]
    names = [ev.get("name", "Terremoto") for ev in earthquake_events]

    # Colore per magnitudine
    mag_colors = []
    for m in mags:
        if m >= 6.0:
            mag_colors.append("#e74c3c")
        elif m >= 5.0:
            mag_colors.append("#e67e22")
        else:
            mag_colors.append("#f1c40f")

    hover_texts = [
        f"<b>{n}</b><br>Magnitudine: M{m}<br>Trimestre: {q}<br>Ponti danneggiati: {d}"
        for n, m, q, d in zip(names, mags, quarters, n_damaged)
    ]

    fig = go.Figure()

    # Barre magnitudine
    fig.add_trace(go.Bar(
        x=quarters,
        y=mags,
        name="Magnitudine",
        marker_color=mag_colors,
        opacity=0.85,
        hovertext=hover_texts,
        hoverinfo="text",
        yaxis="y1",
        text=[f"M{m}" for m in mags],
        textposition="outside",
        textfont=dict(color="#132638", size=9),
    ))

    # Scatter ponti danneggiati (asse secondario)
    if any(d > 0 for d in n_damaged):
        fig.add_trace(go.Scatter(
            x=quarters,
            y=n_damaged,
            name="Ponti Danneggiati",
            mode="markers+lines",
            marker=dict(size=[8 + d * 3 for d in n_damaged], color="#4a90d9", symbol="diamond"),
            line=dict(color="#4a90d9", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="T%{x}<br>Ponti danneggiati: %{y}<extra></extra>",
        ))

    # Linea soglia pericolo (M≥6.0)
    fig.add_hline(y=6.0, line_dash="dash", line_color="#e74c3c", line_width=1,
                  annotation_text="M6.0 soglia alta pericolosità",
                  annotation_font_color="#e74c3c", annotation_position="top right")

    fig.update_layout(
        title=dict(
            text=f"🌍 Timeline Sismicità — {len(earthquake_events)} eventi registrati",
            font=dict(size=16, color="#132638", family="Arial"), x=0.5,
        ),
        xaxis=dict(
            title="Trimestre", gridcolor="rgba(20,60,110,0.09)",
            tickfont=dict(color="#5a6b7b"), title_font=dict(color="#4a5c6e"),
        ),
        yaxis=dict(
            title="Magnitudine (M)", range=[3.5, max(mags) + 0.8] if mags else [3.5, 7.5],
            gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b"),
            title_font=dict(color="#4a5c6e"),
        ),
        yaxis2=dict(
            title="Ponti Danneggiati",
            overlaying="y", side="right",
            tickfont=dict(color="#4a90d9"), title_font=dict(color="#4a90d9"),
            showgrid=False,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)", bordercolor="rgba(20,60,110,0.16)",
            font=dict(color="#132638"),
        ),
        hoverlabel=dict(bgcolor="#eaf3fb", font=dict(color="#132638")),
        height=380,
        margin=dict(l=10, r=60, t=60, b=10),
        barmode="overlay",
    )
    return fig


def render_asset_value_bar(bridges_dict: Dict) -> go.Figure:
    """Bar chart interattivo del valore asset corrente per ponte."""
    names, values, colors = [], [], []

    for bridge in sorted(bridges_dict.values(), key=lambda b: b.current_asset_value, reverse=True):
        names.append(f"[{bridge.id}] {bridge.name[:20]}")
        values.append(bridge.current_asset_value)
        colors.append(condition_to_hex(bridge.min_condition))

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"€{v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(color="#132638", size=10),
        hovertemplate="<b>%{x}</b><br>Valore: €%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="🏛️ Valore Asset per Ponte", font=dict(size=16, color="#132638"), x=0.5),
        xaxis=dict(tickfont=dict(color="#5a6b7b", size=9), tickangle=-35),
        yaxis=dict(title="Valore (€)", tickformat=",.0f", gridcolor="rgba(20,60,110,0.09)", tickfont=dict(color="#5a6b7b")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=380,
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig
