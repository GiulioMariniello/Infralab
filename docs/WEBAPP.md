# InfraLab web app (FastAPI + React)

Frontend serio alternativo alla UI Streamlit: backend FastAPI che espone il
motore `infralab`, frontend React/Vite con il design system InfraLab.

## Architettura

```
backend/app.py        FastAPI — /api/health, /api/asset-types, /api/scenarios,
                      /api/sessions (create/step/state)   [sessioni in-memory]
frontend/             Vite + React + TypeScript
  src/api.ts          client tipizzato dell'API
  src/ui.tsx          primitivi + marchio (palette blu/celeste, IBM Plex)
  src/Landing.tsx     landing
  src/Console.tsx     dashboard live: scenario × agente × seed, KPI, grafici
                      (Recharts), tabella asset, avanzamento trimestri
```

Il frontend riusa lo stesso motore della UI Streamlit (nessuna logica
duplicata): il backend chiama `infralab.lab.build.build_env_from_scenario`,
gli agenti via `infralab.agents.adapters.make_agent`, e legge KPI/affidabilità
dall'`AssetEnv`.

## Avvio in sviluppo

Due processi:

```bash
# 1) backend (dalla radice repo)
./venv/Scripts/python -m uvicorn backend.app:app --reload --port 8000

# 2) frontend
cd frontend
npm install        # solo la prima volta
npm run dev        # http://localhost:5173  (proxy /api -> :8000)
```

Apri http://localhost:5173, "Apri la console", scegli scenario/agente/seed,
"Avvia sessione", poi avanza i trimestri: KPI e grafici si aggiornano dal vivo.

## Build di produzione (processo singolo)

```bash
cd frontend && npm run build     # genera frontend/dist/
./venv/Scripts/python -m uvicorn backend.app:app --port 8000
```
Se `frontend/dist/` esiste, il backend la serve automaticamente alla radice
(`http://localhost:8000/` = app React, `/api/*` = API): un solo processo, un
solo deploy. Origini CORS restringibili con la variabile `INFRALAB_CORS`
(lista separata da virgole).

## Stato

Prima fetta verticale funzionante (scenari, sessione, avanzamento, KPI,
affidabilità/health nel tempo, tabella asset) — multi-asset (ponti e muri di
sostegno) via lo stesso endpoint. La UI Streamlit resta disponibile in
parallelo (`streamlit run app.py`).
