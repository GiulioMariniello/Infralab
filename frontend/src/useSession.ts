import { useCallback, useState } from "react";
import { api, researchApi, type AgentSpec, type SessionState } from "./api";

export interface SessionController {
  state: SessionState | null;
  busy: boolean;
  error: string | null;
  start: (scenarioId: string, agents: AgentSpec[], seed: number, params?: Record<string, unknown>) => Promise<void>;
  step: (n: number, plans?: Record<string, Record<string, string>>) => Promise<void>;
  act: (plan: Record<string, string>, rationale?: string) => Promise<void>;
  toggleSeismic: (enabled: boolean) => Promise<void>;
  advanceBattery: (token: string) => Promise<void>;
  adopt: (state: SessionState) => void;
}

export function useSession(): SessionController {
  const [state, setState] = useState<SessionState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wrap = useCallback(async (fn: () => Promise<SessionState>) => {
    setBusy(true);
    setError(null);
    try {
      setState(await fn());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const start = useCallback(
    (scenarioId: string, agents: AgentSpec[], seed: number, params: Record<string, unknown> = {}) =>
      wrap(() => api.createSession(scenarioId, agents, seed, params)),
    [wrap],
  );
  const step = useCallback(
    async (n: number, plans: Record<string, Record<string, string>> = {}) => {
      if (!state) return;
      await wrap(() => api.step(state.session_id, n, plans));
    },
    [wrap, state],
  );
  const act = useCallback(
    async (plan: Record<string, string>, rationale?: string) => {
      if (!state) return;
      await wrap(() => api.act(state.session_id, plan, rationale));
    },
    [wrap, state],
  );

  const toggleSeismic = useCallback(
    async (enabled: boolean) => {
      if (!state) return;
      await wrap(() => api.toggleSeismic(state.session_id, enabled));
    },
    [wrap, state],
  );

  const advanceBattery = useCallback(
    (token: string) => wrap(() => researchApi.batteryAdvance(token)),
    [wrap],
  );

  const adopt = useCallback((s: SessionState) => setState(s), []);

  return { state, busy, error, start, step, act, toggleSeismic, advanceBattery, adopt };
}

/* ---- formattazione condivisa ---- */
export const fmt = (v: number | null | undefined, d: number) => (v == null ? "—" : v.toFixed(d));
export const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
export const sci = (v: number | null | undefined) => (v == null ? "—" : v.toExponential(1));
export const eur = (v: number | null | undefined) =>
  v == null ? "—" : "€" + Math.round(v).toLocaleString("it-IT");
