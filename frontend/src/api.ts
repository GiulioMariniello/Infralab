// Client tipizzato dell'API InfraLab (FastAPI).

export interface AssetType {
  name: string;
  display_name: string;
  components: string[];
  structural_components: string[];
  ui_meta?: UiMeta;
}

export interface UiMeta {
  icon?: string;
  demand_label?: string;
  demand_unit?: string;
  area_label?: string;
  demand_proxy_note?: string;
}

export interface PropertyField {
  label: string;
  unit?: string;
  type: "number" | "select" | "text";
  default?: number | string;
  min?: number;
  max?: number;
  required?: boolean;
  options?: string[];
}

export interface AssetTypeSchema {
  name: string;
  display_name: string;
  components: string[];
  structural_components: string[];
  property_schema: Record<string, PropertyField>;
  ui_meta: UiMeta;
  hazard_sensitivities: Record<string, number>;
  component_default_lifetimes: Record<string, number>;
  maintenance_ops: Record<string, Record<string, unknown>>;
}

export interface CustomScenario {
  scenario_id: string;
  name: string;
  description?: string;
  asset_type: string;
  params: Record<string, unknown>;
  allowed_actions?: string[] | null;
  bridges: Record<string, unknown>[];
  source?: string;
}

export interface Scenario {
  scenario_id: string;
  name: string;
  description: string;
  asset_type: string;
  asset_count: number;
  allowed_actions: string[] | null;
  source?: string;
}

export interface Kpis {
  quarter: number;
  budget: number;
  health_index: number | null;
  net_asset_value: number;
  min_reliability_index: number | null;
  max_annual_pf: number | null;
  active: boolean;
  asset_count?: number;
  active_agents?: number;
  emissions_tco2?: number;
}

export interface AgentSpec {
  name: string;
  type: string;
  provider?: string; // solo LLM: OpenAI | Google | Locale
  profile?: string;  // solo LLM: stile decisionale
  model?: string;    // solo LLM: override modello
  plugin?: string;   // solo Custom: nome del plugin agente
}

export interface AgentState {
  name: string;
  type: string;
  is_human: boolean;
  kpis: Kpis;
}

export interface HistoryRow extends Kpis {
  agents?: Record<string, Kpis>;
}

export interface ActionOption {
  action: string;
  type: string | null;
  cost: number;
  feasible: boolean;
  reason: string | null;
}

export interface AssetView {
  id: number;
  name: string;
  manager: string;
  status: string;
  busy: boolean;
  min_condition: number;
  health_index: number;
  reliability_index: number | null;
  annual_failure_probability: number | null;
  net_asset_value: number;
  risk_priority_index: number;
  network_importance: number;
  cda: string | null;
  lat: number | null;
  lon: number | null;
  components: Record<string, { condition: number; effective_age_years: number }>;
  actions: ActionOption[];
  seismic: {
    site_pga: number | null;
    distance_to_hazard_km: number | null;
    controlling_pga: number | null;
    failure_probability: number | null;
    enabled: boolean;
  } | null;
}

export interface LimitStateDetail {
  beta_annual: number | null;
  annual_pf: number | null;
  margin_annual: number | null;
  category: string | null;
}

export interface ComponentDetail {
  name: string;
  condition: number;
  effective_age_years: number;
  beta_annual: number | null;
  annual_pf: number | null;
  posterior_beta_annual: number | null;
  governing_limit_state: string | null;
  limit_states: Record<string, LimitStateDetail>;
  degradation: { year: number; condition: number | null; beta: number | null }[];
}

export interface AssetDetail {
  id: number;
  name: string;
  beta_annual: number | null;
  annual_pf: number | null;
  posterior_beta_annual: number | null;
  target_beta_annual: number | null;
  projection: { year: number; beta: number | null; pf: number | null }[];
  components: ComponentDetail[];
}

export interface SeismicAsset {
  id: number;
  name: string;
  lat: number | null;
  lon: number | null;
  site_pga: number | null;
  distance_to_hazard_km: number | null;
  controlling_pga: number | null;
  failure_probability: number | null;
  enabled: boolean;
}

export interface ConfigField {
  key: string;
  label: string;
  group: string;
  type: "number" | "bool" | "select" | "text" | "json";
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}

export interface ConfigSchema {
  fields: ConfigField[];
  defaults: Record<string, unknown>;
}

export interface SeismicState {
  enabled: boolean;
  hazard_center: { lat: number; lon: number };
  peak_pga: number;
  far_pga: number;
  decay_km: number;
  bounds: { lat0: number; lon0: number; lat1: number; lon1: number };
  grid: number[][];
  assets: SeismicAsset[];
}

export interface CatalogEntry {
  action: string;
  type: string | null;
  duration_quarters: number;
  revenue_impact: number;
}

export interface CopilotMessage {
  role: "user" | "copilot";
  content: string;
}

export interface CopilotTurn {
  agent_name: string;
  reply: string;
  proposed_plan: Record<string, string> | null;
  history: CopilotMessage[];
}

export interface BatteryGame {
  index: number;
  length_class: string;
  condition: string;
  done: boolean;
}
export interface BatteryState {
  battery_id: string;
  total: number;
  current_index: number;
  complete: boolean;
  games: BatteryGame[];
}

export interface SessionState {
  session_id: string;
  scenario_id: string;
  asset_type: string;
  condition?: string | null;
  copilot_enabled?: boolean;
  horizon_quarters?: number;
  cadence_quarters?: number;
  finished?: boolean;
  battery?: BatteryState;
  agents: AgentState[];
  action_catalog: CatalogEntry[];
  kpis: Kpis;
  assets: AssetView[];
  history: HistoryRow[];
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  assetTypes: () => fetch("/api/asset-types").then((r) => j<AssetType[]>(r)),
  scenarios: () => fetch("/api/scenarios").then((r) => j<Scenario[]>(r)),
  createSession: (scenario_id: string, agents: AgentSpec[], seed: number, params: Record<string, unknown> = {}) =>
    fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id, agents, seed, params }),
    }).then((r) => j<SessionState>(r)),
  config: () => fetch("/api/config").then((r) => j<ConfigSchema>(r)),
  codes: () => fetch("/api/codes").then((r) => j<CodeFrameworkInfo[]>(r)),
  step: (sid: string, steps: number, plans: Record<string, Record<string, string>> = {}) =>
    fetch(`/api/sessions/${sid}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps, plans }),
    }).then((r) => j<SessionState>(r)),
  act: (sid: string, plan: Record<string, string>, rationale?: string) =>
    fetch(`/api/sessions/${sid}/act`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan, rationale }),
    }).then((r) => j<SessionState>(r)),
  assetDetail: (sid: string, aid: number) =>
    fetch(`/api/sessions/${sid}/assets/${aid}`).then((r) => j<AssetDetail>(r)),
  seismic: (sid: string) => fetch(`/api/sessions/${sid}/seismic`).then((r) => j<SeismicState>(r)),
  copilot: (sid: string, message: string, provider = "Locale", agent_name?: string, model?: string) =>
    fetch(`/api/sessions/${sid}/copilot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, provider, agent_name, model }),
    }).then((r) => j<CopilotTurn>(r)),
  consent: (sid: string) =>
    fetch(`/api/sessions/${sid}/consent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agreed: true }),
    }).then((r) => j<{ consent_given: boolean }>(r)),
  toggleSeismic: (sid: string, enabled: boolean) =>
    fetch(`/api/sessions/${sid}/seismic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }).then((r) => j<SessionState>(r)),
};

export interface AssetImportResult {
  asset_type: string;
  assets: Record<string, unknown>[];
  warnings: string[];
  errors: string[];
  matched_columns: string[];
  ignored_columns: string[];
}

export const builderApi = {
  typeSchema: (name: string) =>
    fetch(`/api/asset-types/${name}/schema`).then((r) => j<AssetTypeSchema>(r)),
  templateUrl: (name: string) => `/api/asset-types/${name}/template.xlsx`,
  importFile: (assetType: string, file: File) => {
    const form = new FormData();
    form.append("asset_type", assetType);
    form.append("file", file);
    return fetch("/api/asset-import", { method: "POST", body: form }).then((r) => j<AssetImportResult>(r));
  },
  list: () => fetch("/api/custom-scenarios").then((r) => j<CustomScenario[]>(r)),
  save: (scenario: Omit<CustomScenario, "source">) =>
    fetch("/api/custom-scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scenario),
    }).then((r) => j<{ scenario_id: string; warnings: string[] }>(r)),
  validate: (scenario: Omit<CustomScenario, "source">) =>
    fetch("/api/custom-scenarios/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scenario),
    }).then((r) => j<{ scenario_id: string; warnings: string[] }>(r)),
  remove: (scenarioId: string) =>
    fetch(`/api/custom-scenarios/${scenarioId}`, { method: "DELETE" }).then((r) => j<{ deleted: string }>(r)),
};

export interface PluginInfo {
  name: string;
  entrypoint: string;
}

export interface QItem {
  key: string;
  label: string;
  type: "number" | "select" | "radio" | "likert" | "slider" | "text";
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  default?: number | string;
  low?: string;
  high?: string;
  max_chars?: number;
}
export interface QSection {
  id: string;
  title: string;
  hint?: string;
  optional?: boolean;
  items: QItem[];
}

export interface StatGroup {
  mean: number;
  lo: number;
  hi: number;
  sd?: number;
  n: number;
}
export interface StatComparison {
  a: string;
  b: string;
  mean_diff: number;
  p_value: number | null;
  significant: boolean;
  cohens_d: number | null;
  cliffs_delta: number | null;
  effect_magnitude: string | null;
}
export interface MetricStats {
  key: string;
  label: string;
  groups: Record<string, StatGroup | null>;
  comparisons: StatComparison[];
  alpha: number;
}

export const studyApi = {
  questionnaireSchema: () => fetch("/api/questionnaire/schema").then((r) => j<{ sections: QSection[] }>(r)),
  submitQuestionnaire: (sid: string, responses: Record<string, unknown>, agent_name?: string) =>
    fetch(`/api/sessions/${sid}/questionnaire`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responses, agent_name }),
    }).then((r) => j<{ saved: boolean; scores: Record<string, number | null> }>(r)),
  questionnaireStatus: (sid: string) =>
    fetch(`/api/sessions/${sid}/questionnaire`).then((r) => j<{ submitted: boolean }>(r)),
  compare: (groups: Record<string, number[]>) =>
    fetch("/api/analysis/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ groups }),
    }).then((r) => j<{ groups: Record<string, StatGroup | null>; comparisons: StatComparison[]; alpha: number }>(r)),
};

export const pluginApi = {
  templates: () => fetch("/api/plugins/templates").then((r) => j<{ agent: string; degradation: string }>(r)),
  list: (kind: "agent" | "degradation") => fetch(`/api/plugins/${kind}`).then((r) => j<PluginInfo[]>(r)),
  get: (kind: "agent" | "degradation", name: string) =>
    fetch(`/api/plugins/${kind}/${name}`).then((r) => j<{ name: string; kind: string; source: string }>(r)),
  validate: (kind: "agent" | "degradation", source: string) =>
    fetch("/api/plugins/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, source }),
    }).then((r) => j<{ ok: boolean; message: string }>(r)),
  save: (kind: "agent" | "degradation", name: string, source: string) =>
    fetch(`/api/plugins/${kind}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, source }),
    }).then((r) => j<{ name: string; kind: string; message: string }>(r)),
  remove: (kind: "agent" | "degradation", name: string) =>
    fetch(`/api/plugins/${kind}/${name}`, { method: "DELETE" }).then((r) => j<{ deleted: string }>(r)),
};

export const AGENT_TYPES = [
  "Euristico",
  "GA Orizzonte Variabile",
  "Algoritmo Memetico",
  "MCTS",
  "RL Q-Learning",
];

export const EDITOR_AGENT_TYPES = ["Umano", ...AGENT_TYPES];

export const CAMPAIGN_SWEEP_PARAMS = [
  "env_degradation_std",
  "deterioration_exponent",
  "deterioration_process_cov",
  "traffic_degradation_factor",
  "reliability_target_beta",
  "gamma_process_theta",
  "traffic_growth_annual",
  "climate_degradation_trend_annual",
];

export interface CampaignAggregate {
  agent: string;
  grid_value: number | null;
  runs: number;
  mean_reward: number | null;
  mean_health: number | null;
  mean_beta_margin: number | null;
  std_beta_margin: number;
  mean_quarters: number | null;
  mean_emissions_tco2: number | null;
  failure_rate: number;
}

export interface CampaignResult {
  scenario_id: string;
  grid_param: string | null;
  run_count: number;
  completed: number;
  failed: number;
  aggregates: CampaignAggregate[];
  stats?: { metrics: MetricStats[]; note?: string };
  runs: Record<string, unknown>[];
}

export interface CampaignStatus {
  status: "running" | "done" | "error";
  done: number;
  total: number;
  result: CampaignResult | null;
  error: string | null;
}

export interface CampaignAgentObj {
  id: string;
  type: string;
  provider?: string;
  profile?: string;
  model?: string;
  plugin?: string;
  max_rounds?: number;
}

export interface CampaignRequest {
  scenario_id: string;
  agents: (string | CampaignAgentObj)[];
  horizon_quarters: number;
  replicates: number;
  seed?: number;
  grid_param?: string | null;
  grid_values?: number[];
  compare_frameworks?: string[];
}

export interface CodeFrameworkInfo {
  name: string;
  display_name: string;
  source: string;
  beta_reference_by_cc: Record<string, number> | null;
  reference_period_years: number | null;
}

export interface LlmStatus {
  openai: { configured: boolean; model: string };
  google: { configured: boolean; model: string };
  locale: { configured: boolean; base_url: string; model: string };
  profiles: string[];
}

export interface PlayerLink {
  participant: string;
  token: string;
  path: string;
  condition?: string;
}

export interface ExperimentAnalysis {
  scenario_id: string;
  n_sessions: number;
  rows: Record<string, unknown>[];
  conditions: string[];
  metrics: MetricStats[];
}

export interface ResearchSession {
  session_id: string;
  scenario_id: string | null;
  phase: string | null;
  quarter: number | null;
  agents?: string[];
  active_agents?: string[];
  archived: boolean;
}

export interface PlayerJoinState extends SessionState {
  player_name: string;
  consent_given?: boolean;
}

function adminHeaders(token: string) {
  return { "Content-Type": "application/json", "X-Admin-Token": token };
}

export const researchApi = {
  adminLogin: (password: string) =>
    fetch("/api/auth/admin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    }).then((r) => j<{ token: string }>(r)),
  participants: (token: string) =>
    fetch("/api/admin/participants", { headers: adminHeaders(token) }).then((r) => j<{ participants: string[] }>(r)),
  saveParticipants: (token: string, participants: string[]) =>
    fetch("/api/admin/participants", {
      method: "PUT",
      headers: adminHeaders(token),
      body: JSON.stringify({ participants }),
    }).then((r) => j<{ participants: string[] }>(r)),
  conditions: (token: string) =>
    fetch("/api/admin/conditions", { headers: adminHeaders(token) }).then((r) => j<{ human_conditions: string[] }>(r)),
  playerLinks: (token: string, scenario_id: string, participants: string[], condition = "Uomo") =>
    fetch("/api/admin/player-links", {
      method: "POST",
      headers: adminHeaders(token),
      body: JSON.stringify({ scenario_id, participants, condition }),
    }).then((r) => j<{ scenario_id: string; condition: string; links: PlayerLink[] }>(r)),
  experiment: (token: string, scenario_id: string) =>
    fetch(`/api/admin/experiment/${scenario_id}`, { headers: adminHeaders(token) }).then((r) => j<ExperimentAnalysis>(r)),
  experimentCsvUrl: (scenario_id: string) => `/api/admin/experiment/${scenario_id}/export.csv`,
  researchSessions: (token: string) =>
    fetch("/api/admin/research-sessions", { headers: adminHeaders(token) }).then((r) => j<ResearchSession[]>(r)),
  playerJoin: (token: string) =>
    fetch("/api/player/join", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }).then((r) => j<PlayerJoinState>(r)),
  batteryAdvance: (token: string) =>
    fetch("/api/battery/advance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }).then((r) => j<PlayerJoinState>(r)),
  batteryLinks: (adminToken: string, participants: string[]) =>
    fetch("/api/admin/battery-links", {
      method: "POST",
      headers: adminHeaders(adminToken),
      body: JSON.stringify({ participants }),
    }).then((r) => j<{ kind: string; n_participants: number; links: (PlayerLink & { games: number; sequence: string })[] }>(r)),
  experimentBattery: (adminToken: string) =>
    fetch("/api/admin/experiment-battery", { headers: adminHeaders(adminToken) }).then((r) => j<BatteryAnalysis>(r)),
  experimentBatteryCsvUrl: () => "/api/admin/experiment-battery/export.csv",
  llmStatus: () => fetch("/api/llm/status").then((r) => j<LlmStatus>(r)),
};

export interface PairedDelta {
  n_pairs: number;
  treat?: string;
  control?: string;
  n?: number;
  mean_diff?: number;
  median_diff?: number;
  p_value?: number | null;
  significant?: boolean;
  effect_r?: number | null;
  note?: string;
}
export interface BatteryMetric {
  key: string;
  label: string;
  by_length: Record<string, PairedDelta | null>;
}
export interface BatteryAnalysis {
  n_games: number;
  n_participants: number;
  length_classes: string[];
  metrics: BatteryMetric[];
  rows: Record<string, unknown>[];
}

export const campaignApi = {
  start: (req: CampaignRequest) =>
    fetch("/api/campaigns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then((r) => j<{ job_id: string; run_count: number }>(r)),
  status: (jobId: string) => fetch(`/api/campaigns/${jobId}`).then((r) => j<CampaignStatus>(r)),
};
