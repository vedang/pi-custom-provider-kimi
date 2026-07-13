import type {
  Api,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  ThinkingLevelMap,
} from "@earendil-works/pi-ai";

export const KIMI_BASE_URL = "https://api.moonshot.ai/v1";
export const KIMI_API_ID = "kimi-custom-openai-completions";
export const DEFAULT_TEMPERATURE = 1.0;
const NON_THINKING_TEMPERATURE = 0.6;
export const DEFAULT_TOP_P = 0.95;

const KIMI_API_KEY_ENV_KEY = "MOONSHOT_API_KEY";

export interface KimiRuntimeSettings {
  temperature: number;
  topP: number;
}

type KimiThinkingPayload =
  | { type: "enabled"; keep: "all" }
  | { type: "disabled" };

interface KimiModelRoutingResult {
  model: unknown;
  apiKey: string | undefined;
}

export interface KimiSimpleOptions
  extends Omit<SimpleStreamOptions, "onPayload"> {
  temperature?: number;
  top_p?: number;
  topP?: number;
  apiKey?: string;
  onPayload?: (
    payload: unknown,
    model: Model<Api>,
  ) => unknown | undefined | Promise<unknown | undefined>;
}

export type KimiStreamSimple = (
  model: Model<Api>,
  context: Context,
  options?: KimiSimpleOptions,
) => AssistantMessageEventStream;

interface KimiProviderConfigInput {
  streamSimple: KimiStreamSimple;
}

interface KimiProviderModelConfig {
  id: string;
  name: string;
  reasoning: boolean;
  input: ["text", "image"];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
  compat: typeof KIMI_COMPAT;
  thinkingLevelMap?: ThinkingLevelMap;
  baseUrl: string;
  api: typeof KIMI_API_ID;
}

interface KimiModelTemplate
  extends Omit<KimiProviderModelConfig, "baseUrl" | "compat" | "api"> {}

export interface KimiProviderConfig {
  baseUrl: string;
  apiKey: string;
  api: typeof KIMI_API_ID;
  streamSimple: KimiStreamSimple;
  models: KimiProviderModelConfig[];
}

const KIMI_COMPAT = {
  supportsStore: false,
  supportsDeveloperRole: false,
  supportsReasoningEffort: false,
  maxTokensField: "max_tokens" as const,
  supportsStrictMode: false,
  thinkingFormat: "deepseek" as const,
} satisfies NonNullable<Model<"openai-completions">["compat"]>;

const KIMI_K27_THINKING_LEVEL_MAP = {
  off: null,
} satisfies ThinkingLevelMap;

const SHARED_MODEL_DEFAULTS = {
  reasoning: true,
  input: ["text", "image"] as ["text", "image"],
  contextWindow: 262_144,
  maxTokens: 32_768,
};

const KIMI_MODELS: KimiModelTemplate[] = [
  {
    ...SHARED_MODEL_DEFAULTS,
    id: "kimi-k2.6",
    name: "Kimi K2.6",
    cost: { input: 0.95, output: 4.0, cacheRead: 0.16, cacheWrite: 0 },
  },
  {
    ...SHARED_MODEL_DEFAULTS,
    id: "kimi-k2.7-code",
    name: "Kimi K2.7 Code",
    thinkingLevelMap: KIMI_K27_THINKING_LEVEL_MAP,
    cost: { input: 0.95, output: 4.0, cacheRead: 0.19, cacheWrite: 0 },
  },
  {
    ...SHARED_MODEL_DEFAULTS,
    id: "kimi-k2.7-code-highspeed",
    name: "Kimi K2.7 Code HighSpeed",
    thinkingLevelMap: KIMI_K27_THINKING_LEVEL_MAP,
    cost: { input: 1.9, output: 8.0, cacheRead: 0.38, cacheWrite: 0 },
  },
];

const KIMI_MODEL_IDS = new Set<string>(KIMI_MODELS.map((model) => model.id));
const KIMI_K27_MODEL_IDS = new Set<string>([
  "kimi-k2.7-code",
  "kimi-k2.7-code-highspeed",
]);

function parseOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function resolveKimiApiKey(
  env: Record<string, string | undefined>,
): string | undefined {
  return parseOptionalString(env[KIMI_API_KEY_ENV_KEY]);
}

function resolveKimiRuntimeSettings(
  thinking: KimiThinkingPayload = { type: "enabled", keep: "all" },
): KimiRuntimeSettings {
  return {
    temperature:
      thinking.type === "disabled"
        ? NON_THINKING_TEMPERATURE
        : DEFAULT_TEMPERATURE,
    topP: DEFAULT_TOP_P,
  };
}

function buildKimiRouteOverrides(): Pick<
  KimiProviderModelConfig,
  "api" | "baseUrl" | "compat"
> {
  return {
    api: KIMI_API_ID,
    baseUrl: KIMI_BASE_URL,
    compat: KIMI_COMPAT,
  };
}

function materializeModel(
  template: KimiModelTemplate,
  env: Record<string, string | undefined>,
): KimiProviderModelConfig | undefined {
  if (!resolveKimiApiKey(env)) return undefined;

  return {
    id: template.id,
    name: template.name,
    reasoning: template.reasoning,
    input: template.input,
    cost: template.cost,
    contextWindow: template.contextWindow,
    maxTokens: template.maxTokens,
    thinkingLevelMap: template.thinkingLevelMap,
    ...buildKimiRouteOverrides(),
  };
}

function resolveModels(
  env: Record<string, string | undefined>,
): KimiProviderModelConfig[] {
  const models: KimiProviderModelConfig[] = [];

  for (const template of KIMI_MODELS) {
    const model = materializeModel(template, env);
    if (model !== undefined) models.push(model);
  }

  return models;
}

function routeModelToKimiEndpoint(
  model: unknown,
  env: Record<string, string | undefined>,
): KimiModelRoutingResult {
  if (!model || typeof model !== "object") {
    return { model, apiKey: undefined };
  }

  const modelRecord = model as Record<string, unknown>;
  const modelId =
    typeof modelRecord.id === "string" ? modelRecord.id.trim() : undefined;
  if (!modelId || !KIMI_MODEL_IDS.has(modelId)) {
    return { model, apiKey: undefined };
  }

  const modelWithoutApiKey = Object.fromEntries(
    Object.entries(modelRecord).filter(([key]) => key !== "apiKey"),
  );

  return {
    model: {
      ...modelWithoutApiKey,
      ...buildKimiRouteOverrides(),
    },
    apiKey: resolveKimiApiKey(env),
  };
}

function resolveKimiThinkingPayload(
  model: unknown,
  reasoning: SimpleStreamOptions["reasoning"],
): KimiThinkingPayload {
  const modelId =
    model &&
    typeof model === "object" &&
    "id" in model &&
    typeof model.id === "string"
      ? model.id.trim()
      : undefined;

  if (modelId && KIMI_K27_MODEL_IDS.has(modelId)) {
    return { type: "enabled", keep: "all" };
  }

  return reasoning ? { type: "enabled", keep: "all" } : { type: "disabled" };
}

function resolvePayloadTarget(
  payload: unknown,
  nextPayload: unknown | undefined,
): unknown {
  return nextPayload === undefined ? payload : nextPayload;
}

function isPromiseLike(value: unknown): value is Promise<unknown | undefined> {
  return (
    !!value &&
    (typeof value === "object" || typeof value === "function") &&
    typeof (value as Promise<unknown>).then === "function"
  );
}

/**
 * [tag:kimi_custom_fixed_sampling]
 * Kimi K2.6/K2.7 Code reject non-default sampling values. Force documented
 * fixed values after caller hooks so role-level temperature cannot break calls.
 *
 * [tag:kimi_custom_preserved_thinking]
 * Kimi K2.7 Code always preserves thinking. Kimi K2.6 supports it via
 * `thinking.keep = "all"` while thinking is enabled, so tool-call turns can
 * replay `reasoning_content` safely.
 */
export function applyKimiPayloadDefaults(
  payload: unknown,
  runtime: KimiRuntimeSettings,
  thinking: KimiThinkingPayload = { type: "enabled", keep: "all" },
): void {
  if (!payload || typeof payload !== "object") return;
  const request = payload as Record<string, unknown>;
  // [ref:kimi_custom_fixed_sampling]
  request.temperature = runtime.temperature;
  request.top_p = runtime.topP;
  // [ref:kimi_custom_preserved_thinking]
  request.thinking = thinking;
}

/**
 * [ref:kimi_custom_fixed_sampling]
 * [ref:kimi_custom_preserved_thinking]
 *
 * [tag:kimi_custom_routed_api_key_precedence]
 * Modern OpenAI Completions streaming reads request auth from options. After
 * model-ID routing, mirror the Kimi key into options so caller-provided keys
 * cannot accidentally authenticate against wrong endpoint.
 */
export function createKimiStreamSimple(
  baseStreamSimple: KimiStreamSimple,
  env: Record<string, string | undefined> = process.env,
): KimiStreamSimple {
  return (model, context, options) => {
    const callerOnPayload = options?.onPayload;
    const routedModel = routeModelToKimiEndpoint(model, env);
    const thinking = resolveKimiThinkingPayload(
      routedModel.model,
      options?.reasoning,
    );
    const runtime = resolveKimiRuntimeSettings(thinking);
    const wrappedOptions: KimiSimpleOptions = {
      ...options,
      // [ref:kimi_custom_routed_api_key_precedence]
      apiKey: routedModel.apiKey ?? options?.apiKey,
      // [ref:kimi_custom_fixed_sampling]
      temperature: runtime.temperature,
      top_p: runtime.topP,
      topP: runtime.topP,
      onPayload: (payload: unknown, payloadModel: Model<Api>) => {
        const nextPayload = callerOnPayload?.(payload, payloadModel);
        if (isPromiseLike(nextPayload)) {
          return nextPayload.then((resolvedPayload) => {
            const target = resolvePayloadTarget(payload, resolvedPayload);
            applyKimiPayloadDefaults(target, runtime, thinking);
            return target;
          });
        }

        const target = resolvePayloadTarget(payload, nextPayload);
        applyKimiPayloadDefaults(target, runtime, thinking);
        return target;
      },
    };
    return baseStreamSimple(
      routedModel.model as Model<Api>,
      context,
      wrappedOptions,
    );
  };
}

function resolveProviderFallbackApiKey(): string {
  return `$${KIMI_API_KEY_ENV_KEY}`;
}

export function buildKimiProviderConfig(
  input: KimiProviderConfigInput,
  env: Record<string, string | undefined> = process.env,
): KimiProviderConfig {
  return {
    baseUrl: KIMI_BASE_URL,
    apiKey: resolveProviderFallbackApiKey(),
    api: KIMI_API_ID,
    streamSimple: input.streamSimple,
    models: resolveModels(env),
  };
}
