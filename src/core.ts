import type {
  Api,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
} from "@mariozechner/pi-ai";

export const KIMI_BASE_URL = "https://api.moonshot.ai/v1";
export const KIMI_API_ID = "kimi-custom-openai-completions";
export const DEFAULT_TEMPERATURE = 1.0;
export const DEFAULT_TOP_P = 0.95;

const API_KEY_ENV_PLACEHOLDER = "MOONSHOT_API_KEY";
const KIMI_API_KEY_ENV_KEY = "MOONSHOT_API_KEY";

export interface KimiRuntimeSettings {
  temperature: number;
  topP: number;
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
  baseUrl: string;
  apiKey: string;
  api: typeof KIMI_API_ID;
}

interface KimiModelTemplate
  extends Omit<
    KimiProviderModelConfig,
    "baseUrl" | "apiKey" | "compat" | "api"
  > {}

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
  maxTokensField: "max_completion_tokens" as const,
};

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
    cost: { input: 0.95, output: 4.0, cacheRead: 0.19, cacheWrite: 0 },
  },
  {
    ...SHARED_MODEL_DEFAULTS,
    id: "kimi-k2.7-code-highspeed",
    name: "Kimi K2.7 Code HighSpeed",
    cost: { input: 1.9, output: 8.0, cacheRead: 0.38, cacheWrite: 0 },
  },
];

const KIMI_MODEL_IDS = new Set<string>(KIMI_MODELS.map((model) => model.id));

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

function resolveKimiRuntimeSettings(): KimiRuntimeSettings {
  return {
    temperature: DEFAULT_TEMPERATURE,
    topP: DEFAULT_TOP_P,
  };
}

function buildKimiRouteOverrides(
  apiKey: string,
): Pick<KimiProviderModelConfig, "api" | "baseUrl" | "apiKey" | "compat"> {
  return {
    api: KIMI_API_ID,
    baseUrl: KIMI_BASE_URL,
    apiKey,
    compat: KIMI_COMPAT,
  };
}

function materializeModel(
  template: KimiModelTemplate,
  env: Record<string, string | undefined>,
): KimiProviderModelConfig | undefined {
  const apiKey = resolveKimiApiKey(env);
  if (!apiKey) return undefined;

  return {
    id: template.id,
    name: template.name,
    reasoning: template.reasoning,
    input: template.input,
    cost: template.cost,
    contextWindow: template.contextWindow,
    maxTokens: template.maxTokens,
    ...buildKimiRouteOverrides(apiKey),
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
): unknown {
  if (!model || typeof model !== "object") return model;

  const modelRecord = model as Record<string, unknown>;
  const modelId =
    typeof modelRecord.id === "string" ? modelRecord.id.trim() : undefined;
  if (!modelId || !KIMI_MODEL_IDS.has(modelId)) return model;

  const apiKey = resolveKimiApiKey(env);
  if (!apiKey) return model;

  return {
    ...modelRecord,
    ...buildKimiRouteOverrides(apiKey),
  };
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
 * `thinking.keep = "all"`. Use one invariant payload for both models so tool
 * call turns can replay `reasoning_content` safely.
 */
export function applyKimiPayloadDefaults(
  payload: unknown,
  runtime: KimiRuntimeSettings,
): void {
  if (!payload || typeof payload !== "object") return;
  const request = payload as Record<string, unknown>;
  // [ref:kimi_custom_fixed_sampling]
  request.temperature = runtime.temperature;
  request.top_p = runtime.topP;
  // [ref:kimi_custom_preserved_thinking]
  request.thinking = { type: "enabled", keep: "all" };
}

/**
 * [ref:kimi_custom_fixed_sampling]
 * [ref:kimi_custom_preserved_thinking]
 *
 * [tag:kimi_custom_routed_api_key_precedence]
 * `streamSimpleOpenAICompletions` prioritizes `options.apiKey` over
 * `model.apiKey`. After model-ID routing, mirror the Kimi key into options so
 * caller-provided keys cannot accidentally authenticate against wrong endpoint.
 */
export function createKimiStreamSimple(
  baseStreamSimple: KimiStreamSimple,
  env: Record<string, string | undefined> = process.env,
): KimiStreamSimple {
  return (model, context, options) => {
    const runtime = resolveKimiRuntimeSettings();
    const callerOnPayload = options?.onPayload;
    const routedModel = routeModelToKimiEndpoint(model, env);
    const routedApiKey =
      routedModel && typeof routedModel === "object"
        ? parseOptionalString((routedModel as Record<string, unknown>).apiKey)
        : undefined;
    const wrappedOptions: KimiSimpleOptions = {
      ...options,
      // [ref:kimi_custom_routed_api_key_precedence]
      apiKey: routedApiKey ?? options?.apiKey,
      // [ref:kimi_custom_fixed_sampling]
      temperature: runtime.temperature,
      top_p: runtime.topP,
      topP: runtime.topP,
      onPayload: (payload: unknown, payloadModel: Model<Api>) => {
        const nextPayload = callerOnPayload?.(payload, payloadModel);
        if (isPromiseLike(nextPayload)) {
          return nextPayload.then((resolvedPayload) => {
            const target = resolvePayloadTarget(payload, resolvedPayload);
            applyKimiPayloadDefaults(target, runtime);
            return target;
          });
        }

        const target = resolvePayloadTarget(payload, nextPayload);
        applyKimiPayloadDefaults(target, runtime);
        return target;
      },
    };
    return baseStreamSimple(routedModel as Model<Api>, context, wrappedOptions);
  };
}

function resolveProviderFallbackApiKey(
  env: Record<string, string | undefined>,
): string {
  return resolveKimiApiKey(env) ?? API_KEY_ENV_PLACEHOLDER;
}

export function buildKimiProviderConfig(
  input: KimiProviderConfigInput,
  env: Record<string, string | undefined> = process.env,
): KimiProviderConfig {
  return {
    baseUrl: KIMI_BASE_URL,
    apiKey: resolveProviderFallbackApiKey(env),
    api: KIMI_API_ID,
    streamSimple: input.streamSimple,
    models: resolveModels(env),
  };
}
