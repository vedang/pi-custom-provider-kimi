import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { test } from "vitest";

import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  KIMI_API_ID,
  KIMI_BASE_URL,
  type KimiProviderConfig,
  type KimiRuntimeSettings,
  applyKimiPayloadDefaults,
  buildKimiProviderConfig,
  createKimiStreamSimple,
} from "../src/core";
import kimiCustomExtension from "../src/index";

const indexPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "src",
  "index.ts",
);

const providerInput = {
  streamSimple: (() => ({}) as never) as never,
};

function buildConfig(
  env: Record<string, string | undefined> = {},
): ReturnType<typeof buildKimiProviderConfig> {
  return buildKimiProviderConfig(providerInput, env);
}

function createTestModel(id = "kimi-k2.6") {
  return {
    id,
    name: `Test model ${id}`,
    provider: "kimi-custom",
    api: "openai-completions",
    baseUrl: "https://example.invalid/v1",
    apiKey: "placeholder-key",
    reasoning: true,
    input: ["text", "image"] as Array<"text" | "image">,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1,
    maxTokens: 1,
  };
}

function createCapturedInvocationRecorder() {
  let capturedOptions: Record<string, unknown> | undefined;
  let capturedModel: Record<string, unknown> | undefined;
  const baseStream = (
    model: unknown,
    _context: unknown,
    options?: Record<string, unknown>,
  ) => {
    capturedModel = model as Record<string, unknown>;
    capturedOptions = options;
    return {
      push() {},
      end() {},
    } as never;
  };
  return {
    baseStream,
    getCapturedOptions: () => capturedOptions,
    getCapturedModel: () => capturedModel,
  };
}

function buildRuntimeSettings(
  overrides: Partial<KimiRuntimeSettings> = {},
): KimiRuntimeSettings {
  return {
    temperature: DEFAULT_TEMPERATURE,
    topP: DEFAULT_TOP_P,
    ...overrides,
  };
}

function applyDefaultsWithRuntime(
  overrides: Partial<KimiRuntimeSettings> = {},
): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  applyKimiPayloadDefaults(payload, buildRuntimeSettings(overrides));
  return payload;
}

function invokeCapturedOnPayload(
  capturedOptions: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  const result = (
    capturedOptions?.onPayload as ((payload: unknown) => unknown) | undefined
  )?.(payload);
  return (result ?? payload) as Record<string, unknown>;
}

function createStreamRecorderWithEnv(env: Record<string, string | undefined>) {
  const recorder = createCapturedInvocationRecorder();
  const streamSimple = createKimiStreamSimple(
    recorder.baseStream as never,
    env,
  );

  return {
    recorder,
    streamSimple,
  };
}

function assertPayloadDefaults(
  payload: Record<string, unknown>,
  temperature = DEFAULT_TEMPERATURE,
  topP = DEFAULT_TOP_P,
): void {
  assert.equal(payload.temperature, temperature);
  assert.equal(payload.top_p, topP);
  assert.deepEqual(payload.thinking, { type: "enabled", keep: "all" });
}

type ExpectedModelProps = {
  id: string;
  name: string;
  reasoning: boolean;
  baseUrl: string;
  apiKey: string;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
};

function assertModelProps(
  model: KimiProviderConfig["models"][number],
  expected: ExpectedModelProps,
) {
  assert.equal(model.id, expected.id);
  assert.equal(model.name, expected.name);
  assert.equal(model.reasoning, expected.reasoning);
  assert.deepEqual(model.input, ["text", "image"]);
  assert.equal(model.baseUrl, expected.baseUrl);
  assert.equal(model.apiKey, expected.apiKey);
  assert.deepEqual(model.cost, expected.cost);
  assert.equal(model.contextWindow, expected.contextWindow);
  assert.equal(model.maxTokens, expected.maxTokens);
  assert.deepEqual(model.compat, {
    supportsStore: false,
    supportsDeveloperRole: false,
    supportsReasoningEffort: false,
    maxTokensField: "max_completion_tokens",
  });
}

function assertModelList(
  models: KimiProviderConfig["models"],
  expectedModels: ExpectedModelProps[],
) {
  assert.equal(models.length, expectedModels.length);
  for (const [index, expectedModel] of expectedModels.entries()) {
    assertModelProps(models[index], expectedModel);
  }
}

test("index extension registers kimi-custom provider", () => {
  const source = readFileSync(indexPath, "utf-8");
  assert.match(source, /registerProvider\([\s\S]*"kimi-custom"/);

  let registeredName: string | undefined;
  let registeredConfig: KimiProviderConfig | undefined;
  kimiCustomExtension({
    registerProvider(name: string, config: KimiProviderConfig) {
      registeredName = name;
      registeredConfig = config;
    },
  } as never);

  assert.equal(registeredName, "kimi-custom");
  assert.equal(registeredConfig?.api, KIMI_API_ID);
  assert.equal(registeredConfig?.baseUrl, KIMI_BASE_URL);
});

test("buildKimiProviderConfig uses dedicated Kimi API wiring for fixed sampling", () => {
  const config = buildConfig({
    MOONSHOT_API_KEY: "moonshot-key",
  });

  assert.equal(config.api, KIMI_API_ID);
  assert.deepEqual(
    config.models.map((model) => model.api),
    [KIMI_API_ID, KIMI_API_ID, KIMI_API_ID],
  );
});

test("buildKimiProviderConfig returns no models when MOONSHOT_API_KEY is not configured", () => {
  const config = buildConfig();

  assert.equal(config.api, KIMI_API_ID);
  assert.equal(config.baseUrl, KIMI_BASE_URL);
  assert.equal(config.apiKey, "MOONSHOT_API_KEY");
  assert.equal(config.models.length, 0);
});

test("buildKimiProviderConfig registers only requested Kimi K2 models when MOONSHOT_API_KEY is set", () => {
  const config = buildConfig({
    MOONSHOT_API_KEY: "moonshot-key",
  });
  const expectedModels: ExpectedModelProps[] = [
    {
      id: "kimi-k2.6",
      name: "Kimi K2.6",
      reasoning: true,
      baseUrl: KIMI_BASE_URL,
      apiKey: "moonshot-key",
      cost: { input: 0.95, output: 4.0, cacheRead: 0.16, cacheWrite: 0 },
      contextWindow: 262_144,
      maxTokens: 32_768,
    },
    {
      id: "kimi-k2.7-code",
      name: "Kimi K2.7 Code",
      reasoning: true,
      baseUrl: KIMI_BASE_URL,
      apiKey: "moonshot-key",
      cost: { input: 0.95, output: 4.0, cacheRead: 0.19, cacheWrite: 0 },
      contextWindow: 262_144,
      maxTokens: 32_768,
    },
    {
      id: "kimi-k2.7-code-highspeed",
      name: "Kimi K2.7 Code HighSpeed",
      reasoning: true,
      baseUrl: KIMI_BASE_URL,
      apiKey: "moonshot-key",
      cost: { input: 1.9, output: 8.0, cacheRead: 0.38, cacheWrite: 0 },
      contextWindow: 262_144,
      maxTokens: 32_768,
    },
  ];

  assertModelList(config.models, expectedModels);
  assert.deepEqual(
    config.models.map((model) => model.id),
    ["kimi-k2.6", "kimi-k2.7-code", "kimi-k2.7-code-highspeed"],
  );
});

test("buildKimiProviderConfig ignores non-official Kimi key env names", () => {
  const config = buildConfig({
    KIMI_API_KEY: "kimi-key",
    PI_KIMI_API_KEY: "pi-kimi-key",
    KIMI_CUSTOM_API_KEY: "custom-key",
  });

  assert.equal(config.models.length, 0);
});

test("buildKimiProviderConfig ignores base-url env overrides", () => {
  const config = buildConfig({
    MOONSHOT_API_KEY: "moonshot-key",
    MOONSHOT_BASE_URL: "https://legacy.example.invalid",
    KIMI_BASE_URL: "https://legacy.example.invalid",
    PI_KIMI_CUSTOM_BASE_URL: "https://legacy.example.invalid",
  });

  assert.equal(config.baseUrl, KIMI_BASE_URL);
  assert.equal(config.models[0].baseUrl, KIMI_BASE_URL);
});

test("applyKimiPayloadDefaults injects Kimi-safe temperature and preserved thinking", () => {
  const payload = applyDefaultsWithRuntime();

  assertPayloadDefaults(payload);
});

test("createKimiStreamSimple routes known model IDs to Kimi endpoint and key", () => {
  const { recorder, streamSimple } = createStreamRecorderWithEnv({
    MOONSHOT_API_KEY: "moonshot-key",
  });

  streamSimple(createTestModel("kimi-k2.7-code"), { messages: [] }, {});

  const capturedModel = recorder.getCapturedModel();
  assert.equal(capturedModel?.baseUrl, KIMI_BASE_URL);
  assert.equal(capturedModel?.apiKey, "moonshot-key");
});

test("createKimiStreamSimple forces Kimi fixed sampling values", () => {
  const { recorder, streamSimple } = createStreamRecorderWithEnv({
    MOONSHOT_API_KEY: "moonshot-key",
    PI_TEMPERATURE: "0.42",
  });

  streamSimple(
    createTestModel("kimi-k2.6"),
    { messages: [] },
    { temperature: 0.75, top_p: 0.5 },
  );

  const capturedOptions = recorder.getCapturedOptions();
  assert.equal(capturedOptions?.temperature, DEFAULT_TEMPERATURE);
  assertPayloadDefaults(invokeCapturedOnPayload(capturedOptions));
});

test("createKimiStreamSimple overrides caller apiKey with routed Kimi key", () => {
  const { recorder, streamSimple } = createStreamRecorderWithEnv({
    MOONSHOT_API_KEY: "moonshot-key",
  });

  streamSimple(
    createTestModel("kimi-k2.6"),
    { messages: [] },
    { apiKey: "caller-key" },
  );

  const capturedModel = recorder.getCapturedModel();
  const capturedOptions = recorder.getCapturedOptions();
  assert.equal(capturedModel?.apiKey, "moonshot-key");
  assert.equal(capturedOptions?.apiKey, "moonshot-key");
});

test("createKimiStreamSimple enforces payload defaults while preserving caller onPayload", () => {
  const { recorder, streamSimple } = createStreamRecorderWithEnv({
    MOONSHOT_API_KEY: "moonshot-key",
  });

  let callerOnPayloadSeen = false;
  streamSimple(
    createTestModel("kimi-k2.6"),
    { messages: [] },
    {
      onPayload(payload) {
        callerOnPayloadSeen = true;
        (payload as Record<string, unknown>).fromCaller = true;
      },
    },
  );

  const payload = invokeCapturedOnPayload(recorder.getCapturedOptions());

  assert.equal(callerOnPayloadSeen, true);
  assert.equal(payload.fromCaller, true);
  assertPayloadDefaults(payload);
});

test("createKimiStreamSimple keeps caller replacement payload and applies defaults", () => {
  const { recorder, streamSimple } = createStreamRecorderWithEnv({
    MOONSHOT_API_KEY: "moonshot-key",
  });

  streamSimple(
    createTestModel("kimi-k2.7-code"),
    { messages: [] },
    {
      onPayload() {
        return { replaced: true };
      },
    },
  );

  const payload = invokeCapturedOnPayload(recorder.getCapturedOptions());

  assert.equal(payload.replaced, true);
  assertPayloadDefaults(payload);
});
