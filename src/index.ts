import {
  type Model,
  type SimpleStreamOptions,
  openAICompletionsApi,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  type KimiStreamSimple,
  buildKimiProviderConfig,
  createKimiStreamSimple,
} from "./core.js";

const openAICompletions = openAICompletionsApi();

const streamSimpleViaOpenAICompletions: KimiStreamSimple = (
  model,
  context,
  options,
) => {
  return openAICompletions.streamSimple(
    model as Model<"openai-completions">,
    context,
    options as SimpleStreamOptions,
  );
};

export default function kimiCustomExtension(pi: ExtensionAPI): void {
  const streamSimple = createKimiStreamSimple(streamSimpleViaOpenAICompletions);
  pi.registerProvider("kimi-custom", buildKimiProviderConfig({ streamSimple }));
}
