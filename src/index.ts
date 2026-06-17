import {
  type Model,
  type SimpleStreamOptions,
  streamSimpleOpenAICompletions,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
  type KimiStreamSimple,
  buildKimiProviderConfig,
  createKimiStreamSimple,
} from "./core.js";

const streamSimpleViaOpenAICompletions: KimiStreamSimple = (
  model,
  context,
  options,
) => {
  return streamSimpleOpenAICompletions(
    model as Model<"openai-completions">,
    context,
    options as SimpleStreamOptions,
  );
};

export default function kimiCustomExtension(pi: ExtensionAPI): void {
  const streamSimple = createKimiStreamSimple(streamSimpleViaOpenAICompletions);
  pi.registerProvider("kimi-custom", buildKimiProviderConfig({ streamSimple }));
}
