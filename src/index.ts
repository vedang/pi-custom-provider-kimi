import type { Model, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
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
  return streamSimple(
    model as Model<"openai-completions">,
    context,
    options as SimpleStreamOptions,
  );
};

export default function kimiCustomExtension(pi: ExtensionAPI): void {
  const streamSimple = createKimiStreamSimple(streamSimpleViaOpenAICompletions);
  pi.registerProvider("kimi-custom", buildKimiProviderConfig({ streamSimple }));
}
