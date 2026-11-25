import { useState } from "react";
import type { SDPrompt } from "../types";

function CopyBlock({ label, text }: { label: string; text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the text is still selectable */
    }
  };
  return (
    <div className="copy">
      <div className="copy__head">
        <span className="label">{label}</span>
        <button type="button" className="btn btn--small" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p className="copy__text">{text}</p>
    </div>
  );
}

export function PromptCard({ prompt }: { prompt: SDPrompt }) {
  return (
    <section className="card">
      <p className="eyebrow">New design: {prompt.variant}</p>
      <p className="muted small">
        Paste this into an AI image tool such as Stable Diffusion to create the design variant.
      </p>
      <CopyBlock label="Prompt" text={prompt.prompt} />
      <CopyBlock label="Negative prompt (what to avoid)" text={prompt.negative_prompt} />
    </section>
  );
}
