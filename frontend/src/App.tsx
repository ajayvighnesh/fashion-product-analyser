import { useState, type FormEvent } from "react";
import { analyse, ApiError, MAX_TEXT, validateFile } from "./api";
import { ImageDrop } from "./components/ImageDrop";
import { PromptCard } from "./components/PromptCard";
import { ResultCard } from "./components/ResultCard";
import type { AnalysisResult } from "./types";

const STEPS = [
  ["Upload", "A photo of one product on a plain background works best."],
  ["Check", "The type and colour are read from the photo and compared with the title."],
  ["Create", "Describe a new version and get a prompt for an AI image tool."],
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [variant, setVariant] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const chooseFile = (f: File) => {
    const problem = validateFile(f);
    setError(problem);
    setResult(null);
    setFile(problem ? null : f);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Add a product photo first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await analyse(file, title, variant));
    } catch (err) {
      setResult(null);
      setError(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <p className="eyebrow">Product data · design ideas</p>
        <h1>Fashion Product Analyser</h1>
        <p className="lede">
          Upload a product photo. See what it is and what colour it is, catch titles that don't
          match the photo, and get a ready-made prompt for a new design.
        </p>
      </header>

      <main className="grid">
        <form className="card" onSubmit={submit} noValidate>
          <ImageDrop file={file} onFile={chooseFile} onClear={() => { setFile(null); setResult(null); }} />

          <label className="field">
            <span className="label">Product title <span className="muted">(optional)</span></span>
            <input
              value={title}
              maxLength={MAX_TEXT}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Puma Men Black Running Shoes"
            />
          </label>

          <label className="field">
            <span className="label">New design idea <span className="muted">(optional)</span></span>
            <input
              value={variant}
              maxLength={MAX_TEXT}
              onChange={(e) => setVariant(e.target.value)}
              placeholder="olive green, linen, summer collection"
            />
          </label>

          {error && <p className="error" role="alert">{error}</p>}

          <button type="submit" className="btn btn--primary" disabled={loading}>
            {loading ? "Analysing…" : "Analyse product"}
          </button>
        </form>

        <div className="stack">
          {loading && (
            <section className="card skeleton" aria-label="Loading result">
              <div className="sk sk--sm" /><div className="sk sk--lg" /><div className="sk" /><div className="sk" />
            </section>
          )}
          {!loading && result && (
            <>
              <ResultCard result={result} />
              {result.prompt && <PromptCard prompt={result.prompt} />}
            </>
          )}
          {!loading && !result && (
            <section className="card card--quiet">
              <p className="eyebrow">How it works</p>
              <ol className="steps">
                {STEPS.map(([h, p], i) => (
                  <li key={h}>
                    <span className="steps__n">{i + 1}</span>
                    <div><strong>{h}</strong><p className="muted">{p}</p></div>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      </main>

      <footer className="footer muted small">
        Built with PyTorch, Hugging Face, OpenCV, LangChain, FastAPI and React ·{" "}
        <a href="https://github.com/ajayvighnesh/fashion-product-analyser">Source code</a>
      </footer>
    </div>
  );
}
