import type { AnalysisResult } from "../types";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const GENDER: Record<string, string> = {
  men: "Men", women: "Women", boys: "Boys", girls: "Girls", kids: "Kids", unisex: "Unisex",
};

function Confidence({ label, value, name }: { label: string; value: number; name: string }) {
  const level = value >= 0.75 ? "high" : value >= 0.5 ? "mid" : "low";
  return (
    <div className="conf">
      <div className="conf__row">
        <span className="conf__label">{label}</span>
        <span className="conf__value">
          <span className="cap">{name}</span> <span className="muted">· {pct(value)} sure</span>
        </span>
      </div>
      <div className="conf__track" role="meter" aria-label={`${label} confidence`}
           aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}>
        <div className={`conf__fill conf__fill--${level}`} style={{ width: pct(value) }} />
      </div>
    </div>
  );
}

export function ResultCard({ result }: { result: AnalysisResult }) {
  const issues = [...result.conflicts, ...result.warnings];
  const gender = result.gender ? GENDER[result.gender] ?? result.gender : null;
  const alternatives = result.top_categories.slice(1).filter(([, p]) => p >= 0.05);

  return (
    <section className="card" aria-live="polite">
      <p className="eyebrow">Result</p>
      <h2 className="result__title">
        <span className="cap">{result.colour === "other" ? "" : result.colour}</span>{" "}
        {result.category}
        {gender && <span className="result__gender">{gender}</span>}
      </h2>

      <div className={`status ${result.needs_review ? "status--warn" : "status--ok"}`}>
        {result.needs_review ? "Needs a human check" : "Looks consistent"}
      </div>

      {issues.length > 0 && (
        <ul className="issues">
          {issues.map((i) => (
            <li key={i}>{i.charAt(0).toUpperCase() + i.slice(1)}</li>
          ))}
        </ul>
      )}

      <div className="divider" />
      <Confidence label="Product type" name={result.category} value={result.image_confidence.category} />
      <Confidence label="Colour" name={result.colour} value={result.image_confidence.colour} />

      {alternatives.length > 0 && (
        <p className="alts">
          Could also be:{" "}
          {alternatives.map(([name, p], i) => (
            <span key={name}>
              {i > 0 && ", "}
              {name} <span className="muted">({pct(p)})</span>
            </span>
          ))}
        </p>
      )}

      <div className="divider" />
      <p className="label">Main colours in the photo</p>
      <div className="swatches">
        {result.dominant_rgb.map(([rgb, share]) => (
          <div key={rgb.join(",")} className="swatch">
            <div className="swatch__chip" style={{ background: `rgb(${rgb.join(",")})` }}
                 title={`rgb(${rgb.join(", ")})`} />
            <span className="muted small">{pct(share)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
