import type { AnalysisResult } from "./types";

const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export const MAX_BYTES = 10 * 1024 * 1024;
export const MAX_TEXT = 300;

export class ApiError extends Error {}

/** Checks done in the browser before anything is uploaded. Returns an error message or null. */
export function validateFile(file: File): string | null {
  if (!file.type.startsWith("image/")) return "That file isn't an image. Choose a JPG or PNG.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_BYTES) return "That image is larger than 10 MB. Choose a smaller one.";
  return null;
}

export async function analyse(file: File, title: string, variant: string): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("title", title.trim());
  form.append("variant", variant.trim());

  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/analyse`, { method: "POST", body: form });
  } catch {
    throw new ApiError("Couldn't reach the analysis server. Check your connection and try again.");
  }
  if (!res.ok) {
    let detail = `The server returned an error (${res.status}). Try again.`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail);
  }
  return res.json();
}
