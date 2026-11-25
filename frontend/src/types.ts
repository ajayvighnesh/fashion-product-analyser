/** Shape of POST /api/analyse (see src/fashion_analyser/api.py). */
export type RGB = [number, number, number];

export interface SDPrompt {
  prompt: string;
  negative_prompt: string;
  variant: string;
}

export interface AnalysisResult {
  category: string;
  colour: string;
  gender: string | null;
  image_confidence: { category: number; colour: number };
  text_attributes: {
    colours: string[];
    raw_colours: string[];
    gender: string | null;
    category: string | null;
  };
  conflicts: string[];
  warnings: string[];
  needs_review: boolean;
  top_categories: [string, number][];
  top_colours: [string, number][];
  dominant_rgb: [RGB, number][];
  prompt: SDPrompt | null;
}
