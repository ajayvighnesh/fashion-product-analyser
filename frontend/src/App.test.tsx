import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { AnalysisResult } from "./types";

const RESULT: AnalysisResult = {
  category: "Shirts",
  colour: "blue",
  gender: "men",
  image_confidence: { category: 0.93, colour: 0.88 },
  text_attributes: { colours: ["red"], raw_colours: ["red"], gender: "men", category: "Shirts" },
  conflicts: ["colour: image says 'blue' (88%), text says 'red'"],
  warnings: [],
  needs_review: true,
  top_categories: [["Shirts", 0.93], ["Tshirts", 0.05], ["Kurtas", 0.01]],
  top_colours: [["blue", 0.88], ["grey", 0.06], ["black", 0.03]],
  dominant_rgb: [[[24, 95, 165], 0.82], [[12, 68, 124], 0.14]],
  prompt: {
    prompt: "men's shirts, olive green, linen, professional e-commerce product photo",
    negative_prompt: "person, watermark",
    variant: "olive green, linen",
  },
};

const png = () => new File([new Uint8Array([137, 80, 78, 71])], "shirt.png", { type: "image/png" });

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({ ok: status < 400, status, json: async () => body });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("shows how it works before anything is uploaded", () => {
    render(<App />);
    expect(screen.getByText("How it works")).toBeInTheDocument();
  });

  it("asks for a photo when submitted empty, without calling the server", async () => {
    const fetchMock = mockFetch(200, RESULT);
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Analyse product" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Add a product photo first.");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects non-image files in the browser", async () => {
    render(<App />);
    const pdf = new File(["%PDF"], "cv.pdf", { type: "application/pdf" });
    await userEvent.upload(screen.getByTestId("file-input"), pdf, { applyAccept: false });
    expect(screen.getByRole("alert")).toHaveTextContent("isn't an image");
  });

  it("uploads, sends title and idea, and shows the result and prompt", async () => {
    const fetchMock = mockFetch(200, RESULT);
    render(<App />);
    await userEvent.upload(screen.getByTestId("file-input"), png());
    expect(screen.getByAltText("Selected product")).toBeInTheDocument();
    await userEvent.type(screen.getByPlaceholderText(/Running Shoes/), "Turtle Men Red Shirt");
    await userEvent.type(screen.getByPlaceholderText(/olive green/), "olive green, linen");
    await userEvent.click(screen.getByRole("button", { name: "Analyse product" }));

    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("title")).toBe("Turtle Men Red Shirt");
    expect(body.get("variant")).toBe("olive green, linen");

    expect(await screen.findByText("Needs a human check")).toBeInTheDocument();
    expect(screen.getByText(/text says 'red'/)).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Product type confidence" })).toHaveAttribute("aria-valuenow", "93");
    expect(screen.getByText(/men's shirts, olive green, linen/)).toBeInTheDocument();
    expect(screen.queryByText(/Kurtas/)).not.toBeInTheDocument(); // under 5%: hidden
  });

  it("shows the server's error message", async () => {
    mockFetch(422, { detail: "That file isn't an image we can read. Try a JPG or PNG." });
    render(<App />);
    await userEvent.upload(screen.getByTestId("file-input"), png());
    await userEvent.click(screen.getByRole("button", { name: "Analyse product" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("isn't an image we can read");
  });

  it("explains when the server can't be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<App />);
    await userEvent.upload(screen.getByTestId("file-input"), png());
    await userEvent.click(screen.getByRole("button", { name: "Analyse product" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't reach the analysis server");
  });

  it("hides the prompt card when no design idea was given", async () => {
    mockFetch(200, { ...RESULT, prompt: null, conflicts: [], needs_review: false });
    render(<App />);
    await userEvent.upload(screen.getByTestId("file-input"), png());
    await userEvent.click(screen.getByRole("button", { name: "Analyse product" }));
    expect(await screen.findByText("Looks consistent")).toBeInTheDocument();
    expect(screen.queryByText(/Negative prompt/)).not.toBeInTheDocument();
  });
});
