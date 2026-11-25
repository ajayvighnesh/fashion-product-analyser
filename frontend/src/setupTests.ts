import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());

// jsdom has no object URLs; the preview only needs a string
URL.createObjectURL = () => "blob:preview";
URL.revokeObjectURL = () => {};
