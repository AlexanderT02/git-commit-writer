import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GeminiProvider } from "../../src/llm/provider/GeminiProvider.js";

const providerConfig = {
  provider: "gemini" as const,
  reasoningModel: "gemini-reasoning-test",
  generationModel: "gemini-generation-test",
};

describe("GeminiProvider", () => {
  beforeEach(() => {
    vi.stubEnv("GEMINI_API_KEY", "test-key");
    vi.stubEnv("GOOGLE_API_KEY", "");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("uses the reasoning model and maps usage for complete calls", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          candidates: [{ content: { parts: [{ text: " result " }] } }],
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 4,
            totalTokenCount: 14,
          },
        }),
        { status: 200 },
      ),
    );

    const result = await new GeminiProvider(providerConfig).complete("prompt");

    expect(result.text).toBe("result");
    expect(result.usage).toEqual({
      inputTokens: 10,
      outputTokens: 4,
      totalTokens: 14,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("gemini-reasoning-test:generateContent"),
      expect.objectContaining({
        headers: expect.objectContaining({ "x-goog-api-key": "test-key" }),
      }),
    );
  });

  it("handles CRLF streaming events and a final event without a separator", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"candidates":[{"content":{"parts":[{"text":"hello "}]}}]}\r\n\r\n',
          ),
        );
        controller.enqueue(
          encoder.encode(
            'data: {"candidates":[{"content":{"parts":[{"text":"world"}]}}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":2,"totalTokenCount":4}}',
          ),
        );
        controller.close();
      },
    });

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(body, { status: 200 }),
    );
    const chunks: string[] = [];

    const result = await new GeminiProvider(providerConfig).stream(
      "prompt",
      (text) => chunks.push(text),
    );

    expect(chunks).toEqual(["hello ", "world"]);
    expect(result.text).toBe("hello world");
    expect(result.usage?.totalTokens).toBe(4);
  });

  it("accepts GOOGLE_API_KEY as a fallback", async () => {
    vi.stubEnv("GEMINI_API_KEY", "");
    vi.stubEnv("GOOGLE_API_KEY", "google-key");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ candidates: [] }), { status: 200 }),
    );

    await new GeminiProvider(providerConfig).complete("prompt");

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ "x-goog-api-key": "google-key" }),
      }),
    );
  });

  it("reports a missing API key before making a request", () => {
    vi.stubEnv("GEMINI_API_KEY", "");
    vi.stubEnv("GOOGLE_API_KEY", "");

    expect(() => new GeminiProvider(providerConfig)).toThrow(
      "missing GEMINI_API_KEY or GOOGLE_API_KEY",
    );
  });
});
