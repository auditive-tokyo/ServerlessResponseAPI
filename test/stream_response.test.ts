import { describe, it, expect, beforeAll, vi, beforeEach, afterEach } from "vitest";
import {
  generateStreamResponse,
  StreamYield,
} from "../lambda_function/stream_response";
import { ResponseCompletedEvent } from "openai/resources/responses/responses";

/**
 * 型ガード: ResponseCompletedEventかどうか
 */
function isCompletedEvent(chunk: StreamYield): chunk is ResponseCompletedEvent {
  return (
    typeof chunk === "object" &&
    chunk !== null &&
    chunk.type === "response.completed"
  );
}

/**
 * 型ガード: エラーレスポンスかどうか
 */
function isErrorResponse(chunk: StreamYield): boolean {
  if (typeof chunk !== "string") return false;
  try {
    const parsed = JSON.parse(chunk.replace("data: ", "").trim());
    return "error" in parsed;
  } catch {
    return false;
  }
}

describe("stream_response", () => {
  // 環境変数チェック
  const hasVectorStore = !!process.env.OPENAI_VECTOR_STORE_ID;
  const hasApiKey = !!process.env.OPENAI_API_KEY;

  beforeAll(() => {
    if (!hasApiKey) {
      console.warn("OPENAI_API_KEY is not set. Skipping integration tests.");
    }
  });

  describe("generateStreamResponse", () => {
    it("should generate stream response with text output", async () => {
      if (!hasApiKey) {
        return;
      }

      const chunks: StreamYield[] = [];
      let hasTextDelta = false;
      let hasCompleted = false;
      let responseId: string | undefined;

      for await (const chunk of generateStreamResponse({
        userMessage: "Hello, this is a test. Please respond briefly.",
        model: "gpt-5-mini",
        previousResponseId: null,
      })) {
        chunks.push(chunk);

        if (
          typeof chunk !== "string" &&
          chunk.type === "response.output_text.delta"
        ) {
          hasTextDelta = true;
        }
        if (typeof chunk !== "string" && chunk.type === "response.completed") {
          hasCompleted = true;
          if (isCompletedEvent(chunk)) {
            responseId = chunk.response?.id;
          }
        }
      }

      expect(chunks.length).toBeGreaterThan(0);
      expect(hasTextDelta).toBe(true);
      expect(hasCompleted).toBe(true);
      expect(responseId).toBeDefined();
      expect(responseId).toMatch(/^resp_/);
    }, 30000); // 30秒タイムアウト

    it("should perform file search when vector store is configured", async () => {
      if (!hasApiKey || !hasVectorStore) {
        console.log(
          "Skipping file search test: missing API key or vector store",
        );
        return;
      }

      const events: string[] = [];

      for await (const chunk of generateStreamResponse({
        userMessage: "Search for any file and summarize it briefly.",
        model: "gpt-5-mini",
        previousResponseId: null,
      })) {
        if (typeof chunk !== "string" && chunk.type) {
          events.push(chunk.type);
        }
      }

      // File search関連のイベントが発生していることを確認
      expect(events).toContain("response.file_search_call.in_progress");
      expect(events).toContain("response.completed");
    }, 30000);

    it("should support previous_response_id for conversation continuity", async () => {
      if (!hasApiKey) {
        return;
      }

      // 最初のレスポンスを取得
      let firstResponseId: string | undefined;

      for await (const chunk of generateStreamResponse({
        userMessage: "Remember this number: 42",
        model: "gpt-5-mini",
        previousResponseId: null,
      })) {
        if (typeof chunk !== "string" && chunk.type === "response.completed") {
          if (isCompletedEvent(chunk)) {
            firstResponseId = chunk.response?.id;
          }
        }
      }

      expect(firstResponseId).toBeDefined();

      // 前のレスポンスIDを使って続きの会話
      let hasResponse = false;
      for await (const chunk of generateStreamResponse({
        userMessage: "What number did I ask you to remember?",
        model: "gpt-5-mini",
        previousResponseId: firstResponseId!,
      })) {
        if (typeof chunk !== "string" && chunk.type === "response.completed") {
          hasResponse = true;
        }
      }

      expect(hasResponse).toBe(true);
    }, 60000); // 2回のAPI呼び出しがあるので60秒

    it("should handle API errors gracefully", async () => {
      if (!hasApiKey) {
        return;
      }

      const chunks: StreamYield[] = [];
      let hasError = false;

      for await (const chunk of generateStreamResponse({
        userMessage: "Test error handling",
        model: "invalid-model-name-that-does-not-exist",
        previousResponseId: null,
      })) {
        chunks.push(chunk);
        if (isErrorResponse(chunk)) {
          hasError = true;
        }
      }

      expect(hasError).toBe(true);
    }, 30000);
  });

  describe("generateStreamResponse without Vector Store", () => {
    const originalVectorStoreId = process.env.OPENAI_VECTOR_STORE_ID;
    const hasApiKey = !!process.env.OPENAI_API_KEY;

    beforeEach(() => {
      // Vector Store IDを一時的に削除
      delete process.env.OPENAI_VECTOR_STORE_ID;
    });

    afterEach(() => {
      // 元に戻す
      if (originalVectorStoreId) {
        process.env.OPENAI_VECTOR_STORE_ID = originalVectorStoreId;
      }
    });

    it("should work without Vector Store ID and log warning", async () => {
      if (!hasApiKey) {
        return;
      }

      // console.warnをスパイ
      const warnSpy = vi.spyOn(console, "warn");

      // モジュールを再読み込み（環境変数の変更を反映）
      vi.resetModules();
      const { generateStreamResponse: freshGenerate } = await import(
        "../lambda_function/stream_response"
      );

      const chunks: StreamYield[] = [];
      let hasCompleted = false;

      for await (const chunk of freshGenerate({
        userMessage: "Say hello briefly without file search.",
        model: "gpt-5-mini",
        previousResponseId: null,
      })) {
        chunks.push(chunk);
        if (typeof chunk !== "string" && chunk.type === "response.completed") {
          hasCompleted = true;
        }
      }

      expect(chunks.length).toBeGreaterThan(0);
      expect(hasCompleted).toBe(true);
      expect(warnSpy).toHaveBeenCalledWith(
        "No Vector Store ID provided, skipping file search tool",
      );

      warnSpy.mockRestore();
    }, 30000);
  });
});
