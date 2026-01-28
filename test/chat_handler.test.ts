import { describe, it, expect, vi, beforeAll } from "vitest";
import { LambdaFunctionURLEvent, Context } from "aws-lambda";

// ResponseStream型
interface ResponseStream {
  write(chunk: string): void;
  end(): void;
}

// awslambdaグローバルをモック（handlerの内部関数をキャプチャ）
let capturedHandler: (
  event: LambdaFunctionURLEvent,
  responseStream: ResponseStream,
  context: Context,
) => Promise<void>;

// awslambdaはLambdaランタイムでのみ存在するためモック
(globalThis as Record<string, unknown>).awslambda = {
  streamifyResponse: (
    fn: (
      event: LambdaFunctionURLEvent,
      responseStream: ResponseStream,
      context: Context,
    ) => Promise<void>,
  ) => {
    capturedHandler = fn;
    return fn;
  },
};

describe("chat_handler", () => {
  const hasApiKey = !!process.env.OPENAI_API_KEY;

  beforeAll(async () => {
    // モック設定後にモジュールをインポート
    await import("../lambda_function/chat_handler");
  });

  describe("Integration tests", () => {
    it("should process a valid request and stream response", async () => {
      if (!hasApiKey) {
        console.log("Skipping: OPENAI_API_KEY not set");
        return;
      }

      const chunks: string[] = [];
      const mockResponseStream: ResponseStream = {
        write: vi.fn((chunk: string) => chunks.push(chunk)),
        end: vi.fn(),
      };

      const event: LambdaFunctionURLEvent = {
        body: JSON.stringify({ message: "Say hello briefly." }),
        headers: {},
        isBase64Encoded: false,
        rawPath: "/",
        rawQueryString: "",
        requestContext: {} as LambdaFunctionURLEvent["requestContext"],
        routeKey: "",
        version: "2.0",
      };

      await capturedHandler(event, mockResponseStream, {} as Context);

      expect(mockResponseStream.write).toHaveBeenCalled();
      expect(mockResponseStream.end).toHaveBeenCalled();
      expect(chunks.length).toBeGreaterThan(0);

      // レスポンスにテキストデルタが含まれていることを確認
      const hasTextDelta = chunks.some((c) =>
        c.includes("response.output_text.delta"),
      );
      expect(hasTextDelta).toBe(true);

      // responseIdが返されていることを確認
      const hasResponseId = chunks.some((c) => c.includes("responseId"));
      expect(hasResponseId).toBe(true);
    }, 30000);

    it("should return error when body is missing", async () => {
      const chunks: string[] = [];
      const mockResponseStream: ResponseStream = {
        write: vi.fn((chunk: string) => chunks.push(chunk)),
        end: vi.fn(),
      };

      const event: LambdaFunctionURLEvent = {
        body: undefined,
        headers: {},
        isBase64Encoded: false,
        rawPath: "/",
        rawQueryString: "",
        requestContext: {} as LambdaFunctionURLEvent["requestContext"],
        routeKey: "",
        version: "2.0",
      };

      await capturedHandler(event, mockResponseStream, {} as Context);

      expect(mockResponseStream.write).toHaveBeenCalledWith(
        JSON.stringify({ error: "Request body is required" }),
      );
      expect(mockResponseStream.end).toHaveBeenCalled();
    });

    it("should return error when message is missing", async () => {
      const chunks: string[] = [];
      const mockResponseStream: ResponseStream = {
        write: vi.fn((chunk: string) => chunks.push(chunk)),
        end: vi.fn(),
      };

      const event: LambdaFunctionURLEvent = {
        body: JSON.stringify({ previous_response_id: "resp_123" }),
        headers: {},
        isBase64Encoded: false,
        rawPath: "/",
        rawQueryString: "",
        requestContext: {} as LambdaFunctionURLEvent["requestContext"],
        routeKey: "",
        version: "2.0",
      };

      await capturedHandler(event, mockResponseStream, {} as Context);

      expect(mockResponseStream.write).toHaveBeenCalledWith(
        JSON.stringify({ error: "Message is required" }),
      );
      expect(mockResponseStream.end).toHaveBeenCalled();
    });

    it("should handle invalid JSON body", async () => {
      const chunks: string[] = [];
      const mockResponseStream: ResponseStream = {
        write: vi.fn((chunk: string) => chunks.push(chunk)),
        end: vi.fn(),
      };

      const event: LambdaFunctionURLEvent = {
        body: "invalid json",
        headers: {},
        isBase64Encoded: false,
        rawPath: "/",
        rawQueryString: "",
        requestContext: {} as LambdaFunctionURLEvent["requestContext"],
        routeKey: "",
        version: "2.0",
      };

      await capturedHandler(event, mockResponseStream, {} as Context);

      expect(mockResponseStream.write).toHaveBeenCalled();
      expect(mockResponseStream.end).toHaveBeenCalled();
      // エラーが返されていることを確認
      const hasError = chunks.some((c) => c.includes("error"));
      expect(hasError).toBe(true);
    });

    it("should support previous_response_id", async () => {
      if (!hasApiKey) {
        console.log("Skipping: OPENAI_API_KEY not set");
        return;
      }

      const chunks: string[] = [];
      const mockResponseStream: ResponseStream = {
        write: vi.fn((chunk: string) => chunks.push(chunk)),
        end: vi.fn(),
      };

      const event: LambdaFunctionURLEvent = {
        body: JSON.stringify({
          message: "Say hi.",
          previous_response_id: null,
        }),
        headers: {},
        isBase64Encoded: false,
        rawPath: "/",
        rawQueryString: "",
        requestContext: {} as LambdaFunctionURLEvent["requestContext"],
        routeKey: "",
        version: "2.0",
      };

      await capturedHandler(event, mockResponseStream, {} as Context);

      expect(mockResponseStream.end).toHaveBeenCalled();
      expect(chunks.length).toBeGreaterThan(0);
    }, 30000);
  });
});
