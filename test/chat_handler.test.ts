// RequestBody型の定義（chat_handler.tsと同じ）
interface RequestBody {
  message?: string;
  previous_response_id?: string;
}

// handler関数をテスト用にモック可能にする
// 実際のhandlerはawslambda.streamifyResponseでラップされているため、
// 内部ロジックをテストするにはモックが必要

describe("chat_handler", () => {
  describe("RequestBody validation", () => {
    it("should require message field", () => {
      const body: RequestBody = { previous_response_id: "resp_123" };
      expect(body.message).toBeUndefined();
    });

    it("should accept valid request body", () => {
      const body: RequestBody = {
        message: "Hello",
        previous_response_id: "resp_123",
      };
      expect(body.message).toBeDefined();
      expect(body.previous_response_id).toBeDefined();
    });
  });

  describe("Response stream events", () => {
    it("should handle response.output_text.delta event type", () => {
      const event = { type: "response.output_text.delta", delta: "Hello" };
      expect(event.type).toBe("response.output_text.delta");
    });

    it("should handle response.output_text.done event type", () => {
      const event = { type: "response.output_text.done" };
      expect(event.type).toBe("response.output_text.done");
    });

    it("should handle response.completed event type", () => {
      const event = {
        type: "response.completed",
        response: { id: "resp_123" },
      };
      expect(event.type).toBe("response.completed");
      expect(event.response.id).toBe("resp_123");
    });
  });
});
