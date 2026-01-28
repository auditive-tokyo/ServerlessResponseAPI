import OpenAI from "openai";
import {
  ResponseCreateParamsStreaming,
  ResponseStreamEvent,
} from "openai/resources/responses/responses";

const PROMPT_TEMPLATE_ID = process.env.PROMPT_TEMPLATE_ID;
const OPENAI_VECTOR_STORE_ID = process.env.OPENAI_VECTOR_STORE_ID;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

/**
 * generateStreamResponseの戻り値型（OpenAIイベントまたは文字列メッセージ）
 */
export type StreamYield = ResponseStreamEvent | string;

export async function* generateStreamResponse({
  userMessage,
  model,
  previousResponseId = null,
}: {
  userMessage: string;
  model: string;
  previousResponseId?: string | null;
}): AsyncGenerator<StreamYield> {
  try {
    console.info(
      `Response API File Search開始: '${userMessage}' (Vector Store: ${OPENAI_VECTOR_STORE_ID})`,
    );

    const tools: ResponseCreateParamsStreaming["tools"] = OPENAI_VECTOR_STORE_ID
      ? [
          {
            type: "file_search",
            vector_store_ids: [OPENAI_VECTOR_STORE_ID],
            max_num_results: 10,
            ranking_options: { score_threshold: 0.2 },
          },
        ]
      : [];

    if (!OPENAI_VECTOR_STORE_ID) {
      console.warn("No Vector Store ID provided, skipping file search tool");
    }

    const requestPayload: ResponseCreateParamsStreaming = {
      model: model,
      input: userMessage,
      tools: tools,
      tool_choice: "auto",
      truncation: "auto",
      stream: true,
      reasoning: { effort: "low" },
      text: { verbosity: "low" },
      ...(PROMPT_TEMPLATE_ID && { prompt: { id: PROMPT_TEMPLATE_ID } }),
      ...(previousResponseId && { previous_response_id: previousResponseId }),
    };

    const response = await openai.responses.create(requestPayload);

    for await (const chunk of response as AsyncIterable<ResponseStreamEvent>) {
      yield chunk;
    }

    yield `data: ${JSON.stringify({ completed: true })}\n\n`;

    console.info("Response API stream completed successfully");
  } catch (e) {
    console.error(`Error in generateStreamResponse: ${e}`);
    yield `data: ${JSON.stringify({ error: String(e) })}\n\n`;
  }
}
