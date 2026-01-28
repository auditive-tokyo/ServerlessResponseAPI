import { generateStreamResponse, StreamYield } from "./stream_response";
import { LambdaFunctionURLEvent, Context, Handler } from "aws-lambda";
import {
  ResponseTextDeltaEvent,
  ResponseTextDoneEvent,
  ResponseCompletedEvent,
} from "openai/resources/responses/responses";

const DEBUG = "true"; // TODO: デバッグモードは環境変数にするか？めんどいよね〜
const MODEL = "gpt-5-mini";

interface RequestBody {
  message?: string;
  previous_response_id?: string;
}

/**
 * AWS Lambda Response Streamのインターフェース
 */
interface ResponseStream {
  write(chunk: string): void;
  end(): void;
}

/**
 * 型ガード: ResponseTextDeltaEventかどうか
 */
function isTextDeltaEvent(chunk: StreamYield): chunk is ResponseTextDeltaEvent {
  return (
    typeof chunk === "object" &&
    chunk !== null &&
    chunk.type === "response.output_text.delta"
  );
}

/**
 * 型ガード: ResponseTextDoneEventかどうか
 */
function isTextDoneEvent(chunk: StreamYield): chunk is ResponseTextDoneEvent {
  return (
    typeof chunk === "object" &&
    chunk !== null &&
    chunk.type === "response.output_text.done"
  );
}

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
 * streamifyResponseのハンドラー型
 */
type StreamifyHandler = (
  event: LambdaFunctionURLEvent,
  responseStream: ResponseStream,
  context: Context,
) => Promise<void>;

// AWS Lambdaランタイムが提供するグローバル変数に型を適用
declare const awslambda: {
  streamifyResponse(handler: StreamifyHandler): Handler;
};

/**
 * ストリーミングチャンクを処理してレスポンスに書き込む
 */
function writeChunkToStream(
  chunk: StreamYield,
  responseStream: ResponseStream,
): void {
  if (DEBUG === "true") {
    console.debug("OpenAI chunk:", chunk);
  }

  // 文字列の場合はそのまま書き込む
  if (typeof chunk === "string") {
    responseStream.write(chunk);
    return;
  }

  // イベントタイプに応じて処理
  if (isTextDeltaEvent(chunk)) {
    responseStream.write(JSON.stringify(chunk) + "\n");
  } else if (isTextDoneEvent(chunk)) {
    responseStream.write("\n" + JSON.stringify(chunk) + "\n");
  } else if (isCompletedEvent(chunk) && chunk.response?.id) {
    responseStream.write(
      "\n" + JSON.stringify({ responseId: chunk.response.id }) + "\n",
    );
  }
}

/**
 * エラーレスポンスを書き込んでストリームを終了する
 */
function writeErrorAndEnd(
  responseStream: ResponseStream,
  errorMessage: string,
): void {
  responseStream.write(JSON.stringify({ error: errorMessage }));
  responseStream.end();
}

export const handler = awslambda.streamifyResponse(
  async (
    event: LambdaFunctionURLEvent,
    responseStream: ResponseStream,
    _context: Context,
  ) => {
    try {
      if (!event.body) {
        writeErrorAndEnd(responseStream, "Request body is required");
        return;
      }

      const body: RequestBody = JSON.parse(event.body);
      const userMessage = body.message;
      const previousResponseId = body.previous_response_id;

      if (!userMessage) {
        writeErrorAndEnd(responseStream, "Message is required");
        return;
      }

      for await (const chunk of generateStreamResponse({
        userMessage,
        model: MODEL,
        previousResponseId,
      })) {
        writeChunkToStream(chunk, responseStream);
      }
      responseStream.end();
    } catch (e) {
      console.error("Error in handler:", e);
      writeErrorAndEnd(responseStream, String(e));
    }
  },
);
