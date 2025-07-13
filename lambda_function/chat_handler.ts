import { createFilterKeys } from './create_filter_key';
import { generateStreamResponse } from './stream_response';
import { LambdaFunctionURLEvent, Context } from 'aws-lambda';

const MODEL = "gpt-4.1-mini";
const VECTOR_SEARCH_FILTER_KEY = process.env.VECTOR_SEARCH_FILTER_KEY;

interface RequestBody {
    message?: string;
    previous_response_id?: string;
    filter_keys?: string;
}

// AWS Lambdaランタイムが提供するグローバル変数に型を適用
declare const awslambda: {
    streamifyResponse(
        handler: (event: LambdaFunctionURLEvent, responseStream: any, context: Context) => Promise<void>
    ): any;
};

export const handler = awslambda.streamifyResponse(
    async (event: LambdaFunctionURLEvent, responseStream: any, context: Context) => {
        try {
            if (!event.body) {
                responseStream.write(JSON.stringify({ error: 'Request body is required' }));
                responseStream.end();
                return;
            }

            const body: RequestBody = JSON.parse(event.body);
            const userMessage = body.message;
            const previousResponseId = body.previous_response_id;
            const filterKeysString = body.filter_keys ?? "";

            if (!userMessage) {
                responseStream.write(JSON.stringify({ error: 'Message is required' }));
                responseStream.end();
                return;
            }

            let filters = null;
            if (VECTOR_SEARCH_FILTER_KEY) {
                filters = createFilterKeys([filterKeysString], "eq", VECTOR_SEARCH_FILTER_KEY);
            }

            for await (const chunk of generateStreamResponse({
                userMessage,
                model: MODEL,
                previousResponseId,
                filters
            })) {
                if (chunk?.type === "response.output_text.delta") {
                    responseStream.write(JSON.stringify(chunk) + "\n");
                }
                if (chunk?.type === "response.output_text.done" && chunk.text) {
                    responseStream.write("\n" + chunk.text + "\n");
                }
                if (chunk?.type === "response.completed" && chunk.response?.id) {
                    responseStream.write("\n" + JSON.stringify({ responseId: chunk.response.id }) + "\n");
                }
            }
            responseStream.end();
        } catch (e) {
            console.error('Error in handler:', e);
            responseStream.write(JSON.stringify({ error: String(e) }));
            responseStream.end();
        }
    }
);