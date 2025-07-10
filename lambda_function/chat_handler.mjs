import { createFilterKeys } from './create_filter_key.js';
import { generateStreamResponse } from './stream_response.js';

const MODEL = "gpt-4.1-mini";
const VECTOR_SEARCH_FILTER_KEY = process.env.VECTOR_SEARCH_FILTER_KEY;

export const handler = awslambda.streamifyResponse(
    async (event, responseStream, context) => {
        try {
            let body;
            try {
                body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body;
            } catch (e) {
                console.error('Request body parse error:', e);
                responseStream.write(JSON.stringify({ error: 'Request body is required' }));
                responseStream.end();
                return;
            }

            const userMessage = body.message || '';
            const previousResponseId = body.previous_response_id || null;
            const filterKeys = Array.isArray(body.filter_keys) ? body.filter_keys : [];

            if (!userMessage) {
                console.warn('Message is required but not provided');
                responseStream.write(JSON.stringify({ error: 'Message is required' }));
                responseStream.end();
                return;
            }

            let filters = null;
            if (VECTOR_SEARCH_FILTER_KEY) {
                filters = createFilterKeys(filterKeys, "eq", VECTOR_SEARCH_FILTER_KEY);
            }

            for await (const chunk of generateStreamResponse({
                userMessage,
                model: MODEL,
                previousResponseId,
                filters
            })) {
                responseStream.write(chunk);
            }
            responseStream.end();
            console.info('Full response text:', responseText);
        } catch (e) {
            console.error('Error in handler:', e);
            responseStream.write(JSON.stringify({ error: String(e) }));
            responseStream.end();
        }
    }
);