# Jimothy landing page

A centered landing page with a live Jimothy/Jev email comparison, CLI and SDK examples, and classification use cases.

Run it locally:

```sh
# Set AI_GATEWAY_API_KEY in your shell environment, then:
npm run demo:email
```

Open **http://127.0.0.1:4320**. Fill in the sender, subject, and body, or choose a clickable example. **Compare** runs both models; **Local only** makes no teacher request, and **Jev only** calls the API. Choosing an example does not submit it. Editing the email clears old results and cancels an outstanding Jev request.

The local model runs as q8 WebAssembly in a browser worker, using the same canonical `{body,from,subject}` JSON as training. Its displayed time excludes model loading. Jev's time includes the request through the local server and the external API. Both show all six category probabilities; JSON responses are available below the results.

Jev uses the question saved in the trained model. The shared API key stays in the server process and is not sent to the browser. It expires at the end of September 25, 2026 PST (September 26 at 08:00 UTC), enforced by the server. After expiry, or when no shared key is configured, visitors can enter their own Vercel API key. Visitor keys stay in page memory and are sent only with Jev requests; they are not saved. Clicking **Jev only** or **Compare** sends that email to Vercel AI Gateway/TypeSafe; the example does not save emails or responses. The server binds only to `127.0.0.1`, accepts same-origin API requests, and limits Jev to one request at a time. API failures leave the local result visible. Restart the server after changing its environment.

Without an API key, local classification still works. The default model is `models/email-300/minilm`; see the [email experiment](../../docs/email-classifier.md) if you need to create it. The SDK requires a format-v3 bundle; retrain older models into a new directory and set `EMAIL_MODEL_DIR` to that path. Optional environment settings:

| Variable | Default |
| --- | --- |
| `EMAIL_DEMO_PORT` | `4320` |
| `EMAIL_MODEL_DIR` | `models/email-300/minilm` |
| `EMAIL_TEACHER_MODEL` | `typesafe-ai/jev` |
| `EMAIL_TEACHER_URL` | `https://ai-gateway.vercel.sh/typesafe/v1/systemone` |

The local model accepts up to 256 wordpieces, including field names and special tokens. The training set was too small to recommend a threshold, so its recommendation is `null`. Both models display predictions without an acceptance policy. Its probabilities are uncalibrated. Browser results can differ from Node or batched inference; no email-specific WebGPU calibration has been performed, so this example uses the existing q8 model. A high probability is not a correctness guarantee, and social was not covered by the original test set.

This local example is separate from the banking playground on port 4319. It is not a public, authenticated hosting service.

Run the proxy and preprocessing checks with `npm run test:email-demo`.

Local inference uses the public [`jimothy/browser` SDK](../../docs/browser-sdk.md). The example server serves its worker and runtime assets; it holds credentials only for the optional Jev comparison.
