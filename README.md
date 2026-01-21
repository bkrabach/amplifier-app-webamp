# Amplifier Browser PoC

Proof of concept demonstrating Amplifier + WebGPU LLM running entirely in the browser - no server required!

## What This Demonstrates

- **Pyodide**: Python 3.12 running in WebAssembly
- **WebLLM**: WebGPU-accelerated LLM inference (Phi-3.5-mini)
- **Amplifier Patterns**: Session management, provider abstraction, hooks
- **Zero Server**: Everything runs client-side

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

Then open http://localhost:5173 in Chrome or Edge (WebGPU required).

## Requirements

- **Browser**: Chrome 113+ or Edge 113+ (WebGPU support required)
- **GPU**: Any modern GPU with WebGPU drivers
- **RAM**: ~4GB free for model loading
- **Storage**: ~2.5GB for model download (cached after first load)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Your Web Application                │    │
│  │                  (index.html)                    │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Pyodide (Python in WASM)              │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │     amplifier_browser.py                │    │    │
│  │  │     - AmplifierBrowserSession           │    │    │
│  │  │     - WebGPUProvider                    │    │    │
│  │  │     - HookRegistry                      │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
│                         │ JS Bridge                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │              WebLLM (WebGPU)                    │    │
│  │         Phi-3.5-mini-instruct-q4f16            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
webgpu-provider/
├── index.html                 # Main HTML with chat UI
├── package.json               # npm dependencies
├── vite.config.js             # Vite dev server config
└── src/
    ├── main.js                # JS entry - loads Pyodide + WebLLM
    └── python/
        └── amplifier_browser.py  # Python Amplifier adapter
```

## How It Works

1. **Loading Phase**:
   - Check WebGPU support
   - Load Pyodide (Python runtime)
   - Load WebLLM engine
   - Download LLM model (~2.4GB, cached)
   - Initialize Amplifier session

2. **Chat Phase**:
   - User sends message
   - Python `AmplifierBrowserSession.execute()` is called
   - Python calls JS bridge `js_llm_complete()` or `js_llm_stream()`
   - WebLLM generates response via WebGPU
   - Response streams back through Python to UI

## Key Components

### WebGPUProvider (Python)

Implements the Amplifier `Provider` protocol, bridging to JavaScript:

```python
class WebGPUProvider:
    async def complete(self, messages: list[Message]) -> CompletionResult:
        # Calls JavaScript WebLLM via Pyodide bridge
        result_json = await js_llm_complete(messages_json)
        return CompletionResult(content=result["content"], ...)
```

### JS Bridge (JavaScript)

Exposes LLM functions to Python:

```javascript
pyodide.globals.set('js_llm_complete', async (messagesJson) => {
    const response = await llmEngine.chat.completions.create({
        messages: JSON.parse(messagesJson),
    });
    return JSON.stringify({ content: response.choices[0].message.content });
});
```

## Configuration

Edit `src/main.js` to change:

```javascript
const CONFIG = {
  // Model options:
  // - 'Phi-3.5-mini-instruct-q4f16_1-MLC' (default, 2.4GB)
  // - 'Llama-3.2-1B-Instruct-q4f16_1-MLC' (smaller, 1GB)
  // - 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC' (1.5GB)
  modelId: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
};
```

## Limitations

- **WebGPU Required**: Only works in Chrome/Edge 113+ (Firefox/Safari support limited)
- **First Load**: ~2.5GB model download on first visit
- **Memory**: Requires ~4GB RAM for model + runtime
- **Model Size**: Limited to models that fit in browser memory (~7B max)

## Next Steps

This PoC demonstrates feasibility. A full implementation would add:

- [ ] RAG support with Transformers.js embeddings
- [ ] IndexedDB persistence for sessions
- [ ] Document upload and chunking
- [ ] Multiple model support
- [ ] Tool calling (if model supports it)
- [ ] Proper error handling and recovery

## License

MIT
