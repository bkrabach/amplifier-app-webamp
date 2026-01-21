/**
 * Amplifier Browser PoC - Main Entry Point
 * 
 * This demonstrates running Amplifier + WebGPU LLM entirely in the browser:
 * - Pyodide: Python runtime in WebAssembly
 * - WebLLM: WebGPU-accelerated LLM inference
 * - No server required!
 */

import { CreateMLCEngine } from '@mlc-ai/web-llm';

// Configuration
const CONFIG = {
  // Model options:
  // - 'Phi-3.5-mini-instruct-q4f16_1-MLC' - Best quality, requires shader-f16 (works on Mac/desktop)
  // - 'Llama-3.2-1B-Instruct-q4f32_1-MLC' - No shader-f16 needed (works in headless Chrome)
  // - 'Llama-3.2-3B-Instruct-q4f32_1-MLC' - Better quality, no shader-f16 needed
  modelId: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',  // Using f32 model for broader compatibility
  
  // Pyodide CDN
  pyodideUrl: 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/',
  
  // Python packages to install (from PyPI)
  pythonPackages: ['pydantic', 'pyyaml', 'typing-extensions'],
  
  // Local wheel for amplifier-core (served from /wheels/)
  amplifierCoreWheel: '/wheels/amplifier_core-1.0.0-py3-none-any.whl',
  
  // Use real amplifier-core (set to false to use simplified version)
  useAmplifierCore: true,
};

// Global state
let pyodide = null;
let llmEngine = null;
let isGenerating = false;

// UI Elements
const elements = {
  loadingScreen: document.getElementById('loading-screen'),
  chatContainer: document.getElementById('chat-container'),
  messages: document.getElementById('messages'),
  userInput: document.getElementById('user-input'),
  sendBtn: document.getElementById('send-btn'),
  pyodideStatus: document.getElementById('pyodide-status'),
  webllmStatus: document.getElementById('webllm-status'),
  statModel: document.getElementById('stat-model'),
  statSpeed: document.getElementById('stat-speed'),
  statMemory: document.getElementById('stat-memory'),
  modelInfo: document.getElementById('model-info'),
};

// Loading step management
function updateStep(stepId, status, detail = '') {
  const step = document.getElementById(stepId);
  const icon = step.querySelector('.icon');
  const detailEl = step.querySelector('.detail');
  
  step.classList.remove('active', 'done', 'error');
  
  if (status === 'loading') {
    step.classList.add('active');
    icon.innerHTML = '<span class="spinner"></span>';
  } else if (status === 'done') {
    step.classList.add('done');
    icon.innerHTML = '<span class="check">✓</span>';
  } else if (status === 'error') {
    step.classList.add('error');
    icon.innerHTML = '<span style="color: #f87171;">✗</span>';
  }
  
  if (detailEl && detail) {
    detailEl.textContent = detail;
  }
}

function updateStatusBadge(elementId, status, text) {
  const el = document.getElementById(elementId);
  el.textContent = text;
  el.className = `status-badge ${status}`;
}

// Check WebGPU support
async function checkWebGPU() {
  updateStep('step-webgpu', 'loading');
  
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
  }
  
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No WebGPU adapter found. Your GPU may not be supported.');
  }
  
  const device = await adapter.requestDevice();
  console.log('WebGPU device:', device);
  
  updateStep('step-webgpu', 'done');
  return true;
}

// Load Pyodide
async function loadPyodide() {
  updateStep('step-pyodide', 'loading', 'Downloading runtime...');
  updateStatusBadge('pyodide-status', 'loading', 'Pyodide: Loading...');
  
  // Load Pyodide from CDN
  const script = document.createElement('script');
  script.src = `${CONFIG.pyodideUrl}pyodide.js`;
  document.head.appendChild(script);
  
  await new Promise((resolve, reject) => {
    script.onload = resolve;
    script.onerror = () => reject(new Error('Failed to load Pyodide script'));
  });
  
  updateStep('step-pyodide', 'loading', 'Initializing Python...');
  
  // Initialize Pyodide
  pyodide = await window.loadPyodide({
    indexURL: CONFIG.pyodideUrl,
  });
  
  updateStep('step-pyodide', 'loading', 'Installing packages...');
  
  // Install required packages
  await pyodide.loadPackage('micropip');
  const micropip = pyodide.pyimport('micropip');
  
  for (const pkg of CONFIG.pythonPackages) {
    updateStep('step-pyodide', 'loading', `Installing ${pkg}...`);
    await micropip.install(pkg);
  }
  
  // Install amplifier-core from local wheel if enabled
  if (CONFIG.useAmplifierCore) {
    updateStep('step-pyodide', 'loading', 'Installing amplifier-core...');
    // micropip needs a full URL, not just a path
    const wheelUrl = new URL(CONFIG.amplifierCoreWheel, window.location.origin).href;
    console.log('Installing amplifier-core from:', wheelUrl);
    await micropip.install(wheelUrl);
  }
  
  updateStep('step-pyodide', 'done', `Python ${pyodide.version} ready`);
  updateStatusBadge('pyodide-status', 'ready', `Pyodide: ${pyodide.version}`);
  
  return pyodide;
}

// Load WebLLM engine
async function loadWebLLM() {
  updateStep('step-webllm', 'loading', 'Initializing engine...');
  updateStatusBadge('webllm-status', 'loading', 'WebLLM: Loading...');
  
  // Create callback for model loading progress
  const initProgressCallback = (progress) => {
    const { text, progress: pct } = progress;
    
    if (pct !== undefined) {
      const percentage = Math.round(pct * 100);
      updateStep('step-model', 'loading', `${percentage}% - ${text}`);
    } else {
      updateStep('step-model', 'loading', text);
    }
  };
  
  updateStep('step-webllm', 'done', 'Engine ready');
  updateStep('step-model', 'loading', 'Starting download...');
  
  // Create the engine with the specified model
  llmEngine = await CreateMLCEngine(CONFIG.modelId, {
    initProgressCallback,
  });
  
  updateStep('step-model', 'done', CONFIG.modelId);
  updateStatusBadge('webllm-status', 'ready', 'WebLLM: Ready');
  
  return llmEngine;
}

// Initialize the Amplifier Python session
async function initAmplifierSession() {
  updateStep('step-amplifier', 'loading');
  
  // Load the Python adapter code
  // Load the appropriate Python module based on config
  const pythonFile = CONFIG.useAmplifierCore 
    ? '/src/python/amplifier_browser_shim.py'  // Real amplifier-core integration
    : '/src/python/amplifier_browser.py';       // Simplified standalone version
  
  const response = await fetch(pythonFile);
  const pythonCode = await response.text();
  
  console.log(`Loading Python module: ${pythonFile}`);
  
  // Run the Python code
  await pyodide.runPythonAsync(pythonCode);
  
  // Set up the bridge between Python and JavaScript
  // This allows Python to call the LLM
  pyodide.globals.set('js_llm_complete', async (messagesJson) => {
    const messages = JSON.parse(messagesJson);
    
    const response = await llmEngine.chat.completions.create({
      messages,
      temperature: 0.7,
      max_tokens: 2048,
    });
    
    return JSON.stringify({
      content: response.choices[0].message.content,
      usage: response.usage,
    });
  });
  
  // Streaming version
  pyodide.globals.set('js_llm_stream', async (messagesJson, onChunk) => {
    const messages = JSON.parse(messagesJson);
    
    const stream = await llmEngine.chat.completions.create({
      messages,
      temperature: 0.7,
      max_tokens: 2048,
      stream: true,
    });
    
    let fullContent = '';
    let usage = null;
    
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || '';
      fullContent += delta;
      
      if (delta) {
        // Call Python callback with the chunk
        onChunk(delta);
      }
      
      if (chunk.usage) {
        usage = chunk.usage;
      }
    }
    
    return JSON.stringify({
      content: fullContent,
      usage: usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    });
  });
  
  // Initialize the session in Python
  const sessionInitCode = CONFIG.useAmplifierCore
    ? `
# Using real amplifier-core with browser shim
session = create_session(model_id="${CONFIG.modelId}")
await session.initialize()
print("Amplifier Browser Session (with amplifier-core) initialized!")
`
    : `
# Using simplified standalone version
session = AmplifierBrowserSession()
print("Amplifier Browser Session (standalone) initialized!")
`;
  
  await pyodide.runPythonAsync(sessionInitCode);
  
  updateStep('step-amplifier', 'done');
  
  return true;
}

// Send a message through the Amplifier session
async function sendMessage(userMessage) {
  if (isGenerating) return;
  isGenerating = true;
  
  // Add user message to UI
  addMessage('user', userMessage);
  
  // Clear input
  elements.userInput.value = '';
  elements.sendBtn.disabled = true;
  
  // Show typing indicator
  const typingEl = addTypingIndicator();
  
  try {
    // Create a message element for streaming
    const assistantMessageEl = document.createElement('div');
    assistantMessageEl.className = 'message assistant';
    assistantMessageEl.innerHTML = `
      <div class="avatar">A</div>
      <div class="content"></div>
    `;
    
    const contentEl = assistantMessageEl.querySelector('.content');
    let fullResponse = '';
    
    // Track timing
    const startTime = performance.now();
    let tokenCount = 0;
    
    // Set up streaming callback in Python
    await pyodide.runPythonAsync(`
import json
from js import document

response_chunks = []

def on_stream_chunk(chunk):
    response_chunks.append(chunk)
    # Update will happen from JS side
    `);
    
    // Create a JavaScript function that Python can call
    const onChunk = (chunk) => {
      // Remove typing indicator on first chunk
      if (typingEl.parentNode) {
        typingEl.remove();
        elements.messages.appendChild(assistantMessageEl);
      }
      
      fullResponse += chunk;
      contentEl.textContent = fullResponse;
      tokenCount++;
      
      // Scroll to bottom
      elements.messages.scrollTop = elements.messages.scrollHeight;
    };
    
    // Execute through Python session
    const resultJson = await pyodide.runPythonAsync(`
import json
import asyncio
from pyodide.ffi import create_proxy

async def run_completion():
    messages_json = json.dumps([
        {"role": "system", "content": session.system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in session.history],
        {"role": "user", "content": ${JSON.stringify(userMessage)}}
    ])
    
    # Add to history
    session.history.append({"role": "user", "content": ${JSON.stringify(userMessage)}})
    
    # Call JavaScript LLM
    result_json = await js_llm_stream(messages_json, create_proxy(lambda c: None))
    result = json.loads(result_json)
    
    # Add response to history
    session.history.append({"role": "assistant", "content": result["content"]})
    
    return result_json

await run_completion()
    `);
    
    // Parse result and do non-streaming fallback if needed
    const result = JSON.parse(resultJson);
    
    // If streaming didn't populate, use final result
    if (!fullResponse) {
      typingEl.remove();
      elements.messages.appendChild(assistantMessageEl);
      contentEl.textContent = result.content;
    }
    
    // Calculate stats
    const endTime = performance.now();
    const duration = (endTime - startTime) / 1000;
    const tokensPerSec = result.usage?.completion_tokens 
      ? (result.usage.completion_tokens / duration).toFixed(1)
      : '--';
    
    elements.statSpeed.textContent = `${tokensPerSec} tok/s`;
    
    // Update memory stat if available
    if (performance.memory) {
      const usedMB = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
      elements.statMemory.textContent = `${usedMB} MB`;
    }
    
  } catch (error) {
    console.error('Error:', error);
    typingEl.remove();
    addMessage('assistant', `Error: ${error.message}`);
  } finally {
    isGenerating = false;
    elements.sendBtn.disabled = false;
    elements.userInput.focus();
  }
}

// UI Helper functions
function addMessage(role, content) {
  const messageEl = document.createElement('div');
  messageEl.className = `message ${role}`;
  
  const avatar = role === 'user' ? 'U' : 'A';
  
  messageEl.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="content">${escapeHtml(content)}</div>
  `;
  
  elements.messages.appendChild(messageEl);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  
  return messageEl;
}

function addTypingIndicator() {
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="avatar">A</div>
    <div class="content">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  elements.messages.appendChild(el);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return el;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showError(message) {
  const errorEl = document.createElement('div');
  errorEl.className = 'error-banner';
  errorEl.textContent = message;
  elements.loadingScreen.prepend(errorEl);
}

// Event listeners
function setupEventListeners() {
  elements.sendBtn.addEventListener('click', () => {
    const message = elements.userInput.value.trim();
    if (message) {
      sendMessage(message);
    }
  });
  
  elements.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const message = elements.userInput.value.trim();
      if (message) {
        sendMessage(message);
      }
    }
  });
  
  // Auto-resize textarea
  elements.userInput.addEventListener('input', () => {
    elements.userInput.style.height = 'auto';
    elements.userInput.style.height = elements.userInput.scrollHeight + 'px';
  });
}

// Main initialization
async function init() {
  try {
    // Check WebGPU first
    await checkWebGPU();
    
    // Load Pyodide and WebLLM in parallel
    const [py, llm] = await Promise.all([
      loadPyodide(),
      loadWebLLM(),
    ]);
    
    // Initialize Amplifier session
    await initAmplifierSession();
    
    // Show chat interface
    elements.loadingScreen.style.display = 'none';
    elements.chatContainer.classList.add('visible');
    elements.userInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.userInput.focus();
    
    // Update stats
    elements.statModel.textContent = CONFIG.modelId.split('-').slice(0, 3).join('-');
    elements.modelInfo.textContent = `Model: ${CONFIG.modelId} | Running locally via WebGPU`;
    
    // Add welcome message
    addMessage('assistant', 
      `Hello! I'm running entirely in your browser using WebGPU - no server required! ` +
      `I'm powered by ${CONFIG.modelId.split('-')[0]} and can help you with questions, writing, coding, and more. ` +
      `What would you like to discuss?`
    );
    
    console.log('Amplifier Browser PoC initialized successfully!');
    
  } catch (error) {
    console.error('Initialization failed:', error);
    showError(`Initialization failed: ${error.message}`);
    updateStatusBadge('pyodide-status', 'error', 'Pyodide: Error');
    updateStatusBadge('webllm-status', 'error', 'WebLLM: Error');
  }
}

// Set up event listeners
setupEventListeners();

// Start initialization
init();
