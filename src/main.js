/**
 * Amplifier Browser PoC - Main Entry Point
 * 
 * This demonstrates running Amplifier + WebGPU LLM entirely in the browser:
 * - Pyodide: Python runtime in WebAssembly
 * - WebLLM: WebGPU-accelerated LLM inference
 * - No server required!
 */

import { CreateMLCEngine } from '@mlc-ai/web-llm';

// Model configurations
const MODELS = {
  // High quality model - requires shader-f16 (works on Mac/desktop with good GPU)
  phi: {
    id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
    name: 'Phi-3.5 Mini',
    requiresF16: true,
    vram: 2500,  // ~2.5GB VRAM
  },
  // Fallback model - no shader-f16 needed (works in headless Chrome, CI, older GPUs)
  llama1b: {
    id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    name: 'Llama-3.2 1B',
    requiresF16: false,
    vram: 1200,  // ~1.2GB VRAM
  },
  // Better fallback - no shader-f16 needed, better quality than 1B
  llama3b: {
    id: 'Llama-3.2-3B-Instruct-q4f32_1-MLC',
    name: 'Llama-3.2 3B',
    requiresF16: false,
    vram: 3000,  // ~3GB VRAM
  },
};

// Configuration
const CONFIG = {
  // Model selection: 'auto', 'phi', 'llama1b', 'llama3b'
  // 'auto' will detect shader-f16 support and choose best available
  modelSelection: 'auto',
  
  // Pyodide CDN
  pyodideUrl: 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/',
  
  // Python packages to install (from PyPI)
  pythonPackages: ['pydantic', 'pyyaml', 'typing-extensions'],
  
  // Local wheel for amplifier-core (served from /wheels/)
  amplifierCoreWheel: '/wheels/amplifier_core-1.0.0-py3-none-any.whl',
};

/**
 * Detect WebGPU capabilities and select the best model.
 * @returns {Object} Selected model config
 */
async function selectModel() {
  // Check for explicit model selection (not auto)
  if (CONFIG.modelSelection !== 'auto' && MODELS[CONFIG.modelSelection]) {
    console.log(`Using explicitly configured model: ${CONFIG.modelSelection}`);
    return MODELS[CONFIG.modelSelection];
  }
  
  // Auto-detect: check for shader-f16 support
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      const hasF16 = adapter.features.has('shader-f16');
      console.log(`WebGPU shader-f16 support: ${hasF16}`);
      
      if (hasF16) {
        console.log('Using Phi-3.5 (f16 supported)');
        return MODELS.phi;
      }
    }
  } catch (e) {
    console.warn('Error detecting WebGPU features:', e);
  }
  
  // Fallback to f32 model
  console.log('Using Llama-3.2-1B (f32 fallback for compatibility)');
  return MODELS.llama1b;
}

// Global state
let pyodide = null;
let llmEngine = null;
let isGenerating = false;
let selectedModel = null;  // Will be set by selectModel()

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
  
  // Select the best model based on GPU capabilities
  selectedModel = await selectModel();
  updateStep('step-model', 'loading', `Downloading ${selectedModel.name}...`);
  
  // Create the engine with the selected model
  llmEngine = await CreateMLCEngine(selectedModel.id, {
    initProgressCallback,
  });
  
  updateStep('step-model', 'done', selectedModel.name);
  updateStatusBadge('webllm-status', 'ready', 'WebLLM: Ready');
  
  return llmEngine;
}

// Initialize the Amplifier Python session
async function initAmplifierSession() {
  updateStep('step-amplifier', 'loading');
  
  // Load the Python adapter code (amplifier-core browser shim)
  const pythonFile = '/src/python/amplifier_browser_shim.py';
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
  
  // Initialize the session in Python (using amplifier-core browser shim)
  await pyodide.runPythonAsync(`
session = create_session(model_id="${selectedModel.id}")
await session.initialize()
print("Amplifier Browser Session (with amplifier-core) initialized!")
`);
  
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
    let firstChunk = true;
    
    // Track timing
    const startTime = performance.now();
    
    // Set up streaming callback that Python can call
    // This gets registered globally so js_llm_stream can use it
    window._streamCallback = (chunk) => {
      // Remove typing indicator on first chunk
      if (firstChunk) {
        firstChunk = false;
        typingEl.remove();
        elements.messages.appendChild(assistantMessageEl);
      }
      
      fullResponse += chunk;
      contentEl.textContent = fullResponse;
      
      // Scroll to bottom
      elements.messages.scrollTop = elements.messages.scrollHeight;
    };
    
    // Execute through Python session with streaming
    const resultJson = await pyodide.runPythonAsync(`
import json
from pyodide.ffi import create_proxy
from js import window

async def run_completion():
    messages_json = json.dumps([
        {"role": "system", "content": session.system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in session.history],
        {"role": "user", "content": ${JSON.stringify(userMessage)}}
    ])
    
    # Add to history
    session.history.append({"role": "user", "content": ${JSON.stringify(userMessage)}})
    
    # Create proxy for streaming callback
    stream_callback = create_proxy(window._streamCallback)
    
    # Call JavaScript LLM with streaming
    result_json = await js_llm_stream(messages_json, stream_callback)
    result = json.loads(result_json)
    
    # Add response to history
    session.history.append({"role": "assistant", "content": result["content"]})
    
    return result_json

await run_completion()
    `);
    
    // Parse result
    const result = JSON.parse(resultJson);
    
    // If streaming didn't populate (fallback), use final result
    if (!fullResponse && result.content) {
      typingEl.remove();
      elements.messages.appendChild(assistantMessageEl);
      contentEl.textContent = result.content;
      fullResponse = result.content;
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
    
    // Clean up
    delete window._streamCallback;
    
  } catch (error) {
    console.error('Error:', error);
    typingEl.remove();
    addMessage('assistant', `Error: ${error.message}`);
    delete window._streamCallback;
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
    elements.statModel.textContent = selectedModel.name;
    elements.modelInfo.textContent = `Model: ${selectedModel.name} | Running locally via WebGPU`;
    
    // Add welcome message
    addMessage('assistant', 
      `Hello! I'm running entirely in your browser using WebGPU - no server required! ` +
      `I'm powered by ${selectedModel.name} and can help you with questions, writing, coding, and more. ` +
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
