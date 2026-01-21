// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * WebGPU Browser Tests
 * 
 * These tests run against the Vite dev server and verify:
 * 1. WebGPU is available
 * 2. Pyodide loads successfully
 * 3. Amplifier-core loads
 * 4. Chat works end-to-end
 */

test.describe('WebGPU Amplifier PoC', () => {
  // Increase timeout for model loading
  test.setTimeout(300000); // 5 minutes for model download

  test.beforeEach(async ({ page }) => {
    // Enable console logging
    page.on('console', msg => {
      console.log(`[Browser ${msg.type()}]: ${msg.text()}`);
    });
    
    page.on('pageerror', err => {
      console.error(`[Browser Error]: ${err.message}`);
    });
  });

  test('WebGPU is available', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    const hasWebGPU = await page.evaluate(() => {
      return !!navigator.gpu;
    });
    
    console.log('WebGPU available:', hasWebGPU);
    expect(hasWebGPU).toBe(true);
  });

  test('can request WebGPU adapter', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    const adapterInfo = await page.evaluate(async () => {
      if (!navigator.gpu) return null;
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;
      
      const info = await adapter.requestAdapterInfo();
      return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      };
    });
    
    console.log('WebGPU Adapter:', JSON.stringify(adapterInfo, null, 2));
    expect(adapterInfo).not.toBeNull();
  });

  test('loading screen appears', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Check that loading screen is visible
    const loadingScreen = page.locator('#loading-screen');
    await expect(loadingScreen).toBeVisible();
    
    // Check that WebGPU step starts
    const webgpuStep = page.locator('#step-webgpu');
    await expect(webgpuStep).toBeVisible();
  });

  test('Pyodide loads successfully', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Wait for Pyodide step to complete (up to 60s)
    const pyodideStep = page.locator('#step-pyodide');
    await expect(pyodideStep).toHaveClass(/done/, { timeout: 60000 });
    
    console.log('Pyodide loaded successfully');
  });

  test('full initialization completes', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Wait for chat container to be visible (means init completed)
    const chatContainer = page.locator('#chat-container');
    await expect(chatContainer).toBeVisible({ timeout: 300000 });
    
    // Check model info is shown (Llama for f32 compat, or Phi for f16)
    const modelInfo = page.locator('#model-info');
    await expect(modelInfo).toContainText('Running locally via WebGPU', { timeout: 5000 });
    
    console.log('Full initialization completed!');
  });

  test('can send a message and get response', async ({ page }) => {
    await page.goto('http://localhost:5173');
    
    // Wait for chat to be ready
    const chatContainer = page.locator('#chat-container');
    await expect(chatContainer).toBeVisible({ timeout: 300000 });
    
    // Type a message
    const input = page.locator('#user-input');
    await input.fill('Say hello in exactly 3 words.');
    
    // Send the message
    const sendBtn = page.locator('#send-btn');
    await sendBtn.click();
    
    // Wait for response (assistant message)
    const assistantMessage = page.locator('.message.assistant').first();
    await expect(assistantMessage).toBeVisible({ timeout: 60000 });
    
    // Get the response text
    const responseText = await assistantMessage.textContent();
    console.log('Got response:', responseText);
    
    expect(responseText).toBeTruthy();
    expect(responseText.length).toBeGreaterThan(0);
  });
});
