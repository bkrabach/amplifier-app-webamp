// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * Playwright config for WebGPU testing
 * 
 * Key: We need to launch Chrome with WebGPU flags enabled
 */
module.exports = defineConfig({
  testDir: './tests',
  
  // Run tests in parallel
  fullyParallel: false, // Sequential for GPU resource sharing
  
  // Fail fast
  forbidOnly: !!process.env.CI,
  
  // Retries
  retries: process.env.CI ? 1 : 0,
  
  // Workers
  workers: 1, // Single worker for GPU
  
  // Reporter
  reporter: [
    ['list'],
    ['html', { open: 'never' }],
  ],
  
  // Global timeout
  timeout: 300000, // 5 minutes
  
  // Expect timeout
  expect: {
    timeout: 30000,
  },

  use: {
    // Base URL
    baseURL: 'http://localhost:5173',
    
    // Trace on failure
    trace: 'on-first-retry',
    
    // Screenshot on failure
    screenshot: 'only-on-failure',
    
    // Video on failure
    video: 'on-first-retry',
  },

  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        
        // Launch args for WebGPU
        launchOptions: {
          args: [
            // Enable WebGPU
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,WebGPU',
            
            // Use GPU
            '--use-gl=angle',
            '--use-angle=vulkan',
            '--enable-gpu-rasterization',
            '--ignore-gpu-blocklist',
            
            // Disable sandbox for CI (if needed)
            '--no-sandbox',
            '--disable-setuid-sandbox',
            
            // Headless GPU support
            '--use-vulkan',
          ],
        },
      },
    },
  ],

  // Web server - start Vite before tests
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 30000,
  },
});
