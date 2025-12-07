/**
 * Ollama Manager
 * Ensures Ollama is running before the Phi4 service starts
 */

const { exec, spawn } = require('child_process');

class OllamaManager {
  constructor() {
    this.ollamaProcess = null;
    // Extract base URL from PHI4_API_URL (remove /api/generate if present)
    const apiUrl = process.env.PHI4_API_URL || 'http://localhost:11434';
    this.ollamaUrl = apiUrl.replace(/\/api\/generate$/, '');
  }

  /**
   * Check if Ollama is running
   */
  async isRunning() {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second timeout (increased from 2s)
      });
      return response.ok;
    } catch (error) {
      // Log the error for debugging
      console.log(`   ℹ️  Ollama check failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Start Ollama if not running
   */
  async ensureRunning() {
    console.log('🔍 Checking if Ollama is running...');
    
    const running = await this.isRunning();
    
    if (running) {
      console.log('✅ Ollama is already running');
      return true;
    }

    console.log('⚠️  Ollama is not running, attempting to start...');
    
    try {
      // Try to start Ollama
      await this.startOllama();
      
      // Wait for Ollama to be ready (max 10 seconds)
      const maxWaitTime = 30000;
      const checkInterval = 500;
      let waited = 0;
      
      while (waited < maxWaitTime) {
        await new Promise(resolve => setTimeout(resolve, checkInterval));
        waited += checkInterval;
        
        if (await this.isRunning()) {
          console.log(`✅ Ollama started successfully (took ${waited}ms)`);
          return true;
        }
      }
      
      console.error('❌ Ollama failed to start within 30 seconds');
      return false;
    } catch (error) {
      console.error('❌ Failed to start Ollama:', error.message);
      return false;
    }
  }

  /**
   * Start Ollama process
   */
  async startOllama() {
    return new Promise((resolve, reject) => {
      // Check if ollama command exists
      exec('which ollama', (error, _stdout) => {
        if (error) {
          reject(new Error('Ollama is not installed. Please install it: https://ollama.ai'));
          return;
        }

        console.log('🚀 Starting Ollama...');
        
        // Start Ollama in the background
        this.ollamaProcess = spawn('ollama', ['serve'], {
          detached: true,
          stdio: 'ignore'
        });

        // Unref so parent can exit independently
        this.ollamaProcess.unref();

        resolve();
      });
    });
  }

  /**
   * Stop Ollama (if we started it)
   */
  async stop() {
    if (this.ollamaProcess) {
      console.log('🛑 Stopping Ollama...');
      this.ollamaProcess.kill();
      this.ollamaProcess = null;
    }
  }
}

module.exports = new OllamaManager();
