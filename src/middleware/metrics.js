/**
 * Metrics Collection Middleware
 * Tracks request metrics and performance
 */

class MetricsCollector {
  constructor() {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      requestsByAction: {},
      requestsByParser: {},
      responseTimes: [],
      errors: []
    };
    
    this.startTime = Date.now();
  }

  recordRequest(action) {
    this.metrics.totalRequests++;
    
    if (!this.metrics.requestsByAction[action]) {
      this.metrics.requestsByAction[action] = 0;
    }
    this.metrics.requestsByAction[action]++;
  }

  recordSuccess(responseTime, parser = null) {
    this.metrics.successfulRequests++;
    this.metrics.responseTimes.push(responseTime);
    
    // Keep only last 1000 response times
    if (this.metrics.responseTimes.length > 1000) {
      this.metrics.responseTimes.shift();
    }
    
    if (parser) {
      if (!this.metrics.requestsByParser[parser]) {
        this.metrics.requestsByParser[parser] = 0;
      }
      this.metrics.requestsByParser[parser]++;
    }
  }

  recordFailure(error) {
    this.metrics.failedRequests++;
    this.metrics.errors.push({
      message: error.message,
      timestamp: new Date().toISOString()
    });
    
    // Keep only last 100 errors
    if (this.metrics.errors.length > 100) {
      this.metrics.errors.shift();
    }
  }

  getMetrics() {
    const uptime = Math.floor((Date.now() - this.startTime) / 1000);
    const errorRate = this.metrics.totalRequests > 0
      ? this.metrics.failedRequests / this.metrics.totalRequests
      : 0;
    
    const avgResponseTime = this.metrics.responseTimes.length > 0
      ? this.metrics.responseTimes.reduce((a, b) => a + b, 0) / this.metrics.responseTimes.length
      : 0;
    
    return {
      totalRequests: this.metrics.totalRequests,
      successfulRequests: this.metrics.successfulRequests,
      failedRequests: this.metrics.failedRequests,
      errorRate: parseFloat(errorRate.toFixed(4)),
      avgResponseTime: Math.round(avgResponseTime),
      uptime,
      requestsByAction: this.metrics.requestsByAction,
      requestsByParser: this.metrics.requestsByParser,
      recentErrors: this.metrics.errors.slice(-10)
    };
  }

  reset() {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      requestsByAction: {},
      requestsByParser: {},
      responseTimes: [],
      errors: []
    };
    this.startTime = Date.now();
  }
}

// Singleton instance
const metricsCollector = new MetricsCollector();

const metricsMiddleware = (req, res, next) => {
  // Record start time
  req.startTime = Date.now();
  
  // Record request
  const action = req.body?.action || 'unknown';
  metricsCollector.recordRequest(action);
  
  // Intercept response
  const originalJson = res.json.bind(res);
  res.json = function(data) {
    const responseTime = Date.now() - req.startTime;
    
    if (data.status === 'ok') {
      const parser = data.data?.parser || null;
      metricsCollector.recordSuccess(responseTime, parser);
    } else if (data.status === 'error') {
      metricsCollector.recordFailure(new Error(data.error?.message || 'Unknown error'));
    }
    
    return originalJson(data);
  };
  
  next();
};

module.exports = {
  metricsMiddleware,
  metricsCollector
};
