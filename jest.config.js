module.exports = {
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{js,cjs}',
    '!src/**/*.test.{js,cjs}',
    '!src/server.js'
  ],
  testMatch: [
    '**/tests/**/*.test.js'
  ],
  testTimeout: 30000,
  verbose: true
};
