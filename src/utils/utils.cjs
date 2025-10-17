/**
 * General utility functions
 */

/**
 * Clean and normalize text
 * @param {string} text - Text to clean
 * @returns {string} Cleaned text
 */
function _clean(text) {
  if (!text) return '';
  return text
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[^\w\s.,!?'-]/g, '');
}

/**
 * Push all items from source array to target array
 * @param {Array} target - Target array
 * @param {Array} source - Source array
 */
function _pushAll(target, source) {
  if (!Array.isArray(target) || !Array.isArray(source)) return;
  source.forEach((item) => target.push(item));
}

/**
 * Deep clone an object
 * @param {*} obj - Object to clone
 * @returns {*} Cloned object
 */
function _deepClone(obj) {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime());
  if (obj instanceof Array) return obj.map((item) => _deepClone(item));
  if (obj instanceof Object) {
    const clonedObj = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = _deepClone(obj[key]);
      }
    }
    return clonedObj;
  }
}

/**
 * Debounce function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function _debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

module.exports = {
  _clean,
  _pushAll,
  _deepClone,
  _debounce,
};
