/**
 * POST /domain.extract
 * Model-driven domain extraction — no hardcoded regex patterns.
 *
 * Pipeline:
 *   1. compromise NLP  → extract noun/verb phrases as candidate terms
 *   2. @xenova/transformers zero-shot NLI → classify message against DOMAIN_LABELS
 *   3. Map top-scoring labels → services + skillHints via DOMAIN_SERVICE_MAP
 *
 * Fallback: keyword list if zero-shot model unavailable (cold start).
 *
 * Payload:  { message: string, threshold?: number }
 * Response: { tags, services, skillHints, scores, method, elapsedMs }
 */
const express = require('express');
const router = express.Router();

let _zeroShotPipeline = null;
async function getZeroShotPipeline() {
  if (!_zeroShotPipeline) {
    const { pipeline } = require('@xenova/transformers');
    _zeroShotPipeline = await pipeline('zero-shot-classification', 'Xenova/nli-deberta-v3-small');
  }
  return _zeroShotPipeline;
}

// Candidate domain labels the NLI model classifies against
const DOMAIN_LABELS = [
  'sms', 'text-message', 'email', 'phone-call',
  'github', 'gitlab', 'pull-request', 'code-review',
  'car-control', 'vehicle', 'smart-home', 'iot',
  'calendar', 'scheduling', 'slack', 'discord', 'notion',
  'aws', 'cloud', 'docker', 'kubernetes',
  'stripe', 'payments', 'billing',
  'twitter', 'social-media', 'spotify', 'music',
  'jira', 'project-management', 'file-system',
  'web-browser', 'web-automation', 'database',
  'twilio', 'sendgrid', 'shopify', 'salesforce',
];

// Maps label → { services[], skillHint }
const DOMAIN_SERVICE_MAP = {
  'sms':                { services: ['twilio', 'clicksend', 'sinch', 'vonage'],         skillHint: 'twilio.sms' },
  'text-message':       { services: ['twilio', 'clicksend', 'sinch'],                   skillHint: 'twilio.sms' },
  'twilio':             { services: ['twilio'],                                          skillHint: 'twilio.sms' },
  'email':              { services: ['gmail', 'sendgrid', 'mailgun', 'smtp'],           skillHint: 'gmail.send' },
  'sendgrid':           { services: ['sendgrid', 'mailgun'],                            skillHint: 'sendgrid.email' },
  'phone-call':         { services: ['twilio', 'vonage'],                               skillHint: 'twilio.call' },
  'github':             { services: ['github'],                                          skillHint: 'github.agent' },
  'gitlab':             { services: ['gitlab'],                                          skillHint: 'gitlab.agent' },
  'pull-request':       { services: ['github', 'gitlab', 'bitbucket'],                 skillHint: 'github.pr-review' },
  'code-review':        { services: ['github', 'gitlab', 'bitbucket'],                 skillHint: 'github.pr-review' },
  'car-control':        { services: ['smartcar', 'tesla', 'onstar'],                   skillHint: 'smartcar.control' },
  'vehicle':            { services: ['smartcar', 'tesla', 'onstar'],                   skillHint: 'smartcar.control' },
  'smart-home':         { services: ['homekit', 'philips-hue', 'nest', 'smartthings'], skillHint: 'homekit.control' },
  'iot':                { services: ['homekit', 'philips-hue', 'smartthings'],          skillHint: 'homekit.control' },
  'calendar':           { services: ['google-calendar', 'outlook', 'calendly'],         skillHint: 'gcal.event' },
  'scheduling':         { services: ['google-calendar', 'outlook', 'calendly'],         skillHint: 'gcal.event' },
  'slack':              { services: ['slack'],                                           skillHint: 'slack.message' },
  'discord':            { services: ['discord'],                                         skillHint: 'discord.message' },
  'notion':             { services: ['notion'],                                          skillHint: 'notion.page' },
  'aws':                { services: ['aws'],                                             skillHint: 'aws.cli' },
  'cloud':              { services: ['aws', 'gcloud', 'azure'],                         skillHint: 'aws.cli' },
  'docker':             { services: ['docker'],                                          skillHint: 'docker.cli' },
  'kubernetes':         { services: ['kubectl', 'kubernetes'],                           skillHint: 'kubectl.cli' },
  'stripe':             { services: ['stripe'],                                          skillHint: 'stripe.api' },
  'payments':           { services: ['stripe', 'paypal'],                               skillHint: 'stripe.api' },
  'billing':            { services: ['stripe', 'chargebee'],                            skillHint: 'stripe.api' },
  'shopify':            { services: ['shopify'],                                         skillHint: 'shopify.api' },
  'salesforce':         { services: ['salesforce'],                                      skillHint: 'salesforce.api' },
  'twitter':            { services: ['twitter'],                                         skillHint: 'twitter.post' },
  'social-media':       { services: ['twitter', 'instagram', 'linkedin'],               skillHint: 'twitter.post' },
  'spotify':            { services: ['spotify'],                                         skillHint: 'spotify.control' },
  'music':              { services: ['spotify', 'apple-music'],                         skillHint: 'spotify.control' },
  'jira':               { services: ['jira', 'linear', 'asana'],                        skillHint: 'jira.ticket' },
  'project-management': { services: ['jira', 'linear', 'asana', 'trello'],             skillHint: 'jira.ticket' },
  'web-browser':        { services: ['playwright', 'browser.act'],                      skillHint: 'browser.act' },
  'web-automation':     { services: ['playwright', 'browser.act'],                      skillHint: 'browser.act' },
  'database':           { services: ['postgres', 'mysql', 'sqlite'],                    skillHint: 'db.query' },
};

// Minimal keyword fallback used when zero-shot model is unavailable
const KEYWORD_FALLBACK = [
  { words: ['text', 'sms', 'texting'],                   label: 'sms' },
  { words: ['email', 'gmail', 'smtp', 'sendgrid'],       label: 'email' },
  { words: ['call', 'phone call', 'ring'],                label: 'phone-call' },
  { words: ['github', 'pull request', ' pr ', 'commit'], label: 'github' },
  { words: ['review', 'merge request', 'diff'],          label: 'pull-request' },
  { words: ['car', 'vehicle', 'truck', 'drive'],         label: 'car-control' },
  { words: ['light', 'thermostat', 'nest', 'hue'],       label: 'smart-home' },
  { words: ['calendar', 'meeting', 'event', 'schedule'], label: 'calendar' },
  { words: ['slack', 'channel', 'workspace'],            label: 'slack' },
  { words: ['notion', 'database', 'page'],               label: 'notion' },
  { words: ['aws', ' s3', ' ec2', 'lambda'],             label: 'aws' },
  { words: ['stripe', 'payment', 'charge', 'billing'],   label: 'stripe' },
  { words: ['docker', 'container', 'image'],             label: 'docker' },
  { words: ['kubernetes', 'kubectl', 'k8s'],             label: 'kubernetes' },
  { words: ['tweet', 'twitter', 'x.com'],                label: 'twitter' },
  { words: ['spotify', 'playlist', 'song'],              label: 'spotify' },
  { words: ['jira', 'ticket', 'sprint'],                 label: 'jira' },
  { words: ['file', 'folder', 'directory'],              label: 'file-system' },
  { words: ['browser', 'chrome', 'website'],             label: 'web-browser' },
];

/**
 * Extract noun/verb phrase candidates from the message using compromise NLP.
 * Used to build supplemental context — not strictly required for zero-shot.
 */
function extractCandidateTerms(message) {
  try {
    const nlp = require('compromise');
    const doc = nlp(message.substring(0, 500));
    const terms = new Set();
    doc.verbs().forEach(v => { const t = v.text().trim().toLowerCase(); if (t && t.length > 2 && t.length < 40) terms.add(t); });
    doc.nouns().forEach(n => { const t = n.text().trim().toLowerCase(); if (t && t.length > 2 && t.length < 40) terms.add(t); });
    doc.organizations().forEach(o => { const t = o.text().trim().toLowerCase(); if (t && t.length > 1) terms.add(t); });
    return [...terms].slice(0, 12);
  } catch (_) {
    return [];
  }
}

router.post('/domain.extract', async (req, res) => {
  const startTime = Date.now();
  const { requestId, payload } = req.body;
  const { message, threshold = 0.35 } = payload || {};

  if (!message) {
    return res.status(400).json({
      version: 'mcp.v1', service: 'phi4', action: 'domain.extract', requestId,
      status: 'error',
      error: { code: 'INVALID_REQUEST', message: 'payload.message is required', retryable: false }
    });
  }

  try {
    const tags = new Set();
    const services = [];
    const skillHints = [];
    const seenServices = new Set();
    const scores = {};
    let method = 'zero-shot';

    const candidates = extractCandidateTerms(message);

    // Attempt zero-shot classification on the full message
    let zeroShotFailed = false;
    try {
      const classifier = await getZeroShotPipeline();
      const result = await classifier(message.substring(0, 512), DOMAIN_LABELS, { multi_label: true });
      result.labels.forEach((label, i) => {
        scores[label] = result.scores[i];
        if (result.scores[i] >= threshold) {
          tags.add(label);
          const mapping = DOMAIN_SERVICE_MAP[label];
          if (mapping) {
            if (mapping.skillHint && !skillHints.includes(mapping.skillHint)) skillHints.push(mapping.skillHint);
            for (const svc of mapping.services) {
              if (!seenServices.has(svc)) { services.push(svc); seenServices.add(svc); }
            }
          }
        }
      });
    } catch (zeroShotErr) {
      console.warn('[phi4/domain.extract] Zero-shot failed, using keyword fallback:', zeroShotErr.message);
      zeroShotFailed = true;
      method = 'keyword-fallback';
    }

    // Keyword fallback if zero-shot unavailable or returned nothing
    if (zeroShotFailed || tags.size === 0) {
      method = 'keyword-fallback';
      const lowerMsg = message.toLowerCase();
      for (const { words, label } of KEYWORD_FALLBACK) {
        if (words.some(w => lowerMsg.includes(w))) {
          tags.add(label);
          const mapping = DOMAIN_SERVICE_MAP[label];
          if (mapping) {
            if (mapping.skillHint && !skillHints.includes(mapping.skillHint)) skillHints.push(mapping.skillHint);
            for (const svc of mapping.services) {
              if (!seenServices.has(svc)) { services.push(svc); seenServices.add(svc); }
            }
          }
        }
      }
    }

    const elapsedMs = Date.now() - startTime;
    res.json({
      version: 'mcp.v1', service: 'phi4', action: 'domain.extract', requestId,
      status: 'ok',
      data: { tags: [...tags], services, skillHints, scores, method, candidates, elapsedMs },
      error: null,
      metrics: { elapsedMs }
    });
  } catch (err) {
    console.error('[phi4/domain.extract] Error:', err.message);
    res.status(500).json({
      version: 'mcp.v1', service: 'phi4', action: 'domain.extract', requestId,
      status: 'error',
      error: { code: 'DOMAIN_EXTRACT_ERROR', message: err.message, retryable: true }
    });
  }
});

module.exports = router;
