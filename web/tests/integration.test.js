import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {cohortSummary} from '../dist/science.js';

const root = new URL('../dist/', import.meta.url);
const html = fs.readFileSync(new URL('index.html', root), 'utf8');
const data = JSON.parse(fs.readFileSync(new URL('results.json', root), 'utf8'));

function staticFiles(directory) {
  return fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const path = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory);
    return entry.isDirectory() ? staticFiles(path) : [path];
  });
}

test('Application has independent working panels and local scripts', () => {
  for (const name of ['image-panel', 'halo-panel', 'evidence-panel', 'file-input',
    'download-halo', 'download-measurement', 'download-evidence']) {
    assert.ok(html.includes(`id="${name}"`), `Missing control ${name}`);
  }
  assert.match(html, /<script\b[^>]*type="module"[^>]*src="\.\/app\.js(?:\?v=[\d.]+)?"/);
  assert.ok(!html.includes('<script src="http'));
  for (const file of ['app.js', 'science.js', 'style.css', 'results.json']) {
    assert.ok(fs.statSync(new URL(file, root)).size > 100, `Missing asset ${file}`);
  }
});

test('Frozen cohort denominators and unknown labels stay intact', () => {
  assert.equal(data.galaxies, 99);
  assert.equal(data.imageFailures, 1);
  const three = cohortSummary(data.rows, 'strong_spiral', 3);
  assert.equal(three.evaluable, 57);
  assert.ok(Math.abs(three.meanAP - .6890343815760316) < 1e-12);
  const five = cohortSummary(data.rows, 'strong_spiral', 5);
  assert.equal(five.evaluable, 53);
  assert.ok(Math.abs(five.meanAP - .486753058525129) < 1e-12);
  assert.equal(cohortSummary(data.rows, 'low_vote_unknown', 3).meanAP, null);
  assert.equal(data.cohortIdentities.strong_spiral, 59);
});

test('Public deployment excludes manuscripts and private workspace material', () => {
  for (const file of staticFiles(root)) {
    const relative = file.pathname.slice(root.pathname.length);
    assert.ok(!/(?:^|\/)(?:paper|papers|manuscript|supplementary|research_review|publication_review)(?:\/|\.)/i.test(relative), relative);
    assert.ok(!/\.(?:tex|bib|bbl|pdf|zip|log)$/i.test(relative), relative);
    if (!/\.(?:html|js|css|json|svg|txt)$/i.test(relative)) continue;
    const text = fs.readFileSync(file, 'utf8');
    assert.ok(!/\/home\/[^/]+\/(?:codes|projects)\//.test(text), relative);
    assert.ok(!/(?:executed_next|research_review|publication_review)\//.test(text), relative);
    assert.ok(!/(?:cdn\.jsdelivr|unpkg\.com)/.test(text), relative);
  }
  assert.ok(!/(?:paper\.pdf|supplementary\.pdf|overleaf\.zip)/i.test(html));
  assert.ok(html.includes('https://github.com/nalin-dhiman/Galaxy-arm-project'));
});

test('Every local HTML asset and navigation fragment resolves', () => {
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
  for (const match of html.matchAll(/\b(?:href|src)=["']([^"']+)["']/g)) {
    const address = match[1];
    if (/^(?:https?:|mailto:|tel:|data:|blob:|\/\/)/.test(address)) continue;
    if (address === '#') continue;
    if (address.startsWith('#')) {
      assert.ok(ids.has(address.slice(1)), `Missing anchor ${address}`);
      continue;
    }
    const asset = new URL(address, root);
    assert.ok(asset.pathname.startsWith(root.pathname), `Asset escapes deployment: ${address}`);
    assert.ok(fs.existsSync(asset), `Broken local asset: ${address}`);
  }
});

test('Browser application module parses before release', () => {
  const result = spawnSync(process.execPath,
    ['--check', fileURLToPath(new URL('app.js', root))], {encoding: 'utf8'});
  assert.equal(result.status, 0, result.stderr);
});
