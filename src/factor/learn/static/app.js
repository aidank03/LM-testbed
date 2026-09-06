const state = { lineEvents: [], runs: [] };
const $ = (id) => document.getElementById(id);

function showLesson(id) {
  document.querySelectorAll('.lesson').forEach(el => el.classList.toggle('visible', el.id === id));
  document.querySelectorAll('.path-item').forEach(el => el.classList.toggle('active', el.dataset.target === id));
  document.getElementById(id).scrollIntoView({ behavior: 'smooth', block: 'start' });
}
document.querySelectorAll('[data-target]').forEach(el => el.addEventListener('click', () => showLesson(el.dataset.target)));

[['line-epochs','line-epochs-out'], ['line-lr','line-lr-out'], ['line-noise','line-noise-out']].forEach(([input, output]) => {
  $(input).addEventListener('input', () => $(output).textContent = $(input).value);
});

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed: ${response.status}`);
  return body;
}

async function loadStatus() {
  try {
    const status = await request('/api/status');
    $('runtime-dot').classList.add('ready');
    $('runtime-text').textContent = `PyTorch ${status.torch_version} · ${status.device.toUpperCase()} · local`;
    $('jax-status').className = `result ${status.jax.installed ? '' : 'empty'}`;
    $('jax-status').innerHTML = `<b>JAX ${status.jax.installed ? 'is installed' : 'is not installed yet'}.</b> ${status.jax.message} No JAX experiment has been executed in this lab.`;
  } catch (error) {
    $('runtime-text').textContent = 'Local server unavailable';
  }
}

function axes(ctx, width, height, labels) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#d9d5c9'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(46, 15); ctx.lineTo(46, height - 32); ctx.lineTo(width - 12, height - 32); ctx.stroke();
  ctx.fillStyle = '#64706c'; ctx.font = '12px system-ui';
  if (labels) { ctx.fillText(labels.y, 8, 17); ctx.fillText(labels.x, width - 55, height - 8); }
}

function drawLoss(events) {
  const canvas = $('loss-chart'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  axes(ctx, w, h, {x: 'step', y: 'loss'});
  if (events.length < 2) return;
  const values = events.flatMap(e => [e.train_loss, e.dev_loss]).filter(v => v > 0);
  const minLog = Math.log10(Math.min(...values)), maxLog = Math.log10(Math.max(...values));
  const draw = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
    events.forEach((e, i) => {
      const x = 47 + i / (events.length - 1) * (w - 61);
      const y = 15 + (maxLog - Math.log10(Math.max(e[key], 1e-8))) / Math.max(maxLog - minLog, .01) * (h - 49);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }); ctx.stroke();
  };
  draw('train_loss', '#0b7158'); draw('dev_loss', '#db6b35');
  ctx.fillStyle = '#0b7158'; ctx.fillText('train', 60, 30); ctx.fillStyle = '#db6b35'; ctx.fillText('development', 110, 30);
}

function drawFit(points) {
  const canvas = $('fit-chart'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  axes(ctx, w, h, {x: 'x', y: 'y'});
  if (!points?.length) return;
  const xs = points.map(p => p.x), ys = points.flatMap(p => [p.observed, p.predicted]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = x => 48 + (x - x0) / (x1 - x0) * (w - 64);
  const py = y => 15 + (y1 - y) / (y1 - y0) * (h - 49);
  ctx.strokeStyle = '#db6b35'; ctx.lineWidth = 3; ctx.beginPath();
  points.forEach((p, i) => i ? ctx.lineTo(px(p.x), py(p.predicted)) : ctx.moveTo(px(p.x), py(p.predicted))); ctx.stroke();
  ctx.fillStyle = '#2e5b9a'; points.forEach(p => { ctx.beginPath(); ctx.arc(px(p.x), py(p.observed), 4, 0, Math.PI*2); ctx.fill(); });
  ctx.fillStyle = '#64706c'; ctx.fillText('blue = unseen data · orange = learned line', 60, 30);
}

function drawMotion(stages) {
  const canvas = $('motion-chart'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  axes(ctx, w, h, {x: 'experiment stage', y: 'MAE (nm)'});
  if (!stages?.length) return;
  const max = Math.max(...stages.map(s => s.metrics.mae_nm)) * 1.18;
  const colors = ['#a43d35', '#db6b35', '#0b7158'];
  stages.forEach((stage, i) => {
    const slot = (w - 70) / stages.length, barW = Math.min(120, slot * .55);
    const x = 48 + i * slot + (slot - barW) / 2;
    const barH = stage.metrics.mae_nm / max * (h - 75), y = h - 32 - barH;
    ctx.fillStyle = colors[i]; ctx.fillRect(x, y, barW, barH);
    ctx.fillStyle = '#17221f'; ctx.textAlign = 'center'; ctx.font = 'bold 14px system-ui';
    ctx.fillText(`${stage.metrics.mae_nm.toFixed(2)} nm`, x + barW/2, y - 8);
    ctx.font = '12px system-ui'; ctx.fillText(stage.name, x + barW/2, h - 9);
  }); ctx.textAlign = 'left';
}

function addHistory(run) {
  const existing = state.runs.findIndex(r => r.id === run.id);
  if (existing >= 0) state.runs[existing] = run;
  else state.runs.unshift(run);
  $('history-list').innerHTML = state.runs.map(r => `<div class="history-row"><b>${r.kind.replace('_',' ')}</b><code>${r.id}</code><span>${r.status}</span></div>`).join('');
}

async function startRun(kind, config, onUpdate) {
  const run = await request('/api/runs', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({kind, config})});
  addHistory(run);
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const fresh = await request(`/api/runs/${run.id}`);
        addHistory(fresh); onUpdate(fresh);
        if (fresh.status === 'completed') resolve(fresh);
        else if (fresh.status === 'failed') reject(new Error(fresh.error));
        else setTimeout(poll, 110);
      } catch (error) { reject(error); }
    }; poll();
  });
}

$('run-line').addEventListener('click', async () => {
  const button = $('run-line'); button.disabled = true; state.lineEvents = [];
  $('line-result').className = 'result empty'; $('line-result').textContent = 'Training… watch the slope, intercept, gradient, and loss change.';
  try {
    await startRun('line_fit', {epochs: Number($('line-epochs').value), learning_rate: Number($('line-lr').value), noise: Number($('line-noise').value)}, run => {
      const epochs = run.events.filter(e => e.kind === 'epoch'); state.lineEvents = epochs;
      const last = epochs.at(-1);
      if (last) {
        $('line-step').textContent = `${last.epoch} / ${last.epochs}`;
        $('line-weight').textContent = last.weight.toFixed(3); $('line-bias').textContent = last.bias.toFixed(3); $('line-grad').textContent = last.gradient_norm.toFixed(4); drawLoss(epochs);
      }
      if (run.result) {
        drawFit(run.result.plot); const pass = run.result.gate.accepted;
        $('line-result').className = `result ${pass ? '' : 'fail'}`;
        $('line-result').innerHTML = `<b>${pass ? 'PASS' : 'NEEDS WORK'}:</b> final test MSE ${run.result.metrics.test_mse.toFixed(4)}. Learned y = ${run.result.learned.weight.toFixed(3)}x ${run.result.learned.bias < 0 ? '−' : '+'} ${Math.abs(run.result.learned.bias).toFixed(3)}. The test set did not change the weights.`;
      }
    });
  } catch (error) { $('line-result').className = 'result fail'; $('line-result').textContent = error.message; }
  finally { button.disabled = false; }
});

$('run-motion').addEventListener('click', async () => {
  const button = $('run-motion'); button.disabled = true;
  document.querySelectorAll('.stage-cards article').forEach(el => { el.classList.remove('active','done'); el.querySelector('.stage-score').textContent = '—'; });
  try {
    await startRun('motion_loop', {}, run => {
      const stageEvent = run.events.filter(e => e.kind === 'stage').at(-1);
      const epochEvent = run.events.filter(e => e.kind === 'stage_epoch').at(-1);
      if (stageEvent) {
        document.querySelectorAll('.stage-cards article').forEach(el => el.classList.toggle('active', el.id === `stage-${stageEvent.stage}`));
        $('motion-live').textContent = `${stageEvent.title}${epochEvent?.stage === stageEvent.stage ? ` · step ${epochEvent.epoch}/${epochEvent.epochs} · development MAE ${epochEvent.development_mae_nm.toFixed(2)} nm` : ''}`;
      }
      run.events.filter(e => e.kind === 'diagnosis').forEach(e => {
        const card = $(`stage-${e.stage}`); card.classList.remove('active'); card.classList.add('done'); card.querySelector('.stage-score').textContent = `${e.metrics.mae_nm.toFixed(2)} nm`;
      });
      if (run.result) {
        const ref = $('stage-reference'); ref.classList.remove('active'); ref.classList.add('done'); ref.querySelector('.stage-score').textContent = `${run.result.stages[2].metrics.mae_nm.toFixed(2)} nm`;
        drawMotion(run.result.stages); const pass = run.result.gate.accepted;
        $('motion-result').className = `result ${pass ? '' : 'fail'}`;
        $('motion-result').innerHTML = `<b>${pass ? 'PASS' : 'NEEDS WORK'}:</b> the reference reduced MAE by ${run.result.gate.mae_improvement_nm.toFixed(2)} nm. More training examples alone could not separate two hidden quantities from their sum. Adding a measurement changed what was identifiable. This is synthetic teaching evidence, not validation of real PDV or liner physics.`;
        $('motion-live').textContent = 'Complete. The locked test set was used for scoring, never for weight updates.';
      }
    });
  } catch (error) { $('motion-result').className = 'result fail'; $('motion-result').textContent = error.message; }
  finally { button.disabled = false; }
});

axes($('loss-chart').getContext('2d'), 620, 260, {x:'step', y:'loss'});
axes($('fit-chart').getContext('2d'), 620, 260, {x:'x', y:'y'});
axes($('motion-chart').getContext('2d'), 900, 300, {x:'experiment stage', y:'MAE (nm)'});
loadStatus();
