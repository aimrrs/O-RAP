var paths = {prediction: '/', research: '/research', about: '/about'};
var samples = [];

function renderPage(name, push) {
  if (!paths[name]) name = 'prediction';
  if (push && location.pathname !== paths[name]) history.pushState({}, '', paths[name]);
  document.querySelectorAll('.page').forEach(function (el) {
    el.hidden = el.id !== name;
    el.classList.toggle('active', el.id === name);
  });
  document.querySelectorAll('[data-page-link]').forEach(function (el) {
    var active = el.dataset.pageLink === name;
    el.classList.toggle('active', active);
    if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
  });
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function pageFromPath() {
  if (location.pathname === '/research') return 'research';
  if (location.pathname === '/about') return 'about';
  return 'prediction';
}

function textElement(tag, cls, value) {
  var el = document.createElement(tag);
  if (cls) el.className = cls;
  el.textContent = value;
  return el;
}

function showError(message) {
  var card = document.getElementById('result-card');
  card.className = 'result-card'; card.replaceChildren();
  card.append(textElement('p','eyebrow','MODEL ESTIMATE'), textElement('h3','','We couldn’t run that estimate'), textElement('p','error-message',message));
  document.getElementById('model-details').hidden = true;
}

function showResult(result) {
  var card = document.getElementById('result-card');
  card.className = 'result-card has-result';
  var pct = (result.probability * 100).toFixed(1), cutoff = (result.threshold * 100).toFixed(1);
  card.innerHTML = '<p class="eyebrow">MODEL ESTIMATE</p><div class="probability-wrap"><span class="probability-value"></span><span class="probability-sign">estimated chance</span></div><span class="risk-badge"><i></i><span class="risk-text"></span></span><p class="result-explanation"></p><div class="result-foot"><span>Study cutoff ' + cutoff + '%</span><span class="mini-dot"></span><span>Calibrated estimate</span></div>';
  card.querySelector('.probability-value').textContent = pct + '%';
  card.querySelector('.risk-badge').classList.toggle('flagged', result.flagged);
  card.querySelector('.risk-text').textContent = result.flagged ? 'Above study cutoff' : 'Below study cutoff';
  card.querySelector('.result-explanation').textContent = 'Estimated probability that this source TLE is associated with more than 1 km of disagreement in the study comparison.';
  var box = document.getElementById('model-details'); box.hidden = false; box.replaceChildren();
  box.append(textElement('p','panel-kicker','MODEL DETAILS'), textElement('h3','','How this estimate was produced'));
  var rows = [['Estimator',result.model_details.estimator],['Calibration',result.model_details.calibration],['Inputs',result.model_details.input_features + ' orbital features parsed from the TLE set'],['Training setup',result.model_details.training_data],['Cutoff',cutoff + '%; selected to maximize F1 on validation data'],['Study label',result.model_details.label]];
  var dl = document.createElement('dl'); dl.className = 'details-list';
  rows.forEach(function (row) { var wrap=document.createElement('div');wrap.className='details-row';wrap.append(textElement('dt','',row[0]),textElement('dd','',row[1]));dl.append(wrap); });
  box.append(dl,textElement('p','details-caveat','Exploratory research estimate; not a physical orbit error or safety forecast.'));
}

async function loadSamples() {
  var select = document.getElementById('sample-select');
  try {
    var response = await fetch('/api/satellites'), data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load example TLE sets.');
    samples = data.satellites;
    select.replaceChildren(new Option('Choose a project example…',''));
    samples.forEach(function (sample,index) { select.add(new Option(sample.name + ' · ' + sample.epoch,String(index))); });
  } catch (err) { select.replaceChildren(new Option('Project examples unavailable','')); var meta=document.getElementById('record-meta'); if(meta) meta.textContent=err.message; }
}

function n(value,digits) { var x=Number(value);return Number.isFinite(x)?x.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits}):'—'; }

function metricGrid(parent, rows) {
  var grid=document.createElement('div');grid.className='metric-grid';
  rows.forEach(function(row){var card=document.createElement('article');card.className='metric-card';card.append(textElement('div','metric-label',row[0]),textElement('div','metric-value',row[1]),textElement('div','metric-note',row[2]));grid.append(card);});
  parent.append(grid);
}

function panel(parent,kicker,title,note) {
  var box=document.createElement('section');box.className='model-panel';
  var header=document.createElement('div');header.className='panel-header';
  var left=document.createElement('div');left.append(textElement('p','panel-kicker',kicker),textElement('h2','panel-title',title));header.append(left);
  if(note)header.append(textElement('p','panel-note',note));box.append(header);parent.append(box);return box;
}

function makeTable(headers, rows) {
  var wrap=document.createElement('div');wrap.className='result-table-wrap';var table=document.createElement('table');table.className='small-table';
  var thead=document.createElement('thead'), tr=document.createElement('tr');headers.forEach(function(h){tr.append(textElement('th','',h));});thead.append(tr);table.append(thead);
  var body=document.createElement('tbody');rows.forEach(function(row){var r=document.createElement('tr');row.forEach(function(v){r.append(textElement('td','',String(v)));});body.append(r);});table.append(body);wrap.append(table);return wrap;
}

function renderResearch(data) {
  var root=document.getElementById('research-content');root.replaceChildren();
  var overview=document.createElement('section');overview.className='research-summary research-overview';
  overview.append(textElement('p','panel-kicker','STUDY AT A GLANCE'),textElement('h2','','From historical TLEs to a time-ordered evaluation'),textElement('p','','Pairs compare two consecutive, distinct element records at a shared epoch. The model uses only orbital features available in the source TLE.'));
  root.append(overview);
  var splits=data.split_table, positive=splits.filter(function(r){return r.Partition!=='Boundary-excluded pairs';}).reduce(function(a,r){return a+Number(r.Positive);},0);
  metricGrid(root,[['Satellites',String(data.dataset.satellites),'with historical GP records'],['Retained pairs',Number(data.dataset.pairs).toLocaleString(),data.dataset.boundary_excluded+' boundary-crossing pairs excluded'],['Positive labels',positive.toLocaleString(),'disagreement greater than 1 km'],['Model inputs',String(data.dataset.features),'orbital fields parsed from source TLE']]);
  var sPanel=panel(root,'CHRONOLOGICAL DESIGN','How the data was split','Sorted by epoch for each satellite');
  var purpose={Train:'Fit the estimator',Validation:'Calibrate probabilities and select cutoff',Test:'Evaluate later-period performance'};
  sPanel.append(makeTable(['Partition','Pairs','Positive','Positive rate','Purpose'],splits.map(function(r){var excluded=r.Partition==='Boundary-excluded pairs';return [r.Partition,Number(r.Samples).toLocaleString(),excluded?'—':Number(r.Positive).toLocaleString(),excluded?'—':r['Positive rate'],purpose[r.Partition]||(excluded?'Crossing pairs removed to prevent split overlap':'')];})));
  var t=data.test, threshold=data.test.threshold;
  var tPanel=panel(root,'ORBITAL-ONLY CHRONOLOGICAL TEST','Performance on held-out later data',Number(t.samples).toLocaleString()+' test pairs · '+Number(t.positives).toLocaleString()+' positive');
  metricGrid(tPanel,[['PR-AUC',n(t.pr_auc,4),'baseline prevalence '+n(t.positive_rate_percent/100,4)],['ROC-AUC',n(t.roc_auc,4),'ranking discrimination'],['F1',n(t.f1,4),'at validation cutoff '+n(threshold*100,2)+'%'],['Brier score',n(t.brier_score,4),'lower is better'],['Precision',n(t.precision*100,1)+'%','flagged pairs that were positive'],['Recall',n(t.recall*100,1)+'%','positive pairs that were flagged']]);
  tPanel.append(textElement('p','subnote','The cutoff was selected on validation data to maximize F1 and then evaluated once on this later test set. PR-AUC should be compared with the positive-rate baseline.'));
  var models=data.models.slice().sort(function(a,b){return Number(b['PR-AUC'])-Number(a['PR-AUC']);});
  var mPanel=panel(root,'MODEL COMPARISON','Precision–recall performance','Same chronological test partition');
  var max=Math.max.apply(null,models.map(function(r){return Number(r['PR-AUC']);})), list=document.createElement('div');list.className='model-list';
  models.forEach(function(row,i){var item=document.createElement('div');item.className='model-row';var name=textElement('span','model-name',row.Model||row.model);var track=document.createElement('div');track.className='bar-track';var fill=document.createElement('div');fill.className='bar-fill '+(i===0?'best':'');fill.style.width=(Number(row['PR-AUC'])/max*100)+'%';track.append(fill);item.append(name,track,textElement('span','model-score',n(row['PR-AUC'],4)));list.append(item);});mPanel.append(list);
  mPanel.append(textElement('p','subnote','Some compared models include record interval, which is only known after the newer TLE arrives. The live prediction uses the orbital-only model.'));
  var cPanel=panel(root,'CALIBRATION','Probability calibration','Held-out chronological test set');
  cPanel.append(makeTable(['Output','PR-AUC','ROC-AUC','Brier','Log loss','Precision','Recall','F1'],data.calibration.map(function(r){return [r.model,n(r.pr_auc,4),n(r.roc_auc,4),n(r.brier_score,4),n(r.log_loss,4),n(r.precision,4),n(r.recall,4),n(r.f1,4)];})));
  cPanel.append(textElement('p','subnote','Platt scaling is fit on validation predictions. Better probability calibration can improve Brier score and log loss; the underlying ranking remains effectively unchanged.'));
  var thPanel=panel(root,'LABEL SENSITIVITY','Results at different disagreement thresholds','Same test cases; label definition changes');
  thPanel.append(makeTable(['Label threshold','Positive','Prevalence','PR-AUC baseline','PR-AUC','ROC-AUC','F1'],data.thresholds.map(function(r){return [r['Label threshold'],Number(r.Positive).toLocaleString(),r.Prevalence,r['PR-AUC baseline'],r['PR-AUC'],r['ROC-AUC'],r.F1];})));
  var loso=data.loso, lPanel=panel(root,'GENERALIZATION','Leave-one-satellite-out evaluation','Each fold holds out one satellite');
  metricGrid(lPanel,[['Held-out samples',Number(loso.samples).toLocaleString(),'across held-out folds'],['Raw PR-AUC',n(loso.raw_pr_auc,4),'pooled ranking score'],['Raw ROC-AUC',n(loso.raw_roc_auc,4),'pooled ranking score'],['Calibrated Brier',n(loso.calibrated_brier_score,4),'per-fold calibration'],['Calibrated log loss',n(loso.calibrated_log_loss,4),'per-fold calibration'],['Calibrated F1',n(loso.calibrated_f1,4),'fold-specific cutoffs']]);
  lPanel.append(textElement('p','subnote','Each held-out satellite fold has its own calibrator. Pooled calibrated ranking scores are not used to compare folds because fold-specific calibration can change score ordering.'));
  var caveat=document.createElement('aside');caveat.className='caution-panel';caveat.append(textElement('h2','','How to interpret the results'),textElement('p','','The outcome is disagreement between two TLE-derived SGP4 positions, not independently measured physical orbit error. Performance varies by satellite, time period, label threshold, and decision cutoff. This research does not provide collision probability or operational safety advice.'));root.append(caveat);
}

async function loadResearch(){try{var r=await fetch('/api/research'),d=await r.json();if(!r.ok)throw new Error(d.error||'Could not load research files.');renderResearch(d);}catch(e){document.getElementById('research-content').textContent=e.message;}}

document.addEventListener('DOMContentLoaded',function(){
  window.addEventListener('popstate',function(){renderPage(pageFromPath(),false);});
  document.querySelectorAll('[data-page-link],.brand,.page-footer a').forEach(function(link){link.addEventListener('click',function(e){var url=new URL(link.href,location.origin);if(url.origin!==location.origin)return;e.preventDefault();renderPage(url.pathname==='/research'?'research':url.pathname==='/about'?'about':'prediction',true);});});
  renderPage(pageFromPath(),false);loadSamples();loadResearch();
  document.getElementById('sample-select').addEventListener('change',function(e){var sample=samples[Number(e.target.value)];if(!sample)return;document.getElementById('tle-input').value=sample.tle;var meta=document.getElementById('record-meta');if(meta)meta.textContent=sample.name+' · source epoch '+sample.epoch;});
  document.getElementById('prediction-form').addEventListener('submit',async function(e){e.preventDefault();var button=e.currentTarget.querySelector('button[type="submit"]');button.disabled=true;button.querySelector('span').textContent='Estimating…';try{var r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tle:document.getElementById('tle-input').value})}),d=await r.json();if(!r.ok)throw new Error(d.error||'Prediction failed.');showResult(d);}catch(err){showError(err.message);}finally{button.disabled=false;button.querySelector('span').textContent='Estimate disagreement risk';}});
});
