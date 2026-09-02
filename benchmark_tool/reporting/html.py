"""Self-contained HTML rendering from canonical JSONL only."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_tool.io import read_jsonl, write_text
from benchmark_tool.results import CanonicalResult


DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenRouter Provider Benchmark</title>
<style>
:root{--bg:#09111f;--panel:#111c2f;--ink:#e8eefb;--muted:#9eb0ca;--line:#263754;--accent:#69d5c3;--warn:#ffcb6b;--bad:#ff6b7a;--good:#61d095}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#0e1830);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:32px}.eyebrow{text-transform:uppercase;letter-spacing:.18em;color:var(--accent);font-size:11px}h1{font-size:34px;margin:.2rem 0}.lede{color:var(--muted);max-width:760px}.cards,.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}.card,.panel{background:rgba(17,28,47,.94);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 12px 30px #0004}.card b{display:block;font-size:29px}.card span,.note{color:var(--muted)}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.controls label{color:var(--muted)}select{display:block;background:#0b1628;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;min-width:170px}.panel h2{font-size:16px;margin:0 0 13px}.chart{min-height:190px}.bar-row{display:grid;grid-template-columns:minmax(130px,1fr) 3fr 76px;gap:9px;align-items:center;margin:9px 0}.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}.track{height:13px;background:#091321;border-radius:99px;overflow:hidden}.bar{height:100%;background:linear-gradient(90deg,var(--accent),#809dff)}.bar-value{text-align:right;font-variant-numeric:tabular-nums}section{margin:25px 0}h2{font-size:20px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;background:#0d182a}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{position:sticky;top:0;background:#15233a;color:#bbcae1;font-size:11px;text-transform:uppercase;letter-spacing:.06em}tr:hover td{background:#13213a}.status{padding:3px 8px;border-radius:99px;background:#24324a}.completed{color:var(--good)}.failed,.unsupported{color:var(--bad)}.planned{color:var(--warn)}details{margin-top:18px}pre{white-space:pre-wrap;max-height:520px;overflow:auto;background:#07101d;padding:16px;border-radius:10px;color:#b9cae4}.empty{color:var(--muted);padding:30px;text-align:center}@media(max-width:700px){main{padding:18px}.bar-row{grid-template-columns:100px 2fr 65px}}
</style>
</head>
<body><main>
<div class="eyebrow">Canonical results · schema 1.0</div>
<h1>OpenRouter Provider Benchmark</h1>
<p class="lede">Provider-pinned performance, reliability, correctness, and measured-token cost estimates. Invalid AgentX submissions and unsupported routes remain visible.</p>
<div id="cards" class="cards"></div>
<div class="controls">
 <label>Model<select id="model"><option value="">All models</option></select></label>
 <label>Provider<select id="provider"><option value="">All providers</option></select></label>
 <label>Status<select id="status"><option value="">All statuses</option></select></label>
</div>
<div class="charts">
 <div class="panel"><h2>TTFT p95 (ms)</h2><div id="ttft" class="chart"></div></div>
 <div class="panel"><h2>Request throughput (req/s)</h2><div id="throughput" class="chart"></div></div>
 <div class="panel"><h2>Reliability success rate</h2><div id="reliability" class="chart"></div></div>
 <div class="panel"><h2>Correctness score</h2><div id="correctness" class="chart"></div></div>
 <div class="panel"><h2>Estimated workload cost (USD)</h2><div id="cost" class="chart"></div></div>
</div>
<section><h2>Runs, models, and providers</h2><div id="run-metadata"></div></section>
<section><h2>Performance and AgentX validity</h2><div id="performance"></div></section>
<section><h2>Reliability error breakdown</h2><div id="errors"></div></section>
<section><h2>Correctness</h2><div id="correctness-table"></div></section>
<section><h2>Failures and unsupported combinations</h2><div id="failures"></div></section>
<details><summary>Canonical JSONL records</summary><pre id="raw"></pre></details>
<p class="note">This report performs presentation transforms only. Metrics and derived values come from the canonical JSONL produced by analysis.py.</p>
</main>
<script>
const records=__DATA__;
const modelFilter=document.getElementById("model"),providerFilter=document.getElementById("provider"),statusFilter=document.getElementById("status");
const esc=v=>String(v??"").replace(/[&<>\"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num=(v,d=3)=>typeof v==="number"?v.toLocaleString(undefined,{maximumFractionDigits:d}):"—";
const label=r=>`${r.model.id} · ${r.provider.id}${r.workload.concurrency?` · c${r.workload.concurrency}`:""}`;
function options(id,values){const e=document.getElementById(id);[...new Set(values)].sort().forEach(v=>e.insertAdjacentHTML("beforeend",`<option>${esc(v)}</option>`));}
options("model",records.map(r=>r.model.id));options("provider",records.map(r=>r.provider.id));options("status",records.map(r=>r.status));
function table(headers,rows){if(!rows.length)return '<div class="empty">No matching records</div>';return `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;}
function bars(id,items,format=v=>num(v)){const root=document.getElementById(id);if(!items.length){root.innerHTML='<div class="empty">No data</div>';return}const max=Math.max(...items.map(x=>x.value),.000001);root.innerHTML=items.sort((a,b)=>b.value-a.value).slice(0,12).map(x=>`<div class="bar-row"><div class="bar-label" title="${esc(x.label)}">${esc(x.label)}</div><div class="track"><div class="bar" style="width:${100*x.value/max}%"></div></div><div class="bar-value">${format(x.value)}</div></div>`).join("");}
function filtered(){return records.filter(r=>(!modelFilter.value||r.model.id===modelFilter.value)&&(!providerFilter.value||r.provider.id===providerFilter.value)&&(!statusFilter.value||r.status===statusFilter.value));}
function render(){const rows=filtered();const counts=k=>rows.filter(r=>r.status===k).length;const invalid=rows.filter(r=>r.agentx&&r.agentx.submission_valid!==true).length;document.getElementById("cards").innerHTML=[["Records",rows.length],["Completed",counts("completed")],["Failed",counts("failed")],["Unsupported",counts("unsupported")],["Invalid/unverified AgentX",invalid]].map(([k,v])=>`<div class="card"><span>${k}</span><b>${v}</b></div>`).join("");
const perf=rows.filter(r=>r.performance);bars("ttft",perf.filter(r=>r.performance.ttft?.p95!=null).map(r=>({label:label(r),value:r.performance.ttft.p95})));bars("throughput",perf.filter(r=>r.performance.request_throughput?.mean!=null).map(r=>({label:label(r),value:r.performance.request_throughput.mean})));bars("reliability",rows.filter(r=>r.reliability?.success_rate!=null).map(r=>({label:label(r),value:r.reliability.success_rate})),v=>(100*v).toFixed(2)+"%");bars("correctness",rows.filter(r=>r.correctness).map(r=>({label:`${label(r)} · ${r.correctness.task}`,value:r.correctness.score})),v=>(100*v).toFixed(2)+"%");bars("cost",rows.filter(r=>r.pricing?.estimated_cost_usd!=null).map(r=>({label:label(r),value:r.pricing.estimated_cost_usd})),v=>"$"+num(v,6));
document.getElementById("run-metadata").innerHTML=table(["Run ID","Started","Completed","Model","OpenRouter model","Provider","OpenRouter provider"],rows.map(r=>[esc(r.run_id),esc(r.run_metadata?.started_at),esc(r.run_metadata?.completed_at),esc(r.model.id),esc(r.model.openrouter_id),esc(r.provider.id),esc(r.provider.openrouter_slug)]));
document.getElementById("performance").innerHTML=table(["Model","Provider","Workload","Concurrency","Status","AgentX valid","TTFT p95","E2E p95","ITL p95","Req/s","Output tok/s","Success","Cost","Cost/request"],perf.map(r=>[esc(r.model.id),esc(r.provider.id),esc(r.workload.name),num(r.workload.concurrency),`<span class="status ${esc(r.status)}">${esc(r.status)}</span>`,!r.agentx?"N/A":r.agentx.submission_valid===true?'<span class="completed">Yes</span>':'<span class="failed">No / unverified</span>',num(r.performance.ttft?.p95),num(r.performance.e2e_latency?.p95),num(r.performance.itl?.p95),num(r.performance.request_throughput?.mean),num(r.performance.output_token_throughput?.mean),r.reliability?.success_rate==null?"—":(100*r.reliability.success_rate).toFixed(2)+"%",r.pricing?.estimated_cost_usd==null?"—":"$"+num(r.pricing.estimated_cost_usd,6),r.pricing?.cost_per_request_usd==null?"—":"$"+num(r.pricing.cost_per_request_usd,6)]));
document.getElementById("errors").innerHTML=table(["Model","Provider","Concurrency","Category","Count"],rows.flatMap(r=>Object.entries(r.reliability?.errors||{}).filter(([,count])=>count).map(([category,count])=>[esc(r.model.id),esc(r.provider.id),num(r.workload.concurrency),esc(category),num(count,0)])));
document.getElementById("correctness-table").innerHTML=table(["Model","Provider","Task","Runner task","Metric","Score","Samples"],rows.filter(r=>r.correctness).map(r=>[esc(r.model.id),esc(r.provider.id),esc(r.correctness.task),esc(r.correctness.runner_task),esc(r.correctness.primary_metric),(100*r.correctness.score).toFixed(2)+"%",num(r.correctness.sample_count)]));
document.getElementById("failures").innerHTML=table(["Model","Provider","Workload","Status","Reason"],rows.filter(r=>r.status!=="completed"||(r.agentx&&r.agentx.submission_valid!==true)).map(r=>{const invalid=Boolean(r.agentx&&r.agentx.submission_valid!==true);const invalidReason=r.agentx?.submission_invalid_reasons?.join(", ")||"AIPerf submission_valid was not true";return [esc(r.model.id),esc(r.provider.id),esc(r.workload.name),`<span class="status ${invalid?'failed':esc(r.status)}">${invalid?'invalid/unverified AgentX':esc(r.status)}</span>`,esc(invalid?invalidReason:r.reason)]}));document.getElementById("raw").textContent=rows.map(r=>JSON.stringify(r)).join("\n");}
[modelFilter,providerFilter,statusFilter].forEach(e=>e.addEventListener("change",render));render();
</script></body></html>"""


def render_report(input_path: Path, output_path: Path) -> list[CanonicalResult]:
    records = [CanonicalResult.model_validate(raw) for raw in read_jsonl(input_path)]
    data = json.dumps([record.json_record() for record in records], ensure_ascii=False).replace("<", "\\u003c")
    write_text(output_path, DOCUMENT.replace("__DATA__", data))
    return records
