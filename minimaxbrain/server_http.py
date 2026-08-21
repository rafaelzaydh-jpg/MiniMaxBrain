"""Minimal HTTP/Web surface for real MiniMaxBrain inference."""
from __future__ import annotations

import hmac
import json
import socketserver
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from .errors import BackendUnavailableError, ConfigurationError, InferenceError, MMBError
from .runtime import MMBRuntime
from .units import format_bytes


WEB_UI_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>MiniMaxBrain</title>
<style>
:root{
  --bg:#f5f5f7;
  --surface:#ffffff;
  --surface-raised:#ffffff;
  --surface-muted:#ececef;
  --text:#1d1d1f;
  --text-secondary:#6e6e73;
  --separator:rgba(60,60,67,.18);
  --accent:#007aff;
  --danger:#c9342f;
  --focus:rgba(0,122,255,.24);
  --shadow:0 14px 40px rgba(0,0,0,.10);
  --radius-sm:9px;
  --radius-md:13px;
  --radius-lg:18px;
  --content:860px;
  --header-h:58px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#111113;
    --surface:#19191c;
    --surface-raised:#202024;
    --surface-muted:#26262a;
    --text:#f5f5f7;
    --text-secondary:#a1a1a6;
    --separator:rgba(255,255,255,.13);
    --accent:#0a84ff;
    --danger:#ff6961;
    --focus:rgba(10,132,255,.30);
    --shadow:0 18px 48px rgba(0,0,0,.34);
  }
}
*{box-sizing:border-box}
html,body{
  width:100%;
  height:100%;
  margin:0;
  overflow:hidden;
}
body{
  background:var(--bg);
  color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16px;
  line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
button,textarea{font:inherit}
button{color:inherit}
button:focus-visible,textarea:focus-visible,summary:focus-visible{
  outline:2px solid var(--accent);
  outline-offset:2px;
}
.app{
  height:100vh;
  height:100dvh;
  min-height:0;
  overflow:hidden;
  display:grid;
  grid-template-rows:auto minmax(0,1fr) auto;
}
.topbar{
  min-width:0;
  border-bottom:1px solid var(--separator);
  background:var(--bg);
}
.topbar-inner{
  width:min(100%,var(--content));
  min-height:var(--header-h);
  margin:0 auto;
  padding:8px 18px;
  display:flex;
  align-items:center;
  gap:14px;
}
.brand{
  min-width:0;
  flex:1;
}
.brand-title{
  font-size:15px;
  line-height:1.25;
  font-weight:600;
  letter-spacing:-.01em;
}
.model-name{
  margin-top:2px;
  color:var(--text-secondary);
  font-size:12px;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.header-actions{
  display:flex;
  align-items:center;
  gap:8px;
  flex:none;
}
.runtime-state{
  color:var(--text-secondary);
  font-size:12px;
  white-space:nowrap;
}
.runtime-state[data-ready="false"]{color:var(--danger)}
.quiet-button,
.technical summary{
  min-height:34px;
  border:1px solid var(--separator);
  border-radius:var(--radius-sm);
  padding:6px 10px;
  background:transparent;
  color:var(--text-secondary);
  font-size:12px;
  line-height:1.2;
  cursor:pointer;
}
.quiet-button:hover,
.technical summary:hover,
.technical[open] summary{
  background:var(--surface-muted);
  color:var(--text);
}
.technical{
  position:relative;
}
.technical summary{
  display:flex;
  align-items:center;
  list-style:none;
  user-select:none;
}
.technical summary::-webkit-details-marker{display:none}
.technical-panel{
  position:absolute;
  z-index:30;
  top:calc(100% + 8px);
  right:0;
  width:min(360px,calc(100vw - 24px));
  max-height:min(520px,calc(100dvh - 84px));
  overflow:auto;
  padding:14px;
  background:var(--surface-raised);
  border:1px solid var(--separator);
  border-radius:var(--radius-md);
  box-shadow:var(--shadow);
}
.metrics{
  display:grid;
  grid-template-columns:minmax(0,1fr) auto;
  gap:8px 16px;
  margin:0;
  font-size:13px;
}
.metrics dt{color:var(--text-secondary)}
.metrics dd{
  min-width:0;
  margin:0;
  text-align:right;
  font-variant-numeric:tabular-nums;
  overflow-wrap:anywhere;
}
.metrics-separator{
  grid-column:1 / -1;
  height:1px;
  margin:4px 0;
  background:var(--separator);
}
.chat-region{
  position:relative;
  min-height:0;
  overflow:hidden;
}
.chat-scroll{
  position:absolute;
  inset:0;
  overflow-y:auto;
  overflow-x:hidden;
  overscroll-behavior-y:contain;
  scrollbar-gutter:stable;
  scrollbar-width:auto;
  scrollbar-color:rgba(110,110,115,.62) transparent;
}
.chat-scroll::-webkit-scrollbar{width:11px}
.chat-scroll::-webkit-scrollbar-track{background:transparent}
.chat-scroll::-webkit-scrollbar-thumb{
  min-height:48px;
  border:3px solid transparent;
  border-radius:999px;
  background-clip:padding-box;
  background-color:rgba(110,110,115,.52);
}
.chat-scroll::-webkit-scrollbar-thumb:hover{
  background-color:rgba(110,110,115,.74);
}
.chat{
  width:min(100%,var(--content));
  min-height:100%;
  margin:0 auto;
  padding:32px 18px 40px;
}
.empty{
  min-height:100%;
  display:grid;
  place-items:center;
  text-align:center;
}
.empty[hidden]{display:none}
.empty-inner{
  max-width:560px;
  padding:40px 0;
}
.empty h1{
  margin:0;
  font-size:clamp(26px,4vw,38px);
  line-height:1.08;
  letter-spacing:-.03em;
  font-weight:600;
}
.empty p{
  margin:10px auto 0;
  max-width:520px;
  color:var(--text-secondary);
  font-size:14px;
}
.messages{display:none}
.messages.active{display:block}
.message{
  display:grid;
  grid-template-columns:72px minmax(0,1fr);
  gap:18px;
  padding:20px 0;
  border-bottom:1px solid var(--separator);
}
.message:last-child{border-bottom:0}
.message-role{
  padding-top:2px;
  color:var(--text-secondary);
  font-size:12px;
  font-weight:600;
}
.message.user .message-role{color:var(--text)}
.message-body{min-width:0}
.message-content{
  min-width:0;
  white-space:pre-wrap;
  overflow-wrap:anywhere;
  word-break:break-word;
  line-height:1.62;
}
.message.user .message-content{
  display:inline-block;
  max-width:100%;
  padding:9px 12px;
  border-radius:var(--radius-md);
  background:var(--surface-muted);
}
.message.error .message-content{color:var(--danger)}
.message-actions{
  min-height:26px;
  margin-top:6px;
}
.copy-button{
  border:0;
  padding:4px 0;
  background:transparent;
  color:var(--text-secondary);
  font-size:12px;
  cursor:pointer;
}
.copy-button:hover{color:var(--text)}
.scroll-latest{
  position:absolute;
  z-index:12;
  left:50%;
  bottom:14px;
  transform:translateX(-50%);
  min-width:40px;
  min-height:36px;
  padding:6px 12px;
  border:1px solid var(--separator);
  border-radius:999px;
  background:var(--surface-raised);
  color:var(--text);
  box-shadow:0 5px 18px rgba(0,0,0,.10);
  cursor:pointer;
}
.scroll-latest[hidden]{display:none}
.composer-shell{
  min-width:0;
  border-top:1px solid var(--separator);
  background:var(--bg);
  padding:9px 18px calc(9px + env(safe-area-inset-bottom));
}
.composer{
  width:min(100%,var(--content));
  margin:0 auto;
}
.composer-box{
  display:flex;
  align-items:flex-end;
  gap:10px;
  padding:7px 7px 7px 13px;
  border:1px solid var(--separator);
  border-radius:var(--radius-lg);
  background:var(--surface);
}
.composer-box:focus-within{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--focus);
}
textarea{
  flex:1;
  min-width:0;
  height:44px;
  min-height:44px;
  max-height:144px;
  resize:none;
  overflow-y:auto;
  border:0;
  outline:0;
  padding:10px 0 8px;
  color:var(--text);
  background:transparent;
  line-height:1.45;
  scrollbar-width:thin;
}
textarea::placeholder{color:var(--text-secondary)}
.send{
  flex:none;
  width:44px;
  height:44px;
  border:0;
  border-radius:50%;
  background:var(--accent);
  color:#fff;
  display:grid;
  place-items:center;
  cursor:pointer;
  font-size:19px;
  line-height:1;
}
.send:hover:not(:disabled){filter:brightness(.96)}
.send:active:not(:disabled){transform:scale(.97)}
.send:disabled{opacity:.34;cursor:default}
.send.stop{
  border-radius:12px;
  background:var(--text);
  color:var(--bg);
}
.composer-meta{
  min-height:22px;
  padding:5px 4px 0;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  color:var(--text-secondary);
  font-size:12px;
}
.composer-status{
  min-width:0;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.tps{
  flex:none;
  color:var(--text);
  font-variant-numeric:tabular-nums;
}
.sr-only{
  position:absolute;
  width:1px;
  height:1px;
  padding:0;
  margin:-1px;
  overflow:hidden;
  clip:rect(0,0,0,0);
  white-space:nowrap;
  border:0;
}
@media (max-width:640px){
  :root{--header-h:54px}
  .topbar-inner{padding-inline:12px}
  .runtime-state{display:none}
  .quiet-button{display:none}
  .chat{padding:22px 14px 30px}
  .message{
    grid-template-columns:1fr;
    gap:5px;
    padding:17px 0;
  }
  .message-role{padding:0}
  .composer-shell{padding-left:10px;padding-right:10px}
  .technical-panel{right:-2px}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{
    scroll-behavior:auto!important;
    transition:none!important;
    animation:none!important;
  }
  .send:active:not(:disabled){transform:none}
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-title">MiniMaxBrain</div>
        <div class="model-name" id="modelName">Carregando modelo…</div>
      </div>
      <div class="header-actions">
        <span class="runtime-state" id="runtimeState" data-ready="false">Verificando…</span>
        <button class="quiet-button" id="newChat" type="button">Novo</button>
        <details class="technical" id="technical">
          <summary>Detalhes</summary>
          <div class="technical-panel">
            <dl class="metrics">
              <dt>Modo</dt><dd id="metricMode">—</dd>
              <dt>RAM do processo</dt><dd id="metricRam">—</dd>
              <dt>Cache de experts</dt><dd id="metricCache">—</dd>
              <dt>Cache hit</dt><dd id="metricHit">—</dd>
              <div class="metrics-separator"></div>
              <dt>Tokens gerados</dt><dd id="metricTokens">—</dd>
              <dt>TTFT</dt><dd id="metricTTFT">—</dd>
              <dt>Velocidade</dt><dd id="metricTPS">—</dd>
              <div class="metrics-separator"></div>
              <dt>Router requests</dt><dd id="metricRouter">0</dd>
              <dt>MMBW lido</dt><dd id="metricBytes">0 B</dd>
              <dt>Acquire acumulado</dt><dd id="metricAcquire">0 ms</dd>
              <dt>I/O acumulado</dt><dd id="metricIO">0 ms</dd>
            </dl>
          </div>
        </details>
      </div>
    </div>
  </header>

  <section class="chat-region" aria-label="Conversa">
    <div class="chat-scroll" id="chatScroll" tabindex="0">
      <main class="chat">
        <section class="empty" id="emptyState">
          <div class="empty-inner">
            <h1>Converse com seu modelo.</h1>
            <p id="emptyText">Verificando o runtime local.</p>
          </div>
        </section>
        <section class="messages" id="messages" aria-live="polite"></section>
      </main>
    </div>
    <button class="scroll-latest" id="scrollLatest" type="button" hidden aria-label="Ir para a mensagem mais recente">↓ Mais recente</button>
  </section>

  <footer class="composer-shell">
    <form class="composer" id="form">
      <label class="sr-only" for="prompt">Mensagem</label>
      <div class="composer-box">
        <textarea id="prompt" rows="1" placeholder="Mensagem" autocomplete="off" disabled></textarea>
        <button class="send" id="send" type="submit" aria-label="Enviar mensagem" title="Enviar" disabled>↑</button>
      </div>
      <div class="composer-meta">
        <span class="composer-status" id="composerStatus">Verificando runtime…</span>
        <span class="tps" id="tpsValue" aria-live="polite">— tok/s</span>
      </div>
    </form>
  </footer>
</div>

<script>
const messagesEl = document.getElementById('messages');
const emptyEl = document.getElementById('emptyState');
const promptEl = document.getElementById('prompt');
const sendEl = document.getElementById('send');
const formEl = document.getElementById('form');
const scrollEl = document.getElementById('chatScroll');
const scrollLatestEl = document.getElementById('scrollLatest');
const newChatEl = document.getElementById('newChat');
const history = [];

let apiToken = sessionStorage.getItem('mmb_api_token') || '';
let runtimeReady = false;
let isStreaming = false;
let abortController = null;
let lastTPS = null;
let pinnedToBottom = true;
let programmaticScroll = false;
let conversationEpoch = 0;

async function apiFetch(url, options = {}){
  options.headers = options.headers || {};
  if(apiToken) options.headers['Authorization'] = 'Bearer ' + apiToken;
  let response = await fetch(url, options);
  if(response.status === 401 && !apiToken){
    const value = window.prompt('Token da API local:');
    if(value){
      apiToken = value.trim();
      sessionStorage.setItem('mmb_api_token', apiToken);
      options.headers['Authorization'] = 'Bearer ' + apiToken;
      response = await fetch(url, options);
    }
  }
  return response;
}

function formatBytes(value){
  const n = Number(value || 0);
  if(!Number.isFinite(n) || n <= 0) return '0 B';
  const units = ['B','KiB','MiB','GiB','TiB'];
  let v = n;
  let index = 0;
  while(v >= 1024 && index < units.length - 1){
    v /= 1024;
    index += 1;
  }
  return (v >= 10 || index === 0 ? v.toFixed(index === 0 ? 0 : 1) : v.toFixed(2)) + ' ' + units[index];
}

function formatMsFromNs(value){
  const n = Number(value || 0);
  if(!Number.isFinite(n) || n <= 0) return '0 ms';
  return (n / 1e6).toFixed(0) + ' ms';
}

function setText(id, text){
  const element = document.getElementById(id);
  if(element) element.textContent = text;
}

function updateGenerationStats(stats){
  if(!stats) return;
  const tokens = Number(stats.tokens_generated ?? 0);
  const tps = Number(stats.tokens_per_second);
  const ttft = Number(stats.ttft_ms);

  setText('metricTokens', Number.isFinite(tokens) ? String(tokens) : '—');
  setText('metricTTFT', Number.isFinite(ttft) ? ttft.toFixed(0) + ' ms' : '—');

  if(Number.isFinite(tps) && tps > 0){
    lastTPS = tps;
    const value = '≈ ' + tps.toFixed(2) + ' tok/s';
    setText('metricTPS', value);
    setText('tpsValue', value);
  }else if(lastTPS === null){
    setText('metricTPS', '—');
    setText('tpsValue', '— tok/s');
  }
}

function setRuntimeUI(stats){
  const pager = stats.native_pager || {};
  const generation = stats.generation || {};
  runtimeReady = Boolean(stats.ready);

  setText('modelName', stats.model_id || 'MiniMaxBrain');
  const state = document.getElementById('runtimeState');
  state.dataset.ready = runtimeReady ? 'true' : 'false';
  state.textContent = runtimeReady ? 'Pronto' : 'Indisponível';

  promptEl.disabled = !runtimeReady || isStreaming;
  updateSendState();

  setText('metricMode', stats.inference_mode || '—');
  setText('metricRam', formatBytes(stats.backend_rss_bytes));
  setText(
    'metricCache',
    formatBytes(stats.expert_cache_bytes) + ' / ' + formatBytes(stats.expert_cache_budget_bytes)
  );

  const hits = Number(pager.cache_hits || 0);
  const misses = Number(pager.cache_misses || 0);
  const total = hits + misses;
  setText('metricHit', total ? ((hits / total) * 100).toFixed(1) + '%' : '—');
  setText('metricRouter', String(pager.real_router_requests || 0));
  setText('metricBytes', formatBytes(pager.bytes_read));
  setText('metricAcquire', formatMsFromNs(pager.acquire_ns));
  setText('metricIO', formatMsFromNs(pager.io_ns));
  updateGenerationStats(generation);

  const status = document.getElementById('composerStatus');
  const emptyText = document.getElementById('emptyText');
  if(runtimeReady){
    status.textContent = stats.inference_mode || 'paged_mmb';
    emptyText.textContent = (stats.model_id || 'Modelo local') + ' · ' + (stats.inference_mode || 'paged_mmb');
  }else{
    const reason = stats.backend_error || 'Runtime local indisponível';
    status.textContent = reason;
    emptyText.textContent = reason;
  }
}

async function updateStats(){
  try{
    const response = await apiFetch('/api/stats');
    if(!response.ok){
      let detail = 'HTTP ' + response.status;
      try{
        const payload = await response.json();
        detail = payload.error?.message || payload.error || detail;
      }catch(_){}
      throw new Error(detail);
    }
    setRuntimeUI(await response.json());
  }catch(error){
    setRuntimeUI({
      ready:false,
      inference_mode:'unavailable',
      backend_error:error?.message || 'Falha ao consultar o runtime'
    });
  }
}

function showMessages(){
  emptyEl.hidden = true;
  messagesEl.classList.add('active');
}

function distanceFromBottom(){
  return Math.max(0, scrollEl.scrollHeight - scrollEl.clientHeight - scrollEl.scrollTop);
}

function updatePinnedState(){
  if(programmaticScroll) return;
  pinnedToBottom = distanceFromBottom() <= 140;
  scrollLatestEl.hidden = pinnedToBottom;
}

function scrollToBottom({force=false} = {}){
  if(!force && !pinnedToBottom) return;
  requestAnimationFrame(() => {
    programmaticScroll = true;
    scrollEl.scrollTop = scrollEl.scrollHeight;
    pinnedToBottom = true;
    scrollLatestEl.hidden = true;
    requestAnimationFrame(() => { programmaticScroll = false; });
  });
}

scrollEl.addEventListener('scroll', updatePinnedState, {passive:true});

scrollLatestEl.addEventListener('click', () => {
  pinnedToBottom = true;
  scrollToBottom({force:true});
  promptEl.focus();
});

function addMessage(role, text, {forceScroll=false} = {}){
  showMessages();
  const article = document.createElement('article');
  article.className = 'message ' + role;

  const roleEl = document.createElement('div');
  roleEl.className = 'message-role';
  roleEl.textContent = role === 'user' ? 'Você' : 'Qwen';

  const body = document.createElement('div');
  body.className = 'message-body';

  const content = document.createElement('div');
  content.className = 'message-content';
  content.textContent = text;
  body.appendChild(content);

  if(role === 'assistant'){
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'copy-button';
    copy.textContent = 'Copiar';
    copy.addEventListener('click', async () => {
      try{
        await navigator.clipboard.writeText(content.textContent || '');
        copy.textContent = 'Copiado';
        window.setTimeout(() => { copy.textContent = 'Copiar'; }, 1000);
      }catch(_){
        copy.textContent = 'Não foi possível copiar';
      }
    });
    actions.appendChild(copy);
    body.appendChild(actions);
  }

  article.append(roleEl, body);
  messagesEl.appendChild(article);
  scrollToBottom({force:forceScroll});
  return {article, content};
}

function autoGrow(){
  promptEl.style.height = '44px';
  const next = Math.min(promptEl.scrollHeight, 144);
  promptEl.style.height = Math.max(44, next) + 'px';
  promptEl.style.overflowY = promptEl.scrollHeight > 144 ? 'auto' : 'hidden';
}

function updateSendState(){
  if(isStreaming){
    sendEl.disabled = false;
    sendEl.classList.add('stop');
    sendEl.textContent = '■';
    sendEl.setAttribute('aria-label', 'Parar geração');
    sendEl.title = 'Parar';
    return;
  }
  sendEl.classList.remove('stop');
  sendEl.textContent = '↑';
  sendEl.setAttribute('aria-label', 'Enviar mensagem');
  sendEl.title = 'Enviar';
  sendEl.disabled = !runtimeReady || !promptEl.value.trim();
}

function handleSSEEvent(block, assistant){
  const dataLines = block
    .split(/\r?\n/)
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).replace(/^ /, ''));

  if(dataLines.length === 0) return false;

  const raw = dataLines.join('\n').trim();
  if(!raw) return false;
  if(raw === '[DONE]') return true;

  let event;
  try{
    event = JSON.parse(raw);
  }catch(error){
    console.error('Invalid SSE payload from MiniMaxBrain:', raw, error);
    throw new Error('O servidor retornou um evento de streaming inválido.');
  }

  if(event.error){
    throw new Error(event.error.message || event.error.code || 'Erro durante a geração');
  }

  updateGenerationStats(event.mmb_stats);

  const chunk = event.choices?.[0]?.delta?.content || '';
  if(chunk){
    assistant.content.textContent += chunk;
    scrollToBottom();
  }
  return false;
}

function consumeSSEBuffer(buffer, assistant){
  let doneEvent = false;
  while(true){
    const boundary = buffer.match(/\r?\n\r?\n/);
    if(!boundary || boundary.index === undefined) break;

    const block = buffer.slice(0, boundary.index);
    buffer = buffer.slice(boundary.index + boundary[0].length);

    if(handleSSEEvent(block, assistant)){
      doneEvent = true;
      break;
    }
  }
  return {buffer, doneEvent};
}

async function submitMessage(text){
  const epoch = conversationEpoch;
  isStreaming = true;
  abortController = new AbortController();
  lastTPS = null;
  updateGenerationStats({tokens_generated:0});
  updateSendState();

  promptEl.value = '';
  autoGrow();

  pinnedToBottom = true;
  addMessage('user', text, {forceScroll:true});
  history.push({role:'user', content:text});
  const assistant = addMessage('assistant', '', {forceScroll:true});

  try{
    const response = await apiFetch('/v1/chat/completions', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      signal:abortController.signal,
      body:JSON.stringify({
        messages:history,
        stream:true,
        max_tokens:128
      })
    });

    if(!response.ok){
      let detail = 'HTTP ' + response.status;
      try{
        const payload = await response.json();
        detail = payload.error?.message || payload.error || detail;
      }catch(_){}
      throw new Error(detail);
    }

    if(!response.body) throw new Error('Streaming não suportado pelo navegador');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let doneEvent = false;

    while(!doneEvent){
      const {value, done} = await reader.read();
      if(done) break;

      buffer += decoder.decode(value, {stream:true});
      const consumed = consumeSSEBuffer(buffer, assistant);
      buffer = consumed.buffer;
      doneEvent = consumed.doneEvent;
    }

    buffer += decoder.decode();
    if(buffer.trim() && !doneEvent){
      const consumed = consumeSSEBuffer(buffer + '\n\n', assistant);
      doneEvent = consumed.doneEvent;
    }

    if(epoch !== conversationEpoch) return;
    const answer = assistant.content.textContent || '';
    if(answer){
      history.push({role:'assistant', content:answer});
    }else if(!doneEvent){
      throw new Error('A geração terminou sem conteúdo.');
    }
  }catch(error){
    if(epoch !== conversationEpoch) return;
    if(error?.name === 'AbortError'){
      const partial = assistant.content.textContent || '';
      if(partial){
        history.push({role:'assistant', content:partial});
        setText('composerStatus', 'Geração interrompida');
      }else{
        assistant.article.remove();
      }
    }else{
      assistant.article.classList.add('error');
      assistant.content.textContent =
        'Não foi possível concluir: ' + (error?.message || String(error));
      setText('composerStatus', 'Falha na geração');
    }
  }finally{
    if(epoch === conversationEpoch){
      isStreaming = false;
      abortController = null;
      promptEl.disabled = !runtimeReady;
      updateSendState();
      promptEl.focus();
      await updateStats();
    }
  }
}

formEl.addEventListener('submit', event => {
  event.preventDefault();
  if(isStreaming){
    abortController?.abort();
    return;
  }
  const text = promptEl.value.trim();
  if(!text || !runtimeReady) return;
  submitMessage(text);
});

promptEl.addEventListener('input', () => {
  autoGrow();
  updateSendState();
});

promptEl.addEventListener('keydown', event => {
  if(event.key === 'Enter' && !event.shiftKey){
    event.preventDefault();
    if(!sendEl.disabled) formEl.requestSubmit();
  }
});

newChatEl.addEventListener('click', () => {
  if(isStreaming){
    const ok = window.confirm('Parar a geração atual e iniciar uma nova conversa?');
    if(!ok) return;
  }
  conversationEpoch += 1;
  abortController?.abort();
  abortController = null;
  isStreaming = false;
  promptEl.disabled = !runtimeReady;
  history.length = 0;
  messagesEl.replaceChildren();
  messagesEl.classList.remove('active');
  emptyEl.hidden = false;
  lastTPS = null;
  updateGenerationStats({tokens_generated:0});
  pinnedToBottom = true;
  scrollLatestEl.hidden = true;
  scrollEl.scrollTop = 0;
  promptEl.value = '';
  autoGrow();
  updateSendState();
  promptEl.focus();
});

document.addEventListener('click', event => {
  const details = document.getElementById('technical');
  if(details.open && !details.contains(event.target)) details.open = false;
});

autoGrow();
updateStats();
window.setInterval(() => {
  if(!isStreaming) updateStats();
}, 5000);
</script>
</body>
</html>"""


class MMBHTTPRequestHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP adapter over :class:`MMBRuntime`."""

    engine: MMBRuntime

    def log_message(self, format: str, *args: Any) -> None:
        # Keep CLI output readable; operational failures are returned explicitly.
        return

    def _send_json(self, status: int, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: int, exc: Exception) -> None:
        code = getattr(exc, "code", "INTERNAL_ERROR")
        self._send_json(status, {"error": {"code": code, "message": str(exc)}})

    def _authorized(self) -> bool:
        expected = self.engine.config.server.api_token
        if expected is None:
            return True
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        return hmac.compare_digest(auth[7:], expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": {"code": "UNAUTHORIZED", "message": "valid Bearer token required"}},
        )
        return False

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise InferenceError("invalid Content-Length") from exc
        if length <= 0:
            raise InferenceError("request body is required")
        limit = int(self.engine.config.server.max_request_bytes)
        if length > limit:
            raise InferenceError(f"request body exceeds {limit} bytes")
        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InferenceError(f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise InferenceError("JSON body must be an object")
        return body

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/"}:
            body = WEB_UI_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/health":
            stats = self.engine.stats()
            status = HTTPStatus.OK if stats["ready"] else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(status, stats)
            return

        if path == "/v1/models":
            if not self._require_auth():
                return
            self._send_json(HTTPStatus.OK, {
                "object": "list",
                "data": [{
                    "id": self.engine.model_map.model_id or "minimaxbrain",
                    "object": "model",
                    "owned_by": "minimaxbrain",
                    "ready": self.engine.ready,
                    "inference_mode": self.engine.inference_mode.value,
                }],
            })
            return

        if path == "/api/stats":
            if not self._require_auth():
                return
            stats = self.engine.stats()
            rss = stats.get("backend_rss_bytes")
            stats["backend_rss_str"] = format_bytes(rss) if isinstance(rss, int) else None
            stats["ram_budget_str"] = format_bytes(self.engine.config.memory.ram_budget_bytes)
            self._send_json(HTTPStatus.OK, stats)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
            return

        if not self._require_auth():
            return

        if not self.engine.ready:
            self._send_error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                BackendUnavailableError(
                    self.engine.backend_error or "inference backend is unavailable"
                ),
            )
            return

        try:
            body = self._read_json()
            messages = body.get("messages")
            stream = bool(body.get("stream", False))
            temperature = float(body.get("temperature", 0.7))
            top_p = float(body.get("top_p", 0.9))
            top_k = int(body.get("top_k", 40))
            max_tokens = int(body.get("max_tokens", 64))
        except (MMBError, TypeError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, exc)
            return

        try:
            max_tokens, temperature, top_p, top_k = self.engine.validate_generation_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            messages = self.engine._validate_messages(messages)
        except MMBError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, exc)
            return

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model_id = self.engine.model_map.model_id or "minimaxbrain"

        if not stream:
            try:
                chunks: list[str] = []
                last_stats: dict[str, Any] = {}
                for chunk, stats in self.engine.stream_chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ):
                    chunks.append(chunk)
                    last_stats = stats
            except BackendUnavailableError as exc:
                self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, exc)
                return
            except MMBError as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)
                return

            self._send_json(HTTPStatus.OK, {
                "id": chat_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(chunks)},
                    "finish_reason": "stop",
                }],
                "mmb_stats": last_stats,
            })
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for chunk, stats in self.engine.stream_chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ):
                event = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                    "mmb_stats": stats,
                }
                raw = (
                    "data: "
                    + json.dumps(event, ensure_ascii=False)
                    + "\n\n"
                ).encode("utf-8")
                self.wfile.write(raw)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except MMBError as exc:
            error_event = {
                "error": {"code": exc.code, "message": str(exc)}
            }
            try:
                self.wfile.write(
                    ("data: " + json.dumps(error_event, ensure_ascii=False) + "\n\n").encode("utf-8")
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        finally:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_mmb_server(
    engine: MMBRuntime,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadedHTTPServer:
    normalized_host = str(host).strip().lower()
    if normalized_host not in {"127.0.0.1", "localhost", "::1"}:
        if engine.config.server.api_token is None:
            raise ConfigurationError(
                "server.api_token is required when exposing Web/API outside loopback"
            )
    handler = type(
        "BoundMMBHTTPRequestHandler",
        (MMBHTTPRequestHandler,),
        {"engine": engine},
    )
    server = ThreadedHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="mmb-http", daemon=True)
    thread.start()
    return server
