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
  color-scheme:light;
  --bg:#f3f7fc;
  --bg-elevated:#ffffff;
  --surface:rgba(255,255,255,.76);
  --surface-solid:#ffffff;
  --surface-subtle:#eaf1f9;
  --surface-accent:#e7f3ff;
  --text:#0b1628;
  --text-secondary:#4c5c70;
  --text-tertiary:#748398;
  --line:rgba(24,52,84,.12);
  --line-strong:rgba(24,52,84,.19);
  --accent:#087bfa;
  --accent-strong:#0068db;
  --accent-contrast:#ffffff;
  --teal:#0d9f96;
  --success:#168a63;
  --warning:#b26a00;
  --danger:#d13d50;
  --shadow-sm:0 1px 2px rgba(26,43,66,.06),0 8px 24px rgba(41,71,105,.08);
  --shadow-lg:0 18px 60px rgba(20,47,78,.16);
  --radius-sm:10px;
  --radius-md:14px;
  --radius-lg:20px;
  --radius-xl:28px;
  --radius-pill:999px;
  --topbar-h:64px;
  --content-max:920px;
  --ease:cubic-bezier(.2,.8,.2,1);
}
html[data-theme="dark"]{
  color-scheme:dark;
  --bg:#07111f;
  --bg-elevated:#0d1929;
  --surface:rgba(12,25,43,.78);
  --surface-solid:#0d1a2c;
  --surface-subtle:#111f33;
  --surface-accent:#102d4b;
  --text:#f6f9ff;
  --text-secondary:#b0bed0;
  --text-tertiary:#7f91a8;
  --line:rgba(177,205,236,.12);
  --line-strong:rgba(177,205,236,.2);
  --accent:#4aa8ff;
  --accent-strong:#72bdff;
  --accent-contrast:#05111e;
  --teal:#43d5c7;
  --success:#58d6a8;
  --warning:#f0b766;
  --danger:#ff7182;
  --shadow-sm:0 1px 2px rgba(0,0,0,.18),0 14px 32px rgba(0,0,0,.2);
  --shadow-lg:0 24px 70px rgba(0,0,0,.38);
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  overflow:hidden;
  background:
    linear-gradient(180deg,color-mix(in srgb,var(--accent) 4%,var(--bg)) 0,var(--bg) 170px);
  color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Segoe UI",system-ui,sans-serif;
  font-size:15px;
  line-height:1.5;
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
}
button,textarea{font:inherit}
button{color:inherit}
button:focus-visible,textarea:focus-visible{
  outline:3px solid color-mix(in srgb,var(--accent) 36%,transparent);
  outline-offset:2px;
}
button{-webkit-tap-highlight-color:transparent}
svg{display:block}
.app{
  height:100%;
  display:grid;
  grid-template-rows:var(--topbar-h) minmax(0,1fr);
}
.topbar{
  position:relative;
  z-index:20;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  padding:0 max(18px,env(safe-area-inset-right)) 0 max(18px,env(safe-area-inset-left));
  border-bottom:1px solid var(--line);
  background:color-mix(in srgb,var(--surface) 92%,transparent);
  backdrop-filter:blur(22px) saturate(145%);
  -webkit-backdrop-filter:blur(22px) saturate(145%);
}
.brand{
  min-width:0;
  display:flex;
  align-items:center;
  gap:11px;
  font-weight:680;
  letter-spacing:-.015em;
}
.brand-mark{
  width:31px;height:31px;border-radius:10px;
  display:grid;place-items:center;
  color:white;
  background:linear-gradient(145deg,#1685ff 0%,#0ba9d4 56%,#1cb59d 100%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.32),0 6px 16px rgba(8,123,250,.2);
}
.brand-copy{display:flex;align-items:baseline;gap:7px;min-width:0}
.brand-name{white-space:nowrap;font-size:15px}
.brand-version{
  color:var(--text-tertiary);
  font-size:11px;
  font-weight:600;
  letter-spacing:.02em;
}
.toolbar{display:flex;align-items:center;gap:8px}
.icon-button,.quiet-button,.runtime-toggle{
  border:1px solid transparent;
  background:transparent;
  cursor:pointer;
  transition:background 160ms var(--ease),border-color 160ms var(--ease),transform 120ms var(--ease);
}
.icon-button:active,.quiet-button:active,.runtime-toggle:active{transform:scale(.97)}
.icon-button{
  width:38px;height:38px;border-radius:12px;
  display:grid;place-items:center;
  color:var(--text-secondary);
}
.icon-button:hover{background:var(--surface-subtle);color:var(--text)}
.quiet-button{
  min-height:38px;
  border-radius:12px;
  padding:0 12px;
  display:flex;align-items:center;gap:8px;
  color:var(--text-secondary);
  font-weight:600;
}
.quiet-button:hover{background:var(--surface-subtle);color:var(--text)}
.runtime-wrap{position:relative}
.runtime-toggle{
  height:38px;
  padding:0 12px;
  border-radius:var(--radius-pill);
  display:flex;
  align-items:center;
  gap:8px;
  background:color-mix(in srgb,var(--surface-subtle) 74%,transparent);
  color:var(--text-secondary);
  font-size:13px;
  font-weight:600;
}
.runtime-toggle:hover{background:var(--surface-subtle)}
.status-dot{
  width:8px;height:8px;border-radius:50%;
  background:var(--text-tertiary);
  box-shadow:0 0 0 4px color-mix(in srgb,var(--text-tertiary) 10%,transparent);
}
.status-dot.ready{background:var(--success);box-shadow:0 0 0 4px color-mix(in srgb,var(--success) 13%,transparent)}
.status-dot.error{background:var(--danger);box-shadow:0 0 0 4px color-mix(in srgb,var(--danger) 13%,transparent)}
.chevron{transition:transform 180ms var(--ease)}
.runtime-toggle[aria-expanded="true"] .chevron{transform:rotate(180deg)}
.runtime-panel{
  position:absolute;
  top:46px;
  right:0;
  width:min(360px,calc(100vw - 24px));
  padding:14px;
  border:1px solid var(--line-strong);
  border-radius:20px;
  background:color-mix(in srgb,var(--surface-solid) 88%,transparent);
  backdrop-filter:blur(26px) saturate(150%);
  -webkit-backdrop-filter:blur(26px) saturate(150%);
  box-shadow:var(--shadow-lg);
  opacity:0;
  transform:translateY(-6px) scale(.985);
  transform-origin:top right;
  pointer-events:none;
  transition:opacity 160ms var(--ease),transform 180ms var(--ease);
}
.runtime-panel.open{opacity:1;transform:none;pointer-events:auto}
.runtime-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:3px 3px 11px;
}
.runtime-title{font-size:14px;font-weight:700;letter-spacing:-.01em}
.runtime-badge{
  padding:5px 8px;border-radius:var(--radius-pill);
  background:var(--surface-subtle);
  color:var(--text-secondary);
  font-size:11px;font-weight:700;
}
.runtime-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.metric{
  min-width:0;
  padding:11px 12px;
  background:var(--surface-subtle);
  border:1px solid var(--line);
  border-radius:14px;
}
.metric-label{font-size:11px;color:var(--text-tertiary);margin-bottom:3px}
.metric-value{
  min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-size:13px;font-weight:650;color:var(--text)
}
.runtime-note{
  margin:10px 3px 1px;
  color:var(--text-tertiary);
  font-size:12px;
  line-height:1.45;
}
.main{
  min-height:0;
  display:flex;
  justify-content:center;
  position:relative;
}
.conversation{
  width:min(var(--content-max),100%);
  min-height:0;
  display:grid;
  grid-template-rows:minmax(0,1fr) auto;
  padding:0 24px max(14px,env(safe-area-inset-bottom));
}
.scroll-area{
  min-height:0;
  overflow:auto;
  scrollbar-gutter:stable;
  overscroll-behavior:contain;
  padding:34px 4px 28px;
  scroll-behavior:smooth;
}
.scroll-area::-webkit-scrollbar{width:10px}
.scroll-area::-webkit-scrollbar-thumb{
  background:color-mix(in srgb,var(--text-tertiary) 25%,transparent);
  border:3px solid transparent;
  background-clip:padding-box;
  border-radius:999px;
}
.welcome{
  min-height:100%;
  display:grid;
  place-items:center;
  text-align:center;
  padding:30px 0 68px;
}
.welcome-inner{max-width:610px}
.welcome-kicker{
  display:inline-flex;align-items:center;gap:8px;
  color:var(--accent);
  font-weight:650;
  font-size:13px;
  margin-bottom:14px;
}
.welcome h1{
  margin:0;
  color:var(--text);
  font-size:clamp(31px,5vw,48px);
  line-height:1.03;
  letter-spacing:-.045em;
  font-weight:720;
}
.welcome p{
  margin:16px auto 0;
  max-width:540px;
  color:var(--text-secondary);
  font-size:16px;
  line-height:1.6;
}
.welcome-status{
  margin:24px auto 0;
  width:fit-content;
  max-width:100%;
  display:flex;
  align-items:center;
  gap:10px;
  padding:9px 12px;
  border-radius:var(--radius-pill);
  background:var(--surface);
  border:1px solid var(--line);
  color:var(--text-secondary);
  box-shadow:var(--shadow-sm);
  font-size:12px;
}
#messages{
  display:none;
  flex-direction:column;
  gap:26px;
  width:100%;
}
#messages.active{display:flex}
.message{
  display:flex;
  gap:12px;
  width:100%;
  animation:message-in 260ms var(--ease) both;
}
.message.assistant{align-items:flex-start}
.message.user{justify-content:flex-end}
.assistant-mark{
  flex:0 0 auto;
  width:28px;height:28px;
  margin-top:1px;
  border-radius:9px;
  display:grid;place-items:center;
  color:var(--accent-contrast);
  background:var(--accent);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.26);
}
.message-body{min-width:0}
.message.assistant .message-body{
  max-width:min(760px,calc(100% - 40px));
  padding-top:2px;
}
.message.user .message-body{
  max-width:min(72%,680px);
  padding:10px 14px;
  border-radius:18px 18px 6px 18px;
  background:var(--surface-accent);
  border:1px solid color-mix(in srgb,var(--accent) 16%,var(--line));
}
.message-content{
  white-space:pre-wrap;
  overflow-wrap:anywhere;
  color:var(--text);
  font-size:15.5px;
  line-height:1.62;
}
.message.assistant .message-content:empty::after{
  content:"";
  display:inline-block;
  width:7px;height:16px;
  vertical-align:-3px;
  border-radius:3px;
  background:var(--accent);
  animation:cursor-pulse 900ms ease-in-out infinite;
}
.message.error .message-body{
  padding:11px 13px;
  border-radius:14px;
  background:color-mix(in srgb,var(--danger) 9%,var(--surface-solid));
  border:1px solid color-mix(in srgb,var(--danger) 28%,transparent);
}
.message.error .message-content{color:var(--danger)}
.message-actions{
  margin-top:6px;
  min-height:28px;
  display:flex;
  align-items:center;
  gap:4px;
}
.message-action{
  width:28px;height:28px;
  border:0;border-radius:9px;
  background:transparent;
  color:var(--text-tertiary);
  cursor:pointer;
  opacity:0;
  transition:opacity 140ms var(--ease),background 140ms var(--ease),color 140ms var(--ease);
}
.message:hover .message-action,.message-action:focus-visible{opacity:1}
.message-action:hover{background:var(--surface-subtle);color:var(--text-secondary)}
.composer-zone{
  position:relative;
  z-index:10;
  padding:0 0 2px;
}
.composer-zone::before{
  content:"";
  position:absolute;
  left:-24px;right:-24px;bottom:-16px;
  height:120px;
  z-index:-1;
  pointer-events:none;
  background:linear-gradient(to bottom,transparent,var(--bg) 55%);
}
.composer{
  position:relative;
  display:flex;
  align-items:flex-end;
  gap:9px;
  padding:8px 8px 8px 14px;
  border:1px solid var(--line-strong);
  border-radius:22px;
  background:color-mix(in srgb,var(--surface-solid) 82%,transparent);
  backdrop-filter:blur(24px) saturate(145%);
  -webkit-backdrop-filter:blur(24px) saturate(145%);
  box-shadow:var(--shadow-sm);
  transition:border-color 160ms var(--ease),box-shadow 160ms var(--ease),transform 160ms var(--ease);
}
.composer:focus-within{
  border-color:color-mix(in srgb,var(--accent) 42%,var(--line));
  box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 8%,transparent),var(--shadow-sm);
}
#prompt{
  flex:1;
  min-width:0;
  min-height:38px;
  max-height:180px;
  resize:none;
  overflow:auto;
  padding:8px 2px 7px;
  border:0;
  outline:0;
  background:transparent;
  color:var(--text);
  line-height:1.5;
}
#prompt::placeholder{color:var(--text-tertiary)}
.send-button{
  flex:0 0 auto;
  width:40px;height:40px;
  border:0;border-radius:13px;
  display:grid;place-items:center;
  background:var(--accent);
  color:var(--accent-contrast);
  cursor:pointer;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.24),0 5px 14px color-mix(in srgb,var(--accent) 22%,transparent);
  transition:transform 130ms var(--ease),background 160ms var(--ease),opacity 160ms var(--ease);
}
.send-button:hover:not(:disabled){background:var(--accent-strong)}
.send-button:active:not(:disabled){transform:scale(.94)}
.send-button:disabled{opacity:.34;cursor:default;box-shadow:none}
.send-button.stop{background:var(--text);color:var(--bg)}
.send-button .stop-icon{display:none}
.send-button.stop .send-icon{display:none}
.send-button.stop .stop-icon{display:block}
.composer-meta{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:12px;
  padding:7px 5px 0;
  color:var(--text-tertiary);
  font-size:11px;
}
.composer-meta .live{display:flex;align-items:center;gap:6px}
.tiny-dot{width:6px;height:6px;border-radius:50%;background:var(--text-tertiary)}
.tiny-dot.ready{background:var(--success)}
.tiny-dot.error{background:var(--danger)}
.toast{
  position:fixed;
  left:50%;
  bottom:max(26px,env(safe-area-inset-bottom));
  z-index:50;
  transform:translate(-50%,12px);
  padding:9px 13px;
  border-radius:var(--radius-pill);
  border:1px solid var(--line-strong);
  background:var(--surface-solid);
  color:var(--text);
  box-shadow:var(--shadow-lg);
  font-size:12px;font-weight:600;
  opacity:0;pointer-events:none;
  transition:opacity 160ms var(--ease),transform 180ms var(--ease);
}
.toast.show{opacity:1;transform:translate(-50%,0)}
@keyframes message-in{
  from{opacity:0;transform:translateY(7px)}
  to{opacity:1;transform:none}
}
@keyframes cursor-pulse{
  0%,100%{opacity:.25}
  50%{opacity:1}
}
@media(max-width:700px){
  :root{--topbar-h:60px}
  .topbar{padding-left:14px;padding-right:12px}
  .brand-version,.quiet-button span,.runtime-toggle .runtime-label{display:none}
  .quiet-button{width:38px;padding:0;justify-content:center}
  .runtime-toggle{width:38px;padding:0;justify-content:center}
  .runtime-toggle .chevron{display:none}
  .conversation{padding-left:14px;padding-right:14px}
  .scroll-area{padding-top:24px}
  .message.user .message-body{max-width:86%}
  .welcome{padding-bottom:46px}
  .composer-zone::before{left:-14px;right:-14px}
  .composer-meta .shortcut{display:none}
}
@media(max-width:420px){
  .toolbar{gap:3px}
  .brand-copy{gap:0}
  .brand-name{font-size:14px}
  .message{gap:9px}
  .assistant-mark{width:26px;height:26px;border-radius:8px}
  .message.assistant .message-body{max-width:calc(100% - 35px)}
}
@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{
    scroll-behavior:auto!important;
    animation-duration:.001ms!important;
    animation-iteration-count:1!important;
    transition-duration:.001ms!important;
  }
}
@media(prefers-reduced-transparency:reduce){
  .topbar,.runtime-panel,.composer{
    backdrop-filter:none;
    -webkit-backdrop-filter:none;
    background:var(--surface-solid);
  }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand" aria-label="MiniMaxBrain">
      <div class="brand-mark" aria-hidden="true">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
          <path d="M7 16.5V8.7c0-1 .8-1.7 1.7-1.7.7 0 1.3.4 1.6 1l1.7 4 1.7-4c.3-.6.9-1 1.6-1 .9 0 1.7.7 1.7 1.7v7.8" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="brand-copy">
        <span class="brand-name">MiniMaxBrain</span>
        <span class="brand-version">0.3</span>
      </div>
    </div>

    <nav class="toolbar" aria-label="Ações do chat">
      <button class="quiet-button" id="newChat" type="button" title="Novo chat">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <span>Novo chat</span>
      </button>

      <div class="runtime-wrap">
        <button class="runtime-toggle" id="runtimeToggle" type="button" aria-expanded="false" aria-controls="runtimePanel">
          <span class="status-dot" id="statusDot" aria-hidden="true"></span>
          <span class="runtime-label" id="statusLabel">Conectando</span>
          <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="m7 10 5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>

        <aside class="runtime-panel" id="runtimePanel" aria-label="Detalhes do runtime">
          <div class="runtime-head">
            <div class="runtime-title">Runtime local</div>
            <div class="runtime-badge" id="runtimeBadge">verificando</div>
          </div>
          <div class="runtime-grid">
            <div class="metric">
              <div class="metric-label">Modelo</div>
              <div class="metric-value" id="metricModel">—</div>
            </div>
            <div class="metric">
              <div class="metric-label">Modo</div>
              <div class="metric-value" id="metricMode">—</div>
            </div>
            <div class="metric">
              <div class="metric-label">RAM do backend</div>
              <div class="metric-value" id="metricRam">—</div>
            </div>
            <div class="metric">
              <div class="metric-label">Orçamento RAM</div>
              <div class="metric-value" id="metricBudget">—</div>
            </div>
          </div>
          <div class="runtime-note" id="runtimeNote">Consultando o backend local…</div>
        </aside>
      </div>

      <button class="icon-button" id="themeToggle" type="button" aria-label="Alternar aparência" title="Alternar aparência">
        <svg id="themeIcon" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 3a9 9 0 1 0 9 9c0-.4 0-.8-.1-1.2A7 7 0 0 1 13.2 3H12Z" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/>
        </svg>
      </button>
    </nav>
  </header>

  <main class="main">
    <section class="conversation" aria-label="Conversa">
      <div class="scroll-area" id="scrollArea">
        <div class="welcome" id="welcome">
          <div class="welcome-inner">
            <div class="welcome-kicker">
              <span class="status-dot" id="welcomeDot" aria-hidden="true"></span>
              <span id="welcomeKicker">Verificando o runtime local</span>
            </div>
            <h1>Converse com seu modelo,<br>sem ruído.</h1>
            <p id="welcomeText">O MiniMaxBrain mantém a experiência focada na conversa enquanto o runtime local cuida da inferência.</p>
            <div class="welcome-status" id="welcomeStatus">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M8 12.5 10.5 15 16 9.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.7"/>
              </svg>
              <span id="welcomeStatusText">Preparando conexão…</span>
            </div>
          </div>
        </div>
        <div id="messages" role="log" aria-live="polite" aria-relevant="additions text"></div>
      </div>

      <div class="composer-zone">
        <form class="composer" id="form">
          <textarea id="prompt" rows="1" autocomplete="off" spellcheck="true" aria-label="Mensagem" placeholder="Pergunte alguma coisa…"></textarea>
          <button class="send-button" id="send" type="submit" disabled aria-label="Enviar mensagem" title="Enviar">
            <svg class="send-icon" width="19" height="19" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M12 19V5m0 0L6.5 10.5M12 5l5.5 5.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <svg class="stop-icon" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <rect x="6" y="6" width="12" height="12" rx="2"/>
            </svg>
          </button>
        </form>
        <div class="composer-meta">
          <div class="live">
            <span class="tiny-dot" id="composerDot" aria-hidden="true"></span>
            <span id="composerStatus">Conectando ao runtime</span>
          </div>
          <span class="shortcut">Enter envia · Shift+Enter quebra linha</span>
        </div>
      </div>
    </section>
  </main>
</div>

<div class="toast" id="toast" role="status" aria-live="polite"></div>

<script>
const messagesEl = document.getElementById('messages');
const scrollArea = document.getElementById('scrollArea');
const welcomeEl = document.getElementById('welcome');
const promptEl = document.getElementById('prompt');
const sendEl = document.getElementById('send');
const formEl = document.getElementById('form');
const history = [];

let apiToken = sessionStorage.getItem('mmb_api_token') || '';
let runtimeReady = false;
let isStreaming = false;
let abortController = null;
let toastTimer = null;

const icons = {
  copy: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="8" y="8" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" stroke="currentColor" stroke-width="1.8"/></svg>',
  check: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m6 12.5 4 4L18 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
};

function showToast(text){
  const el = document.getElementById('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 1500);
}

async function apiFetch(url, options = {}){
  options.headers = options.headers || {};
  if(apiToken) options.headers['Authorization'] = 'Bearer ' + apiToken;
  let response = await fetch(url, options);
  if(response.status === 401){
    const entered = window.prompt('Token da API do MiniMaxBrain:');
    if(entered){
      apiToken = entered.trim();
      sessionStorage.setItem('mmb_api_token', apiToken);
      options.headers['Authorization'] = 'Bearer ' + apiToken;
      response = await fetch(url, options);
    }
  }
  return response;
}

function setMessageMode(active){
  welcomeEl.hidden = active;
  messagesEl.classList.toggle('active', active);
}

function scrollToBottom(){
  requestAnimationFrame(() => {
    scrollArea.scrollTop = scrollArea.scrollHeight;
  });
}

function addAssistantActions(article, contentEl){
  const actions = document.createElement('div');
  actions.className = 'message-actions';

  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'message-action';
  copy.setAttribute('aria-label', 'Copiar resposta');
  copy.title = 'Copiar';
  copy.innerHTML = icons.copy;
  copy.addEventListener('click', async () => {
    try{
      await navigator.clipboard.writeText(contentEl.textContent || '');
      copy.innerHTML = icons.check;
      showToast('Resposta copiada');
      setTimeout(() => { copy.innerHTML = icons.copy; }, 1200);
    }catch(_){
      showToast('Não foi possível copiar');
    }
  });

  actions.appendChild(copy);
  article.querySelector('.message-body').appendChild(actions);
}

function addMessage(role, text, options = {}){
  setMessageMode(true);

  const article = document.createElement('article');
  article.className = 'message ' + role + (options.error ? ' error' : '');

  if(role === 'assistant'){
    const mark = document.createElement('div');
    mark.className = 'assistant-mark';
    mark.setAttribute('aria-hidden', 'true');
    mark.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M7 16.5V8.7c0-1 .8-1.7 1.7-1.7.7 0 1.3.4 1.6 1l1.7 4 1.7-4c.3-.6.9-1 1.6-1 .9 0 1.7.7 1.7 1.7v7.8" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    article.appendChild(mark);
  }

  const body = document.createElement('div');
  body.className = 'message-body';
  const content = document.createElement('div');
  content.className = 'message-content';
  content.textContent = text;
  body.appendChild(content);
  article.appendChild(body);
  messagesEl.appendChild(article);
  scrollToBottom();

  return {article, content};
}

function finalizeAssistant(message){
  if(!message || message.article.classList.contains('error')) return;
  if(!message.article.querySelector('.message-actions')) {
    addAssistantActions(message.article, message.content);
  }
}

function autoGrow(){
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 180) + 'px';
}

function updateSendState(){
  if(isStreaming){
    sendEl.disabled = false;
    sendEl.classList.add('stop');
    sendEl.setAttribute('aria-label', 'Parar geração');
    sendEl.title = 'Parar';
    return;
  }
  sendEl.classList.remove('stop');
  sendEl.setAttribute('aria-label', 'Enviar mensagem');
  sendEl.title = 'Enviar';
  sendEl.disabled = !promptEl.value.trim() || !runtimeReady;
}

function humanBytes(value){
  if(typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—';
  const units = ['B','KiB','MiB','GiB','TiB'];
  let n = value, i = 0;
  while(n >= 1024 && i < units.length - 1){ n /= 1024; i++; }
  const decimals = n >= 10 || i === 0 ? 0 : 1;
  return n.toFixed(decimals) + ' ' + units[i];
}

function setRuntimeUI(data){
  runtimeReady = Boolean(data && data.ready);

  const dot = document.getElementById('statusDot');
  const welcomeDot = document.getElementById('welcomeDot');
  const composerDot = document.getElementById('composerDot');
  [dot, welcomeDot, composerDot].forEach(el => {
    el.classList.remove('ready','error');
    el.classList.add(runtimeReady ? 'ready' : 'error');
  });

  const mode = data?.inference_mode || 'unavailable';
  const model = data?.model_id || 'MiniMaxBrain';
  const ram = data?.backend_rss_str || humanBytes(data?.backend_rss_bytes);
  const budget = data?.ram_budget_str || humanBytes(data?.ram_budget_bytes);

  document.getElementById('statusLabel').textContent = runtimeReady ? 'Pronto' : 'Indisponível';
  document.getElementById('runtimeBadge').textContent = runtimeReady ? 'online' : 'atenção';
  document.getElementById('metricModel').textContent = model;
  document.getElementById('metricMode').textContent = mode;
  document.getElementById('metricRam').textContent = ram || '—';
  document.getElementById('metricBudget').textContent = budget || '—';

  const paged = Boolean(data?.paged_experts_used);
  const runtimeNote = document.getElementById('runtimeNote');
  if(runtimeReady){
    runtimeNote.textContent = paged
      ? 'Inferência ativa com especialistas paginados pelo runtime local.'
      : 'Runtime local pronto para inferência.';
    document.getElementById('welcomeKicker').textContent = model + ' está pronto';
    document.getElementById('welcomeStatusText').textContent = 'Inferência local · ' + mode;
    document.getElementById('welcomeText').textContent = 'Uma interface calma para conversar com ' + model + ', com o estado técnico disponível apenas quando você precisar.';
    document.getElementById('composerStatus').textContent = 'Runtime pronto';
  }else{
    const reason = data?.backend_error || 'backend não inicializado';
    runtimeNote.textContent = reason;
    document.getElementById('welcomeKicker').textContent = 'Runtime indisponível';
    document.getElementById('welcomeStatusText').textContent = reason;
    document.getElementById('welcomeText').textContent = 'A interface está pronta, mas o backend de inferência precisa estar disponível antes de enviar mensagens.';
    document.getElementById('composerStatus').textContent = 'Backend indisponível';
  }
  updateSendState();
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
    const data = await response.json();
    setRuntimeUI(data);
  }catch(error){
    setRuntimeUI({
      ready:false,
      inference_mode:'unavailable',
      backend_error:error?.message || 'Falha ao consultar o runtime'
    });
  }
}

function handleSSELine(line, message){
  if(!line.startsWith('data:')) return false;
  const raw = line.slice(5).trim();
  if(!raw) return false;
  if(raw === '[DONE]') return true;

  const event = JSON.parse(raw);
  if(event.error){
    throw new Error(event.error.message || event.error.code || 'Erro durante a geração');
  }
  const chunk = event.choices?.[0]?.delta?.content || '';
  if(chunk){
    message.content.textContent += chunk;
    scrollToBottom();
  }
  return false;
}

async function submitMessage(text){
  isStreaming = true;
  abortController = new AbortController();
  updateSendState();

  promptEl.value = '';
  autoGrow();

  addMessage('user', text);
  history.push({role:'user', content:text});
  const assistant = addMessage('assistant', '');

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
      const lines = buffer.split('
');
      buffer = lines.pop() || '';
      for(const line of lines){
        if(handleSSELine(line, assistant)){
          doneEvent = true;
          break;
        }
      }
    }

    buffer += decoder.decode();
    if(buffer.trim() && !doneEvent){
      for(const line of buffer.split('
')){
        if(handleSSELine(line, assistant)) break;
      }
    }

    const answer = assistant.content.textContent || '';
    if(answer) history.push({role:'assistant', content:answer});
    finalizeAssistant(assistant);
  }catch(error){
    if(error?.name === 'AbortError'){
      const partial = assistant.content.textContent || '';
      if(partial){
        history.push({role:'assistant', content:partial});
        finalizeAssistant(assistant);
        showToast('Geração interrompida');
      }else{
        assistant.article.remove();
        showToast('Geração interrompida');
      }
    }else{
      assistant.article.classList.add('error');
      assistant.content.textContent = 'Não foi possível concluir: ' + (error?.message || String(error));
    }
  }finally{
    isStreaming = false;
    abortController = null;
    updateSendState();
    promptEl.focus();
    updateStats();
  }
}

formEl.addEventListener('submit', async event => {
  event.preventDefault();
  if(isStreaming){
    abortController?.abort();
    return;
  }
  const text = promptEl.value.trim();
  if(!text || !runtimeReady) return;
  await submitMessage(text);
});

promptEl.addEventListener('input', () => {
  autoGrow();
  updateSendState();
});
promptEl.addEventListener('keydown', event => {
  if(event.key === 'Enter' && !event.shiftKey){
    event.preventDefault();
    if(isStreaming) return;
    if(promptEl.value.trim() && runtimeReady) formEl.requestSubmit();
  }
});

document.getElementById('newChat').addEventListener('click', () => {
  if(isStreaming) abortController?.abort();
  history.length = 0;
  messagesEl.replaceChildren();
  setMessageMode(false);
  promptEl.value = '';
  autoGrow();
  updateSendState();
  promptEl.focus();
  showToast('Novo chat iniciado');
});

const runtimeToggle = document.getElementById('runtimeToggle');
const runtimePanel = document.getElementById('runtimePanel');
function setRuntimePanel(open){
  runtimePanel.classList.toggle('open', open);
  runtimeToggle.setAttribute('aria-expanded', String(open));
}
runtimeToggle.addEventListener('click', event => {
  event.stopPropagation();
  setRuntimePanel(!runtimePanel.classList.contains('open'));
});
runtimePanel.addEventListener('click', event => event.stopPropagation());
document.addEventListener('click', () => setRuntimePanel(false));
document.addEventListener('keydown', event => {
  if(event.key === 'Escape') setRuntimePanel(false);
});

const themeToggle = document.getElementById('themeToggle');
const mediaDark = window.matchMedia('(prefers-color-scheme: dark)');
function preferredTheme(){
  return localStorage.getItem('mmb_theme') || (mediaDark.matches ? 'dark' : 'light');
}
function applyTheme(theme){
  document.documentElement.dataset.theme = theme;
  themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Usar aparência clara' : 'Usar aparência escura');
  themeToggle.title = theme === 'dark' ? 'Aparência clara' : 'Aparência escura';
}
applyTheme(preferredTheme());

themeToggle.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('mmb_theme', next);
  applyTheme(next);
});

mediaDark.addEventListener?.('change', () => {
  if(!localStorage.getItem('mmb_theme')) applyTheme(preferredTheme());
});

setMessageMode(false);
autoGrow();
updateSendState();
updateStats();
setInterval(updateStats, 3000);
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
        self.send_header("Cache-Control", "no-cache")
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
