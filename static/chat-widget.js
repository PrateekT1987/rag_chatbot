/*! Product Catalog Chatbot Widget
 *  Self-contained: no dependencies. Drop one <script> tag into any page.
 *
 *  Configuration (data attributes on the <script> tag):
 *    data-api-url      Backend base URL (default: same origin as this script)
 *    data-title        Header title        (default: "Product Assistant")
 *    data-subtitle     Header subtitle     (default: "Ask about products, specs & pricing")
 *    data-color        Accent color        (default: #4f46e5)
 *    data-position     "right" or "left"   (default: "right")
 *    data-suggestions  Semicolon-separated starter questions for the empty state
 */
(function () {
  "use strict";

  var script =
    document.currentScript ||
    document.querySelector('script[data-product-chatbot], script[src*="chat-widget"]');
  var src = script && script.src ? new URL(script.src) : null;

  var CFG = {
    apiUrl:
      (script && script.getAttribute("data-api-url")) ||
      (src ? src.origin : "") ||
      "",
    title: (script && script.getAttribute("data-title")) || "Product Assistant",
    subtitle:
      (script && script.getAttribute("data-subtitle")) ||
      "Ask about products, specs & pricing",
    color: (script && script.getAttribute("data-color")) || "#4f46e5",
    position: (script && script.getAttribute("data-position")) || "right",
    suggestions: (
      (script && script.getAttribute("data-suggestions")) ||
      "What headphones do you have?;" +
        "Which headphones have the longest battery life?;" +
        "Best monitor under $400?;" +
        "Compare the Aurora H700 and the Aurora Pro"
    ).split(";"),
  };

  var Z = 2147483000;
  var history = []; // chat turns sent for context
  var MAX_HISTORY = 10;

  /* ---------------------------------------------------------------- DOM -- */

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  var style = document.createElement("style");
  style.textContent = [
    ".pc-root{--pc-primary:" + CFG.color + ";position:fixed;bottom:20px;" +
      CFG.position + ":20px;z-index:" + Z + ";font-family:inherit;}",
    ".pc-launcher{width:60px;height:60px;border-radius:50%;border:0;cursor:pointer;" +
      "background:var(--pc-primary);color:#fff;box-shadow:0 8px 24px rgba(0,0,0,.28);" +
      "display:flex;align-items:center;justify-content:center;transition:transform .18s ease;}",
    ".pc-launcher:hover{transform:scale(1.07)}",
    ".pc-launcher svg{width:28px;height:28px;fill:currentColor}",
    ".pc-window{position:fixed;bottom:96px;" + CFG.position + ":20px;width:380px;" +
      "max-width:calc(100vw - 40px);height:600px;max-height:calc(100vh - 130px);" +
      "background:#fff;border-radius:18px;box-shadow:0 18px 48px rgba(0,0,0,.22);" +
      "display:none;flex-direction:column;overflow:hidden;}",
    ".pc-window.pc-open{display:flex;animation:pc-in .18s ease-out}",
    "@keyframes pc-in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}",
    ".pc-header{background:linear-gradient(135deg,var(--pc-primary),color-mix(in srgb,var(--pc-primary) 70%, #000));" +
      "color:#fff;padding:16px 18px;display:flex;align-items:center;gap:12px;flex:none}",
    ".pc-header .pc-badge{width:38px;height:38px;border-radius:50%;background:rgba(255,255,255,.18);" +
      "display:flex;align-items:center;justify-content:center;flex:none}",
    ".pc-header .pc-badge svg{width:20px;height:20px;fill:#fff}",
    ".pc-header h3{margin:0;font-size:16px;font-weight:600;line-height:1.25}",
    ".pc-header p{margin:2px 0 0;font-size:12px;opacity:.85}",
    ".pc-close{margin-left:auto;background:transparent;border:0;color:#fff;cursor:pointer;" +
      "font-size:22px;line-height:1;padding:4px 6px;border-radius:8px;opacity:.85}",
    ".pc-close:hover{opacity:1;background:rgba(255,255,255,.15)}",
    ".pc-messages{flex:1;overflow-y:auto;padding:16px 14px;background:#f7f8fa;" +
      "display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth}",
    ".pc-msg{max-width:86%;padding:10px 13px;border-radius:14px;font-size:14px;" +
      "line-height:1.45;word-wrap:break-word;white-space:normal}",
    ".pc-msg.pc-user{align-self:flex-end;background:var(--pc-primary);color:#fff;" +
      "border-bottom-right-radius:4px}",
    ".pc-msg.pc-bot{align-self:flex-start;background:#fff;color:#1f2430;" +
      "border:1px solid #e6e8ee;border-bottom-left-radius:4px;" +
      "box-shadow:0 1px 2px rgba(16,24,40,.05)}",
    ".pc-msg.pc-bot.pc-fallback{background:#fff8e6;border-color:#f2dfa8}",
    ".pc-msg.pc-error{align-self:flex-start;background:#fdecec;border:1px solid #f5c2c2;" +
      "color:#8a1f1f;font-size:13px}",
    ".pc-msg ul{margin:6px 0 0;padding-left:18px}",
    ".pc-msg li{margin:2px 0}",
    ".pc-msg strong{font-weight:650}",
    ".pc-cards{align-self:flex-start;display:flex;flex-direction:column;gap:6px;" +
      "max-width:86%;width:86%}",
    ".pc-card{background:#fff;border:1px solid #e6e8ee;border-left:4px solid var(--pc-primary);" +
      "border-radius:10px;padding:9px 12px;text-decoration:none;color:inherit;" +
      "box-shadow:0 1px 2px rgba(16,24,40,.05);transition:box-shadow .15s;}",
    ".pc-card:hover{box-shadow:0 4px 12px rgba(16,24,40,.12)}",
    ".pc-card .pc-card-name{font-size:13.5px;font-weight:650;display:flex;" +
      "justify-content:space-between;gap:8px;align-items:baseline}",
    ".pc-card .pc-card-price{color:var(--pc-primary);font-weight:700;white-space:nowrap}",
    ".pc-card .pc-card-meta{font-size:12px;color:#667085;margin-top:2px}",
    ".pc-typing{align-self:flex-start;background:#fff;border:1px solid #e6e8ee;" +
      "border-radius:14px;border-bottom-left-radius:4px;padding:12px 16px;display:flex;gap:5px}",
    ".pc-typing span{width:7px;height:7px;border-radius:50%;background:#9aa2b1;" +
      "animation:pc-bounce 1.2s infinite ease-in-out}",
    ".pc-typing span:nth-child(2){animation-delay:.15s}",
    ".pc-typing span:nth-child(3){animation-delay:.3s}",
    "@keyframes pc-bounce{0%,80%,100%{transform:translateY(0);opacity:.5}" +
      "40%{transform:translateY(-5px);opacity:1}}",
    ".pc-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}",
    ".pc-chip{border:1px solid var(--pc-primary);color:var(--pc-primary);background:#fff;" +
      "border-radius:999px;padding:5px 11px;font-size:12.5px;cursor:pointer;text-align:left}",
    ".pc-chip:hover{background:color-mix(in srgb,var(--pc-primary) 8%, #fff)}",
    ".pc-welcome{align-self:flex-start;max-width:92%}",
    ".pc-form{flex:none;display:flex;align-items:flex-end;gap:8px;padding:10px 12px;" +
      "background:#fff;border-top:1px solid #e6e8ee}",
    ".pc-input{flex:1;border:1px solid #d7dae2;border-radius:12px;padding:10px 12px;" +
      "font-size:14px;font-family:inherit;resize:none;max-height:96px;outline:none;" +
      "line-height:1.4;color:#1f2430}",
    ".pc-input:focus{border-color:var(--pc-primary);box-shadow:0 0 0 3px color-mix(in srgb,var(--pc-primary) 16%, transparent)}",
    ".pc-send{flex:none;width:42px;height:42px;border:0;border-radius:12px;cursor:pointer;" +
      "background:var(--pc-primary);color:#fff;display:flex;align-items:center;justify-content:center;" +
      "transition:opacity .15s}",
    ".pc-send:disabled{opacity:.45;cursor:not-allowed}",
    ".pc-send svg{width:19px;height:19px;fill:currentColor}",
    ".pc-note{font-size:11px;color:#98a2b3;text-align:center;padding:0 12px 8px;background:#fff;flex:none}",
    "@media (max-width:480px){",
    ".pc-window{bottom:0;" + CFG.position + ":0;width:100%;max-width:100%;height:100%;" +
      "max-height:100%;border-radius:0}",
    ".pc-window.pc-open{animation:none}",
    ".pc-header{padding-top:calc(16px + env(safe-area-inset-top))}",
    ".pc-form{padding-bottom:calc(10px + env(safe-area-inset-bottom))}",
    ".pc-launcher{bottom:16px;" + CFG.position + ":16px}",
    "}",
  ].join("\n");
  document.head.appendChild(style);

  var root = el("div", "pc-root");
  root.innerHTML =
    '<button class="pc-launcher" aria-label="Open chat" aria-expanded="false">' +
    '<svg viewBox="0 0 24 24"><path d="M12 3C6.99 3 3 6.36 3 10.5c0 2.3 1.24 4.35 3.2 5.76-.14 1.2-.6 2.3-1.35 3.24 1.65-.2 3.13-.8 4.33-1.66 .9.24 1.87.36 2.82.36 5.01 0 9-3.36 9-7.5S17.01 3 12 3z"/></svg></button>' +
    '<div class="pc-window" role="dialog" aria-label="' + CFG.title + '">' +
    '<div class="pc-header">' +
    '<div class="pc-badge"><svg viewBox="0 0 24 24"><path d="M12 3C6.99 3 3 6.36 3 10.5c0 2.3 1.24 4.35 3.2 5.76-.14 1.2-.6 2.3-1.35 3.24 1.65-.2 3.13-.8 4.33-1.66.9.24 1.87.36 2.82.36 5.01 0 9-3.36 9-7.5S17.01 3 12 3z"/></svg></div>' +
    "<div><h3></h3><p></p></div>" +
    '<button class="pc-close" aria-label="Close chat">&times;</button></div>' +
    '<div class="pc-messages" aria-live="polite"></div>' +
    '<form class="pc-form"><textarea class="pc-input" rows="1" placeholder="Ask about a product…" aria-label="Message"></textarea>' +
    '<button class="pc-send" type="submit" aria-label="Send">' +
    '<svg viewBox="0 0 24 24"><path d="M3.4 20.4l17.45-7.48a1 1 0 000-1.84L3.4 3.6a.89.89 0 00-1.18 1.08L4.5 11a1 1 0 00.73.73L18 14.5l-12.7 5.7a1 1 0 00-.9-1.8z"/></svg></button></form>' +
    '<div class="pc-note">Answers come from our product catalog</div></div>';
 document.body.appendChild(root);

  var launcher = root.querySelector(".pc-launcher");
  var win = root.querySelector(".pc-window");
  var closeBtn = root.querySelector(".pc-close");
  var messages = root.querySelector(".pc-messages");
  var form = root.querySelector(".pc-form");
  var input = root.querySelector(".pc-input");
  var sendBtn = root.querySelector(".pc-send");
  root.querySelector("h3").textContent = CFG.title;
  root.querySelector(".pc-header p").textContent = CFG.subtitle;

  var pending = false;
  var opened = false;
  var greeted = false;

  /* ------------------------------------------------------------ helpers -- */

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Minimal, safe markdown: **bold**, `code`, - / • bullets, line breaks.
  function format(text) {
    var esc = escapeHtml(text);
    var lines = esc.split(/\r?\n/);
    var out = [];
    var inList = false;
    lines.forEach(function (line) {
      var m = line.match(/^\s*[•\-*]\s+(.*)$/);
      if (m) {
        if (!inList) { out.push("<ul>"); inList = true; }
        out.push("<li>" + inline(m[1]) + "</li>");
      } else {
        if (inList) { out.push("</ul>"); inList = false; }
        if (line.trim()) out.push("<div>" + inline(line) + "</div>");
      }
    });
    if (inList) out.push("</ul>");
    return out.join("");
  }

  function inline(s) {
    return s
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function scrollDown() {
    messages.scrollTop = messages.scrollHeight;
  }

  function addBubble(cls, html) {
    var b = el("div", "pc-msg " + cls);
    if (html != null) b.innerHTML = html;
    messages.appendChild(b);
    scrollDown();
    return b;
  }

  function addBot(text, fallback) {
    return addBubble("pc-bot" + (fallback ? " pc-fallback" : ""), format(text));
  }

  function addCards(products) {
    if (!products || !products.length) return;
    var wrap = el("div", "pc-cards");
    products.slice(0, 3).forEach(function (p) {
      var card;
      if (p.url) {
        card = el("a", "pc-card");
        card.href = p.url;
        card.target = "_blank";
        card.rel = "noopener noreferrer";
      } else {
        card = el("div", "pc-card");
      }
      var name = el("div", "pc-card-name");
      var label = el("span", null, p.name);
      var price = el(
        "span",
        "pc-card-price",
        Number(p.price).toLocaleString(undefined, {
          style: "currency",
          currency: p.currency || "USD",
        })
      );
      name.appendChild(label);
      name.appendChild(price);
      var meta = el("div", "pc-card-meta", p.sku + (p.availability ? " · " + p.availability : ""));
      card.appendChild(name);
      card.appendChild(meta);
      wrap.appendChild(card);
    });
    messages.appendChild(wrap);
    scrollDown();
  }

  var typingNode = null;
  function showTyping() {
    if (typingNode) return;
    typingNode = el("div", "pc-typing");
    typingNode.innerHTML = "<span></span><span></span><span></span>";
    typingNode.setAttribute("aria-label", "Assistant is typing");
    messages.appendChild(typingNode);
    scrollDown();
  }
  function hideTyping() {
    if (typingNode) { typingNode.remove(); typingNode = null; }
  }

  function showWelcome() {
    var wrap = el("div", "pc-welcome");
    var bubble = el("div", "pc-msg pc-bot");
    bubble.innerHTML =
      "Hi! I can answer questions about our product catalog — prices, " +
      "specifications, availability and comparisons.";
    wrap.appendChild(bubble);
    if (CFG.suggestions.length) {
      var chips = el("div", "pc-chips");
      CFG.suggestions.forEach(function (q) {
        q = q.trim();
        if (!q) return;
        var chip = el("button", "pc-chip", q);
        chip.type = "button";
        chip.addEventListener("click", function () {
          input.value = q;
          submit();
        });
        chips.appendChild(chip);
      });
      wrap.appendChild(chips);
    }
    messages.appendChild(wrap);
  }

  /* ------------------------------------------------------------- events -- */

  function open() {
    opened = true;
    win.classList.add("pc-open");
    launcher.setAttribute("aria-expanded", "true");
    if (!greeted) { greeted = true; showWelcome(); }
    input.focus();
  }

  function close() {
    opened = false;
    win.classList.remove("pc-open");
    launcher.setAttribute("aria-expanded", "false");
    launcher.focus();
  }

  launcher.addEventListener("click", function () {
    opened ? close() : open();
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && opened) close();
  });

  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 96) + "px";
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    submit();
  });

  function pushHistory(role, content) {
    history.push({ role: role, content: content });
    if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY);
  }

  async function submit() {
    var text = input.value.trim();
    if (!text || pending) return;

    var hadGreeting = greeted;
    if (!hadGreeting) { greeted = true; showWelcome(); }
    var welcome = messages.querySelector(".pc-welcome");
    if (welcome) welcome.remove();

    addBubble("pc-user", format(text));
    pushHistory("user", text);
    input.value = "";
    input.style.height = "auto";
    setPending(true);
    showTyping();

    try {
      var res = await fetch(CFG.apiUrl + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: history.slice(0, -1) }),
      });
      hideTyping();
      if (res.status === 429) {
        addBubble("pc-error", "Too many messages — please wait a moment and try again.");
        return;
      }
      if (!res.ok) throw new Error("HTTP " + res.status);
      var data = await res.json();
      addBot(data.answer, data.fallback);
      addCards(data.products);
      pushHistory("assistant", data.answer);
    } catch (err) {
      hideTyping();
      addBubble(
        "pc-error",
        "Sorry, I couldn't reach the server. Please check your connection and try again."
      );
    } finally {
      setPending(false);
    }
  }

  function setPending(state) {
    pending = state;
    sendBtn.disabled = state;
    input.disabled = state;
    if (!state) input.focus();
  }
})();
