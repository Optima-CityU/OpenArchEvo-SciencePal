/* OpenArchEvo version switcher — shared by every OpenArchEvo project page.
   Usage: <script src="assets/oae-versions.js" data-current="v1.5" defer></script>
   To add a version, append one entry to VERSIONS and copy this file to every site. */
(function () {
  var VERSIONS = [
    { id: 'v1', name: 'OpenArchEvo',
      en: 'LLM-written architecture programs, selected by evolution', zh: '大模型写架构程序，由演化择优',
      meta: 'arXiv 2609.40258', url: 'https://optima-cityu.github.io/OpenArchEvo/' },
    { id: 'v1.5', name: 'OpenArchEvo × SciencePal',
      en: 'Agents propose, evolution decides', zh: '众智提出，演化择优',
      meta: 'SciencePal AI4Science Hackathon 2026', url: 'https://optima-cityu.github.io/OpenArchEvo-SciencePal/' }
  ];

  var me = document.currentScript;
  var current = (me && me.getAttribute('data-current')) || '';

  function init() {
    var brand = document.querySelector('.topbar .brand, a.brand');
    if (!brand || brand.closest('.oaev')) return;

    var css = document.createElement('style');
    css.textContent = [
      '.oaev{position:relative;display:inline-flex;flex:none}',
      /* the brand becomes the front card of a small stack: the cards behind hint that there are other versions */
      '.oaev>.brand{position:relative;z-index:1;padding:7px 11px 7px 9px;margin:-7px -11px -7px -9px;border-radius:10px;cursor:pointer;',
      '  background:transparent;transition:background .2s,box-shadow .2s,transform .2s}',
      '.oaev::before,.oaev::after{content:"";position:absolute;inset:-7px -11px -7px -9px;border-radius:10px;pointer-events:none;',
      '  border:1px solid var(--line,#d9d9d9);background:var(--surface,var(--bg,#fff));transition:transform .25s,opacity .25s}',
      '.oaev::before{transform:translate(3px,3px);opacity:.55;z-index:0}',
      '.oaev::after{transform:translate(6px,6px);opacity:.28;z-index:-1}',
      '.oaev>.brand::after{content:"";width:6px;height:6px;margin-left:3px;border-right:1.6px solid currentColor;border-bottom:1.6px solid currentColor;',
      '  transform:translateY(-2px) rotate(45deg);opacity:.35;transition:opacity .2s,transform .2s}',
      '.oaev>.brand,.oaev>.brand:hover,.oaev>.brand:focus{color:var(--ink,#111);background:var(--surface,var(--bg,#fff));box-shadow:0 0 0 1px var(--line,#d9d9d9)}',
      '.oaev:hover::before{transform:translate(4px,5px);opacity:.75}',
      '.oaev:hover::after{transform:translate(8px,10px);opacity:.45}',
      '.oaev:hover>.brand::after,.oaev.open>.brand::after{opacity:.75}',
      '.oaev.open>.brand::after{transform:translateY(1px) rotate(-135deg)}',
      '.oaev>.brand:focus-visible{outline:2px solid var(--ink,#111);outline-offset:3px}',
      '.oaev-menu{position:absolute;top:calc(100% + 16px);left:-9px;z-index:100;width:340px;max-width:calc(100vw - 32px);padding:6px;',
      '  background:var(--surface,var(--bg,#fff));border:1px solid var(--line,#d9d9d9);border-radius:14px;',
      '  box-shadow:0 18px 50px -12px rgba(0,0,0,.28),0 2px 6px rgba(0,0,0,.06);',
      '  opacity:0;transform:translateY(-6px);pointer-events:none;transition:opacity .18s,transform .18s}',
      '.oaev.open .oaev-menu{opacity:1;transform:none;pointer-events:auto}',
      '.oaev-menu .h{font:600 .68rem/1 var(--f-mono,ui-monospace,monospace);letter-spacing:.08em;text-transform:uppercase;color:var(--muted,#777);padding:10px 12px 8px}',
      '.oaev-menu a{display:grid;grid-template-columns:44px 1fr;gap:2px 10px;padding:10px 12px;border-radius:10px;color:var(--ink,#111);text-decoration:none}',
      '.oaev-menu a:hover,.oaev-menu a:focus-visible{background:color-mix(in srgb,var(--ink,#111) 6%,transparent);outline:none}',
      '.oaev-menu .v{grid-row:span 3;align-self:start;font:700 .78rem/1 var(--f-mono,ui-monospace,monospace);padding:5px 0;text-align:center;',
      '  border-radius:7px;border:1px solid var(--line,#d9d9d9);color:var(--muted,#777)}',
      '.oaev-menu a.cur .v{background:var(--ink,#111);border-color:var(--ink,#111);color:var(--bg,#fff)}',
      '.oaev-menu .n{font:700 .92rem/1.25 var(--f-display,inherit)}',
      '.oaev-menu .d{font-size:.82rem;line-height:1.35;color:var(--muted,#666)}',
      '.oaev-menu .m{font:500 .7rem/1.3 var(--f-mono,ui-monospace,monospace);color:var(--faint,var(--muted,#999))}',
      '.oaev-menu .here{font:600 .66rem/1 var(--f-mono,ui-monospace,monospace);margin-left:6px;padding:2px 5px;border-radius:5px;vertical-align:2px;',
      '  background:color-mix(in srgb,var(--ink,#111) 8%,transparent);color:var(--muted,#666)}',
      '@media (prefers-reduced-motion:reduce){.oaev *,.oaev::before,.oaev::after,.oaev-menu{transition:none!important}}'
    ].join('\n');
    document.head.appendChild(css);

    var wrap = document.createElement('span');
    wrap.className = 'oaev';
    brand.parentNode.insertBefore(wrap, brand);
    wrap.appendChild(brand);
    brand.setAttribute('role', 'button');
    brand.setAttribute('aria-haspopup', 'true');
    brand.setAttribute('aria-expanded', 'false');
    brand.setAttribute('title', 'OpenArchEvo versions');

    var menu = document.createElement('div');
    menu.className = 'oaev-menu';
    menu.setAttribute('role', 'menu');
    wrap.appendChild(menu);

    function zh() { return (document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0; }
    function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
    function render() {
      var z = zh();
      menu.innerHTML = '<div class="h">' + (z ? 'OpenArchEvo 系列' : 'OpenArchEvo versions') + '</div>' +
        VERSIONS.map(function (v) {
          var cur = v.id === current;
          return '<a role="menuitem" href="' + esc(v.url) + '"' + (cur ? ' class="cur" aria-current="page"' : '') + '>' +
            '<span class="v">' + esc(v.id) + '</span>' +
            '<span class="n">' + esc(v.name) + (cur ? '<span class="here">' + (z ? '当前' : 'here') + '</span>' : '') + '</span>' +
            '<span class="d">' + esc(z ? v.zh : v.en) + '</span>' +
            '<span class="m">' + esc(v.meta) + '</span></a>';
        }).join('');
    }
    function setOpen(o) {
      if (o) render();
      wrap.classList.toggle('open', o);
      brand.setAttribute('aria-expanded', o ? 'true' : 'false');
    }
    brand.addEventListener('click', function (e) { e.preventDefault(); setOpen(!wrap.classList.contains('open')); });
    brand.addEventListener('keydown', function (e) {
      if (e.key === ' ' ) { e.preventDefault(); setOpen(!wrap.classList.contains('open')); }
      if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); var a = menu.querySelector('a'); a && a.focus(); }
    });
    document.addEventListener('click', function (e) { if (!wrap.contains(e.target)) setOpen(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && wrap.classList.contains('open')) { setOpen(false); brand.focus(); } });
    render();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
