// Compact Phase 1 User State Cards for the clinician and superuser dashboards.
// renderCards(container, patients, api, onOpen) draws one card per patient, then fills each in as its
// rule-based State Card arrives (no LLM, so a whole caseload loads in seconds).
(function () {
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const STRESSOR = {economic: "money", work: "work", school_exam: "exams", family: "family", relationships: "relationships",
                    health: "health", isolation: "loneliness", substance_use: "alcohol / substances", domestic: "home conflict",
                    covid19: "COVID-19"};

  function score(t) {
    if (!t) return `<span class="muted">-</span>`;
    const ch = t.n_administrations > 1 ? ` <span class="${t.change > 0 ? "up" : t.change < 0 ? "down" : ""}">${t.change > 0 ? "+" : ""}${t.change}</span>` : "";
    return `<b>${t.latest_total}</b>${ch} <span class="muted">${esc(t.latest_severity)}</span>`;
  }

  function body(card) {
    const q = card.questionnaires || {}, cues = card.cues || {};
    const stress = Object.entries(cues.stressors || {}).sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([k]) => STRESSOR[k] || k).join(", ");
    return `<div class="mc-scores"><div>PHQ-9 ${score(q["PHQ-9"])}</div><div>GAD-7 ${score(q["GAD-7"])}</div></div>
      <p class="mc-sum">${esc(card.narrative?.clinician_summary || "")}</p>
      <div class="mc-meta">${cues.n_entries || 0} check-ins${stress ? ` · stressors: ${esc(stress)}` : ""}</div>`;
  }

  window.renderCards = function (el, patients, api, onOpen, opts = {}) {
    if (!patients.length) { el.innerHTML = `<p class="muted">No patients to show yet.</p>`; return; }
    el.innerHTML = patients.map((p) => `<article class="mini-card" data-p="${esc(p.id)}">
      <header><b>${esc(p.id)}</b>${opts.showClinician ? `<span class="muted small">${esc(p.clinician ? p.clinician.name : "no clinician")}</span>` : ""}
        <span class="badge lvl-none" data-risk>…</span></header>
      <div class="mc-body">${p.consents && p.consents.care_team ? `<p class="muted small">Building card…</p>`
        : `<p class="muted small">This patient has not agreed to share with the care team, so only alerts are shown.</p>`}</div>
      ${p.consents && p.consents.care_team ? `<button class="mc-open">Open full summary</button>` : ""}
    </article>`).join("");
    el.querySelectorAll(".mc-open").forEach((b) => b.onclick = () => onOpen(b.closest(".mini-card").dataset.p));
    patients.forEach(async (p) => {
      const node = el.querySelector(`.mini-card[data-p="${CSS.escape(p.id)}"]`);
      const badge = node.querySelector("[data-risk]");
      if (!(p.consents && p.consents.care_team)) {
        badge.className = `badge lvl-${p.highest_open_alert}`; badge.textContent = p.open_alerts ? `${p.open_alerts} alert` : "none";
        return;
      }
      try {
        const res = await api(`/v1/clinician/patients/${encodeURIComponent(p.id)}/state-card?use_llm=false`);
        badge.className = `badge lvl-${res.card.risk.level}`; badge.textContent = res.card.risk.level;
        node.querySelector(".mc-body").innerHTML = body(res.card);
      } catch (e) {
        node.querySelector(".mc-body").innerHTML = `<p class="err small">${esc(e.message)}</p>`;
      }
    });
  };
})();
