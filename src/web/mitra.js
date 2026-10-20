/* Mitra: the MitraMind companion character, drawn as inline SVG (no images or libraries).
   It breathes and blinks on its own, can look like it is listening, thinking, speaking (mouth shapes in time
   with speech) or concerned, and can lead a breathing exercise. It is deliberately a cartoon, so nobody
   mistakes the AI companion for a person. Styles live in patient.css (.mitra ...). */
(function () {
  const MOUTH = {
    rest: {line: "M107 160 Q120 171 133 160"},
    smile: {line: "M104 158 Q120 175 136 158"},
    m: {line: "M109 163 Q120 165 131 163"},
    sad: {line: "M109 166 Q120 159 131 166"},
    a: {fill: "M106 158 Q120 155 134 158 Q132 178 120 179 Q108 178 106 158 Z"},
    e: {fill: "M103 159 Q120 157 137 159 Q131 170 120 171 Q109 170 103 159 Z"},
    o: {fill: "M120 156 C127 156 130 162 130 167 C130 173 126 177 120 177 C114 177 110 173 110 167 C110 162 113 156 120 156 Z"},
    oo: {fill: "M120 160 C124 160 126 163 126 166 C126 170 124 172 120 172 C116 172 114 170 114 166 C114 163 116 160 120 160 Z"},
  };
  const TALK = ["a", "e", "o", "a", "m", "e", "a", "oo"];

  const SVG = `
<svg class="mitra-svg" viewBox="0 0 240 262" role="img" aria-label="Mitra, your AI companion">
  <defs>
    <radialGradient id="mitraBody" cx="38%" cy="30%" r="75%"><stop offset="0" stop-color="#c7e6d3"/><stop offset=".6" stop-color="#9ccbb1"/><stop offset="1" stop-color="#78ad93"/></radialGradient>
    <radialGradient id="mitraBelly" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#f4faf6"/><stop offset="1" stop-color="#d9ede1"/></radialGradient>
  </defs>
  <ellipse class="m-shadow" cx="120" cy="250" rx="60" ry="8"/>
  <g class="m-float">
    <g class="m-breath">
      <g class="m-sprout">
        <path d="M120 58 C119 46 120 38 122 30" stroke="#5f9a7d" stroke-width="4" fill="none" stroke-linecap="round"/>
        <path d="M121 34 C108 18 92 20 86 26 C96 36 110 38 121 34 Z" fill="#b7a8d6"/>
        <path d="M123 32 C134 14 152 14 158 20 C150 32 136 36 123 32 Z" fill="#8cbca0"/>
      </g>
      <path class="m-arm m-arm-l" d="M60 170 Q38 182 44 206" stroke="#86b89e" stroke-width="15" fill="none" stroke-linecap="round"/>
      <path class="m-arm m-arm-r" d="M180 170 Q202 182 196 206" stroke="#86b89e" stroke-width="15" fill="none" stroke-linecap="round"/>
      <path class="m-body" d="M120 54 C174 54 202 102 202 160 C202 214 168 242 120 242 C72 242 38 214 38 160 C38 102 66 54 120 54 Z" fill="url(#mitraBody)"/>
      <ellipse class="m-belly" cx="120" cy="200" rx="46" ry="32" fill="url(#mitraBelly)"/>
      <g class="m-face">
        <path class="m-brow m-brow-l" d="M84 112 Q95 106 106 111" stroke="#3f5c4e" stroke-width="3.6" fill="none" stroke-linecap="round"/>
        <path class="m-brow m-brow-r" d="M134 111 Q145 106 156 112" stroke="#3f5c4e" stroke-width="3.6" fill="none" stroke-linecap="round"/>
        <g class="m-eyes">
          <g class="m-eye"><ellipse cx="95" cy="134" rx="11" ry="13.5" fill="#2f3a3d"/><circle class="m-glint" cx="99" cy="128.5" r="3.8" fill="#fff"/><circle cx="91" cy="139" r="1.7" fill="#fff" opacity=".7"/></g>
          <g class="m-eye"><ellipse cx="145" cy="134" rx="11" ry="13.5" fill="#2f3a3d"/><circle class="m-glint" cx="149" cy="128.5" r="3.8" fill="#fff"/><circle cx="141" cy="139" r="1.7" fill="#fff" opacity=".7"/></g>
        </g>
        <g class="m-lids">
          <path d="M84 135 Q95 143 106 135" stroke="#2f3a3d" stroke-width="3.6" fill="none" stroke-linecap="round"/>
          <path d="M134 135 Q145 143 156 135" stroke="#2f3a3d" stroke-width="3.6" fill="none" stroke-linecap="round"/>
        </g>
        <ellipse cx="76" cy="157" rx="11" ry="7" fill="#f2a7a0" opacity=".45"/>
        <ellipse cx="164" cy="157" rx="11" ry="7" fill="#f2a7a0" opacity=".45"/>
        <path class="m-mouth-fill" d="" fill="#6e3d3d"/>
        <path class="m-mouth-line" d="" stroke="#4a2c2c" stroke-width="3.6" fill="none" stroke-linecap="round"/>
      </g>
    </g>
  </g>
  <g class="m-puffs"><circle cx="120" cy="190" r="5"/><circle cx="112" cy="196" r="3.5"/><circle cx="128" cy="198" r="3"/></g>
  <g class="m-think"><circle cx="176" cy="66" r="4"/><circle cx="190" cy="52" r="6"/><circle cx="208" cy="34" r="8.5"/></g>
</svg>`;

  let instances = 0;

  class Mitra {
    constructor(host) {
      this.host = host;
      host.classList.add("mitra");
      // Gradient ids must be unique per page: a second Mitra pointing at a hidden first one's gradients renders blank.
      const n = ++instances;
      host.innerHTML = SVG.replace(/mitraBody|mitraBelly/g, (id) => `${id}${n}`);
      this.fill = host.querySelector(".m-mouth-fill");
      this.line = host.querySelector(".m-mouth-line");
      this.state = "idle";
      this.talking = null;
      this.mouth("rest");
      this._blinkLoop();
    }

    mouth(shape) {
      const m = MOUTH[shape] || MOUTH.rest;
      this.fill.setAttribute("d", m.fill || "");
      this.line.setAttribute("d", m.line || "");
    }

    /** idle | listening | thinking | speaking | concerned */
    setState(state) {
      this.host.classList.remove(`is-${this.state}`);
      this.state = state;
      this.host.classList.add(`is-${state}`);
      if (state !== "speaking") this.talk(false);
      if (state === "concerned") this.mouth("sad");
      else if (state === "listening") this.mouth("smile");
      else if (state !== "speaking") this.mouth("rest");
    }

    /** Start or stop the talking mouth: a quick random sequence of mouth shapes. */
    talk(on) {
      clearTimeout(this.talking);
      this.talking = null;
      if (!on) { if (this.state !== "concerned") this.mouth("rest"); return; }
      const step = () => {
        this.mouth(TALK[Math.floor(Math.random() * TALK.length)]);
        this.talking = setTimeout(step, 85 + Math.random() * 75);
      };
      step();
    }

    /** A word started (speech boundary event): open the mouth for it right away. */
    word() { if (this.talking) this.mouth(Math.random() < 0.6 ? "a" : "e"); }

    /** Lead a breathing exercise: "in" | "hold" | "out" | "calm" | "rest", over `seconds`; null stops guiding. */
    breathe(phase, seconds = 4, eyesClosed = true) {
      const h = this.host;
      h.classList.remove("b-in", "b-hold", "b-out", "b-calm", "b-rest", "eyes-closed");
      h.style.setProperty("--bdur", `${seconds}s`);
      if (!phase) { h.classList.remove("guiding"); this.mouth("rest"); return; }
      h.classList.add("guiding", `b-${phase}`);
      if (eyesClosed && phase !== "calm") h.classList.add("eyes-closed");
      this.mouth(phase === "out" ? "oo" : phase === "calm" ? "smile" : "m");
    }

    _blinkLoop() {
      const next = () => {
        this._blinkTimer = setTimeout(() => {
          this.host.classList.add("blink");
          setTimeout(() => this.host.classList.remove("blink"), 150);
          next();
        }, 2200 + Math.random() * 4200);
      };
      next();
    }

    destroy() { clearTimeout(this._blinkTimer); this.talk(false); this.host.innerHTML = ""; }
  }

  window.Mitra = Mitra;
})();
