"""Draw the MitraMind architecture diagram: docs/architecture.pdf (submission) and docs/architecture.png (README).

    python docs/make_architecture.py        # needs matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
INK, MUTED = "#2f3a3d", "#5d6967"
SAGE, SAGE_SOFT, LAV, LAV_SOFT = "#457560", "#e6f0e9", "#5b4a86", "#f0ebf8"
CORAL, CORAL_SOFT, SAND, SAND_SOFT, CREAM, LINE = "#9c3b24", "#fdeee9", "#7a5a17", "#fbf3e2", "#faf7f2", "#d8d1c4"

fig, ax = plt.subplots(figsize=(17, 10.4))
fig.patch.set_facecolor(CREAM)
ax.set_xlim(0, 17)
ax.set_ylim(0, 10.4)
ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"


def box(x, y, w, h, title, lines=(), fg=SAGE, bg=SAGE_SOFT, size=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=bg, ec=fg, lw=1.4))
    ax.text(x + 0.16, y + h - 0.2, title, ha="left", va="top", fontsize=10.2, fontweight="bold", color=fg)
    for i, line in enumerate(lines):
        ax.text(x + 0.16, y + h - 0.52 - i * 0.27, line, ha="left", va="top", fontsize=size, color=INK)
    return (x, y, w, h)


def column(x, w, title):
    ax.add_patch(FancyBboxPatch((x, 0.95), w, 8.55, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="white", ec=LINE, lw=1, alpha=.85))
    ax.text(x + w / 2, 9.25, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)


def arrow(a, b, color=MUTED, text=None, rad=0.0, tx=0, ty=0.12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, ha="center", va="bottom", fontsize=8.3,
                color=color, fontstyle="italic")


# ------------------------------------------------------------------ title
ax.text(0.3, 10.05, "MitraMind: system architecture", fontsize=17, fontweight="bold", color=INK, va="center")
ax.text(0.3, 9.7, "Safety-first AI companion for patients between hospital visits. Everything runs locally: "
        "FastAPI + SQLite + Ollama (open-weight LLMs) + PyTorch models.", fontsize=10, color=MUTED, va="center")

# ------------------------------------------------------------------ columns
column(0.3, 3.3, "People and channels")
column(3.9, 2.9, "API layer")
column(7.1, 5.0, "Safety-first message pipeline")
column(12.4, 4.3, "Care team and data")

# people
patient = box(0.5, 6.45, 2.9, 2.45, "Patient app  ( / )", [
    "Home (welcome, team, privacy)", "Chat with Mitra: text, voice,", "  photos; voice mode with", "  lip-synced 2D Mitra",
    "Check-in: PHQ-9, GAD-7", "Mind Pause · My summary"], SAGE, SAGE_SOFT)
clin = box(0.5, 3.95, 2.9, 2.2, "Clinician dashboard", [
    "( /clinician )", "Alerts: acknowledge / resolve", "Patient cards, scores, usage", "Pre-visit summary + rating",
    "Care-team code to link patients", "Reads alerts, cards, summaries"], LAV, LAV_SOFT)
admin = box(0.5, 2.25, 2.9, 1.4, "Superuser view  ( /admin )", [
    "All clinicians and patients", "Link patients · clinician report"], SAND, SAND_SOFT)
wa = box(0.5, 1.15, 2.9, 0.85, "WhatsApp webhook", ["chat, voice notes, PHQ-9 / GAD-7"], MUTED, "#f3efe7")

# API layer
api = box(4.1, 5.0, 2.5, 3.9, "FastAPI + Uvicorn", [
    "REST + NDJSON streaming", "(sentence by sentence)", "", "/v1/patients/...", "/v1/clinician/...",
    "/v1/admin/...", "/v1/auth/...", "", "OpenAPI docs at /docs"], SAGE, "white")
auth = box(4.1, 2.75, 2.5, 2.0, "Auth and consent", [
    "Passwords: scrypt hashes", "Bearer session tokens", "Gmail SMTP sign-up and", "  reset codes",
    "Consent gate (DPDP Act)"], LAV, LAV_SOFT)
audit = box(4.1, 1.15, 2.5, 1.35, "Audit log", ["every AI reply: model,", "prompt, output, guardrails"], MUTED, "#f3efe7")

# pipeline
inp = box(7.3, 7.55, 4.6, 1.35, "1. Input", [
    "Text (English / Hinglish / Hindi)", "Voice → Whisper-small speech-to-text (local)", "Photo → shrunk in browser, EXIF removed"],
    SAGE, "white")
safety = box(7.3, 5.6, 4.6, 1.7, "2. Safety layer  (no LLM, runs first)", [
    "Crisis lexicons: English, Hinglish, Hindi phrases", "TF-IDF + logistic regression crisis classifier",
    "(test ROC-AUC 0.92)", "→ risk: none / watch / high / urgent"], CORAL, CORAL_SOFT)
crisis = box(7.3, 4.25, 2.15, 1.1, "Crisis reply", ["fixed text, never LLM", "Tele-MANAS 14416, 112"], CORAL, "white", 8.6)
cues = box(9.75, 3.0, 2.15, 2.35, "3. Cue models", [
    "m1 XLM-R  Hindi", "m2 XLM-R  Hinglish", "m3 WavLM  voice", "m4 WavLM-L arousal", "m5 lexicons (VADER,", "  Empath, PHQ/GAD)"],
    LAV, LAV_SOFT, 8.6)
llm = box(7.3, 1.15, 4.6, 1.6, "4. Companion LLM: Gemma 4 E4B (Ollama)", [
    "JSON-schema output, guardrails (no diagnosis, no", "  medication, no 'I am your therapist'), retry,",
    "  rule-based fallback · streamed reply + Mind Pause tip"], SAGE, SAGE_SOFT, 8.6)

# care side
store = box(12.6, 7.1, 3.9, 1.8, "SQLite store", [
    "patients, accounts, consents", "messages (+ photo / audio files)", "check-ins, alerts, ratings, audit"], SAGE, "white")
alerts = box(12.6, 5.35, 3.9, 1.5, "Alert queue", [
    "from messages (crisis language, photos)", "from PHQ-9 (item 9, worsening)", "watch / high / urgent"],
    CORAL, CORAL_SOFT)
quest = box(12.6, 4.15, 3.9, 0.95, "Questionnaires", ["PHQ-9 / GAD-7 scoring, severity, trends"], SAND, SAND_SOFT)
card = box(12.6, 2.2, 3.9, 1.7, "Phase 1 orchestrator", [
    "User State Card per patient (rule-based)", "Narrator LLM: Qwen 3.5 via Ollama", "→ pre-visit summary for the clinician"],
    LAV, LAV_SOFT)
export = box(12.6, 1.15, 3.9, 0.8, "Exports", ["FHIR R4 · de-identified pilot data"], MUTED, "#f3efe7")

# ------------------------------------------------------------------ arrows
arrow((3.4, 7.7), (4.1, 7.7), SAGE, "HTTPS / JSON", ty=0.08)
arrow((3.4, 5.1), (4.1, 6.0), LAV)
arrow((3.4, 2.95), (4.1, 3.6), SAND)
arrow((3.4, 1.6), (4.1, 2.95), MUTED)
arrow((6.6, 8.2), (7.3, 8.2), SAGE, "chat / voice", ty=0.08)
arrow((9.6, 7.55), (9.6, 7.3), CORAL)
arrow((8.35, 5.6), (8.35, 5.35), CORAL, "high / urgent", tx=-0.55, ty=-0.02)
arrow((10.8, 5.6), (10.8, 5.35), LAV, "none / watch", tx=0.55, ty=-0.02)
arrow((10.8, 3.0), (10.2, 2.75), SAGE, "cue digest", tx=0.45, ty=-0.05)
arrow((11.9, 6.45), (12.6, 6.1), CORAL, "alert", ty=0.04)
arrow((11.9, 8.2), (12.6, 8.2), SAGE, "store", ty=0.08)
arrow((7.3, 2.45), (6.6, 5.1), SAGE, "streamed reply", tx=0.32, ty=-0.62)
arrow((14.55, 7.1), (14.55, 6.85), MUTED)
arrow((14.55, 4.15), (14.55, 3.9), SAND)
arrow((16.1, 5.1), (16.1, 5.35), SAND)
arrow((7.3, 4.85), (6.6, 5.25), CORAL, "reply", tx=0.0, ty=0.05)

ax.text(8.5, 0.55, "Patients' chats stay private unless they share them; clinicians see scores and usage only with consent, "
        "and safety alerts always reach the care team.  MitraMind is a wellness support tool, not a diagnostic device.",
        ha="center", va="center", fontsize=9.2, color=MUTED, fontstyle="italic")

fig.savefig(OUT / "architecture.pdf", facecolor=CREAM, bbox_inches="tight", pad_inches=0.25)
fig.savefig(OUT / "architecture.png", facecolor=CREAM, bbox_inches="tight", pad_inches=0.25, dpi=150)
print("wrote", OUT / "architecture.pdf", "and", OUT / "architecture.png")
