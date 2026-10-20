"""FHIR R4 export, so scores, alerts and the pre-visit summary can flow into the hospital's EHR.

A `collection` Bundle with:
  Patient            pseudonymous identifier only (the hospital maps it to its MRN)
  Observation        one per PHQ-9 / GAD-7 total (LOINC 44261-6 / 70274-6), with item answers as components
  Flag               one per open alert (risk level, reasons)
  DocumentReference  the latest pre-visit summary as Markdown (optional)
ABDM (Ayushman Bharat Digital Mission) health-information exchange needs the NRCeS FHIR IG profiles and a
registered HIP; that is still to do (task.md).
"""
from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

LOINC = {"PHQ-9": ("44261-6", "Patient Health Questionnaire 9 item (PHQ-9) total score [Reported]"),
         "GAD-7": ("70274-6", "Generalized anxiety disorder 7 item (GAD-7) total score [Reported.PHQ]")}
ID_SYSTEM = "urn:mitramind:patient"


def _entry(resource: dict) -> dict:
    return {"fullUrl": f"urn:uuid:{uuid.uuid4()}", "resource": resource}


def patient_bundle(store, patient_id: str, card_markdown: str | None = None) -> dict:
    subject = {"identifier": {"system": ID_SYSTEM, "value": patient_id}}
    entries = [_entry({"resourceType": "Patient", "identifier": [{"system": ID_SYSTEM, "value": patient_id}]})]
    for q in store.questionnaires(patient_id):
        code, display = LOINC[q["instrument"]]
        entries.append(_entry({
            "resourceType": "Observation", "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                      "code": "survey", "display": "Survey"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}],
                     "text": f"{q['instrument']} total score"},
            "subject": subject, "effectiveDateTime": q["date"], "valueInteger": q["total"],
            "component": [{"code": {"text": f"{q['instrument']} item {i + 1}"}, "valueInteger": v}
                          for i, v in enumerate(q["items"])],
            "note": [{"text": f"Collected via MitraMind ({q['channel']})"}]}))
    for a in store.alerts("open", patient_id):
        entries.append(_entry({
            "resourceType": "Flag", "status": "active",
            "category": [{"text": "MitraMind risk triage"}],
            "code": {"text": f"MitraMind risk level {a['level']}: " + "; ".join(a["reasons"])},
            "subject": subject, "period": {"start": a["created_at"]}}))
    if card_markdown:
        entries.append(_entry({
            "resourceType": "DocumentReference", "status": "current",
            "type": {"text": "MitraMind pre-visit summary (not a diagnosis)"}, "subject": subject,
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "content": [{"attachment": {"contentType": "text/markdown",
                                        "data": base64.b64encode(card_markdown.encode()).decode()}}]}))
    return {"resourceType": "Bundle", "type": "collection",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "entry": entries}
