from __future__ import annotations

from typing import Any

from friday.cited_report import build_cited_evidence_data
from friday.storage import FridayStore


def build_claim_support_audit(store: FridayStore, batch_id: str) -> dict[str, Any]:
    cited = build_cited_evidence_data(store, batch_id)
    supported_claims: list[dict[str, Any]] = []
    missing_page_anchor: list[dict[str, Any]] = []

    for evidence_type in ("claim", "method", "result", "dataset_population", "limitation"):
        for record in cited["evidence"].get(evidence_type, []):
            audit_record = {
                "claim_id": f"C{len(supported_claims) + len(missing_page_anchor) + 1}",
                "evidence_type": evidence_type,
                "citation": record["citation"],
                "paper": record["paper"],
                "page_number": record["page_number"],
                "text": record["text"],
                "quality_label": record.get("quality_label"),
                "quality_score": record.get("quality_score"),
                "quality_flags": record.get("quality_flags", []),
                "parse_confidence": record.get("parse_confidence"),
                "parse_flags": record.get("parse_flags", []),
            }
            if record["page_number"] > 0 and record["citation"]:
                supported_claims.append({**audit_record, "support_status": "SUPPORTED"})
            else:
                missing_page_anchor.append({**audit_record, "support_status": "MISSING_PAGE_ANCHOR"})

    material_gaps = []
    if not supported_claims and not missing_page_anchor:
        material_gaps.append(
            {
                "reason": "no_extracted_evidence",
                "message": "No page-anchored extracted evidence is available for claim support auditing.",
            }
        )

    return {
        "schema_version": "1.0",
        "artifact_type": "claim_support_audit",
        "batch_id": batch_id,
        "status": "pass" if supported_claims and not missing_page_anchor and not material_gaps else "gaps",
        "counts": {
            "supported": len(supported_claims),
            "missing_page_anchor": len(missing_page_anchor),
            "material_gaps": len(material_gaps),
            "total_checked": len(supported_claims) + len(missing_page_anchor),
        },
        "supported_claims": supported_claims,
        "missing_page_anchor": missing_page_anchor,
        "material_gaps": material_gaps,
    }
