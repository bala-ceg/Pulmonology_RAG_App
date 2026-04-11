"""
Blueprint: PDF generation routes.

Covers /generate_patient_pdf, /generate_chat_pdf, and
/generate_conversation_pdf (original main.py lines 2137-3033).

Tries to import pdf_service first; if unavailable the full inline
reportlab logic from main.py is used as a fallback.

Azure uploads are performed via the upload_pdf_to_azure helper
(requires azure_storage to be installed and configured).
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime

from flask import Blueprint, jsonify, make_response, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

pdf_bp = Blueprint("pdf_bp", __name__)

# ---------------------------------------------------------------------------
# Optional service layer
# ---------------------------------------------------------------------------
try:
    from services.pdf_service import pdf_service  # type: ignore[import]

    PDF_SERVICE_AVAILABLE = True
except ImportError:
    PDF_SERVICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional Azure storage
# ---------------------------------------------------------------------------
try:
    from azure_storage import get_storage_manager

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Azure upload helper (mirrors upload_pdf_to_azure in main.py)
# ---------------------------------------------------------------------------

def _upload_pdf_to_azure(
    pdf_content: bytes,
    filename: str,
    pdf_type: str,
    metadata: dict | None = None,
) -> str | None:
    """Upload PDF bytes to Azure Blob Storage. Returns URL or None."""
    if not AZURE_AVAILABLE:
        return None
    try:
        storage_manager = get_storage_manager()
        metadata = metadata or {}

        if pdf_type == "research":
            return storage_manager.upload_research_pdf(
                pdf_content, filename, metadata.get("patient_problem"), metadata
            )
        if pdf_type == "patient_summary":
            patient_data = {
                "patient_name": metadata.get("patient_name"),
                "patient_id": metadata.get("patient_id"),
                "doctor_name": metadata.get("doctor_name"),
                "session_date": metadata.get("date_time"),
            }
            return storage_manager.upload_patient_summary_pdf(
                pdf_content, filename, patient_data, metadata
            )
        if pdf_type == "conversation":
            conversation_data = {
                "doctor_name": metadata.get("doctor_name"),
                "patient_name": metadata.get("patient_name"),
                "duration": metadata.get("duration", "Unknown"),
                "session_date": metadata.get("date_time"),
            }
            return storage_manager.upload_conversation_pdf(
                pdf_content, filename, conversation_data, metadata
            )
    except Exception as exc:
        logger.warning("Azure upload failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@pdf_bp.route("/generate_patient_pdf", methods=["POST"])
@handle_route_errors
def generate_patient_pdf():
    """Generate PDF for patient notes with patient problem capture and Azure upload."""
    if PDF_SERVICE_AVAILABLE:
        data = request.get_json(silent=True) or {}
        pdf_bytes, azure_url = pdf_service.generate_patient_pdf(data)
        patient_name = data.get("patientName", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = (
            f"{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
            if patient_name
            else f"patient_notes-{timestamp}.pdf"
        )
        resp = make_response(pdf_bytes)
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        if azure_url:
            resp.headers["X-Azure-URL"] = azure_url
        return resp

    try:
        from reportlab.lib.colors import black  # noqa: F401
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4  # noqa: F401
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return (
            jsonify({"error": "PDF generation not available. Please install reportlab."}),
            500,
        )

    data = request.json

    doctor_name = data.get("doctorName", "")
    patient_name = data.get("patientName", "")
    patient_id = data.get("patientId", "")
    date_time = data.get("dateTime", "")
    transcription = data.get("transcription", "")
    summary = data.get("summary", "")
    conclusion = data.get("conclusion", "")
    patient_problem = data.get("patientProblem", "")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=10,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName="Helvetica",
        textColor="blue",
    )
    header_style = ParagraphStyle(
        "CustomHeader",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=15,
        fontName="Helvetica-Bold",
    )
    session_style = ParagraphStyle(
        "SessionStyle",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica",
        spaceAfter=5,
    )

    story: list = []

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ClientLogo101.png")
    patient_display_name = patient_name if patient_name else "Patient Name"
    main_title = f"Patient – {patient_display_name} – Recording Notes"

    if os.path.exists(logo_path):
        try:
            logo_img = Image(logo_path, width=1.5 * inch, height=0.9 * inch)
            title_paragraph = Paragraph(main_title, title_style)
            header_data = [[title_paragraph, logo_img]]
            header_table = Table(header_data, colWidths=[4.5 * inch, 2 * inch])
            header_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )
            story.append(header_table)
            story.append(Spacer(1, 15))
        except Exception as exc:
            logger.warning("Could not add logo to PDF: %s", exc)
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
    else:
        story.append(Paragraph(main_title, title_style))
        story.append(Spacer(1, 15))

    story.append(Paragraph("Patient Recording Notes", subtitle_style))
    story.append(Spacer(1, 10))

    patient_id_display = patient_id if patient_id else "N/A"
    story.append(
        Paragraph(
            f"Patient Name: {patient_display_name} – Patient ID: {patient_id_display}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            f"Doctor's Name: {doctor_name if doctor_name else 'Unknown Doctor'}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))

    story.append(Paragraph("Session Information", header_style))
    story.append(
        Paragraph(
            "Transcription Engine: Whisper; Summary Engine: OpenAI", session_style
        )
    )
    display_date = (
        date_time if date_time else datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
    )
    story.append(Paragraph(f"Date: {display_date}", session_style))
    story.append(Spacer(1, 15))

    if patient_problem:
        story.append(Paragraph("Patient Problem", header_style))
        story.append(Paragraph(patient_problem, styles["Normal"]))
        story.append(Spacer(1, 15))

    story.append(Paragraph("Original Transcription Text", header_style))
    story.append(
        Paragraph(
            transcription if transcription else "No transcription available",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))

    story.append(Paragraph("Medical Summary", header_style))
    story.append(
        Paragraph(
            summary if summary else "No summary available", styles["Normal"]
        )
    )
    story.append(Spacer(1, 15))

    story.append(Paragraph("Conclusion & Recommendations", header_style))
    story.append(
        Paragraph(
            conclusion if conclusion else "No conclusion available", styles["Normal"]
        )
    )

    doc.build(story)
    buffer.seek(0)
    pdf_content = buffer.getvalue()

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"

    azure_url = _upload_pdf_to_azure(
        pdf_content,
        filename,
        "patient_summary",
        {
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "date_time": date_time,
            "patient_problem": patient_problem,
            "pdf_type": "patient_notes",
            "generated_at": datetime.now().isoformat(),
        },
    )

    response = make_response(pdf_content)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if azure_url:
        response.headers["X-Azure-URL"] = azure_url
    return response


@pdf_bp.route("/generate_chat_pdf", methods=["POST"])
@handle_route_errors
def generate_chat_pdf():
    """Generate PDF for chat conversation with specified format."""
    if PDF_SERVICE_AVAILABLE:
        data = request.get_json(silent=True) or {}
        pdf_bytes, azure_url = pdf_service.generate_chat_pdf(data)
        doctor_name = data.get("doctorName", "")
        patient_name = data.get("patientName", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        if patient_name:
            filename = (
                f"{doctor_name.upper().replace(' ', '')}"
                f"-{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
            )
        else:
            filename = f"{doctor_name.upper().replace(' ', '') or 'chat'}-{timestamp}.pdf"
        resp = make_response(pdf_bytes)
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        if azure_url:
            resp.headers["X-Azure-URL"] = azure_url
        return resp

    try:
        from bs4 import BeautifulSoup
        from reportlab.lib.colors import black  # noqa: F401
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4  # noqa: F401
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            PageBreak,  # noqa: F401
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return (
            jsonify({"error": "PDF generation not available. Please install reportlab."}),
            500,
        )

    data = request.json

    doctor_name = data.get("doctorName", "Dr. Name")
    patient_name = data.get("patientName", "")
    patient_id = data.get("patientId", "")
    patient_problem = data.get("patientProblem", "")
    messages = data.get("messages", [])
    json_data = data.get("jsonData", "")

    now = datetime.now()
    formatted_date = now.strftime("%m/%d/%Y, %I:%M:%S %p")
    timestamp = now.strftime("%Y%m%d%H%M")

    if patient_name:
        filename = (
            f"{doctor_name.upper().replace(' ', '')}"
            f"-{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
        )
    else:
        filename = f"{doctor_name.upper().replace(' ', '')}-{timestamp}.pdf"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=10,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName="Helvetica",
        textColor="blue",
    )
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=15,
        fontName="Helvetica-Bold",
    )
    session_style = ParagraphStyle(
        "SessionStyle",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica",
        spaceAfter=5,
    )
    content_style = ParagraphStyle(
        "ContentStyle",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica",
        spaceAfter=8,
        alignment=TA_LEFT,
        spaceBefore=2,
        leading=14,
    )
    citation_style = ParagraphStyle(
        "CitationStyle",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica",
        spaceAfter=4,
        alignment=TA_LEFT,
        leftIndent=10,
        leading=12,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        spaceAfter=6,
        spaceBefore=12,
        alignment=TA_CENTER,
    )

    story: list = []

    patient_display_name = patient_name if patient_name else "Patient Name"
    main_title = f"Patient – {patient_display_name} – Research"

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ClientLogo101.png")
    if os.path.exists(logo_path):
        try:
            logo_img = Image(logo_path, width=1.5 * inch, height=0.9 * inch)
            title_paragraph = Paragraph(main_title, title_style)
            header_data = [[title_paragraph, logo_img]]
            header_table = Table(header_data, colWidths=[4.5 * inch, 2 * inch])
            header_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )
            story.append(header_table)
            story.append(Spacer(1, 15))
        except Exception as exc:
            logger.warning("Could not add logo to PDF: %s", exc)
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
    else:
        story.append(Paragraph(main_title, title_style))
        story.append(Spacer(1, 15))

    story.append(Paragraph("Patient Recording Notes", subtitle_style))
    story.append(Spacer(1, 10))

    if patient_id:
        patient_info = f"Patient Name: {patient_display_name} – Patient ID: {patient_id}"
    else:
        patient_info = f"Patient Name: {patient_display_name}"
    story.append(Paragraph(patient_info, styles["Normal"]))
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            f"Doctor's Name: {doctor_name if doctor_name else 'Unknown Doctor'}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))

    story.append(Paragraph("Session Information", header_style))
    story.append(Paragraph(f"Date: {formatted_date}", session_style))
    story.append(Spacer(1, 12))

    if patient_problem:
        story.append(Paragraph("Patient Problem", header_style))
        story.append(Paragraph(patient_problem, styles["Normal"]))
        story.append(Spacer(1, 12))

    if json_data:
        story.append(Paragraph("JSON Data:", section_header_style))
        story.append(Paragraph(json_data, content_style))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Conversation", header_style))
    story.append(Spacer(1, 6))

    def _convert_markdown_to_reportlab(text: str) -> str:
        """Convert enhanced-tools HTML format to ReportLab-compatible text."""
        if "<div" in text and "<h4" in text:
            try:
                soup = BeautifulSoup(text, "html.parser")
                sections: dict[str, str] = {
                    "answer": "",
                    "source": "",
                    "tool_routing": "",
                }
                divs = soup.find_all(
                    "div", style=lambda x: x and "margin-bottom" in x
                )
                for div in divs:
                    h4 = div.find("h4")
                    if h4:
                        header_text = h4.get_text().strip().lower()
                        content_div = div.find(
                            "div",
                            style=lambda x: x
                            and ("background-color" in x or "padding" in x),
                        )
                        if "answer" in header_text:
                            if content_div:
                                sections["answer"] = content_div.get_text().strip()
                        elif "source" in header_text:
                            if content_div:
                                sources = []
                                links = content_div.find_all("a")
                                if links:
                                    for link in links:
                                        link_text = link.get_text().strip()
                                        next_text = link.next_sibling
                                        if next_text and isinstance(next_text, str):
                                            sources.append(
                                                f"{link_text} {next_text.strip()}"
                                            )
                                        else:
                                            sources.append(link_text)
                                else:
                                    sources = [content_div.get_text().strip()]
                                sections["source"] = "\n".join(sources)
                        elif (
                            "tool selection" in header_text
                            or "routing" in header_text
                        ):
                            if content_div:
                                sections["tool_routing"] = str(content_div)

                formatted_parts: list[str] = []
                if sections["answer"]:
                    formatted_parts.append(sections["answer"])
                if sections["source"]:
                    srcs = [
                        s.strip()
                        for s in sections["source"].split("\n")
                        if s.strip()
                    ]
                    if srcs:
                        formatted_parts.append(
                            "<br/><br/><b>Sources:</b><br/>• "
                            + "<br/>• ".join(srcs)
                        )
                if sections["tool_routing"]:
                    routing_soup = BeautifulSoup(sections["tool_routing"], "html.parser")
                    raw = routing_soup.get_text()
                    confidence = tools_used = reasoning = ""
                    lines = raw.replace("\n", " ").split("Tools Used:")
                    if len(lines) >= 2:
                        cm = re.search(
                            r"Confidence:\s*(.+?)$",
                            lines[0].strip(),
                            re.IGNORECASE,
                        )
                        if cm:
                            confidence = cm.group(1).strip()
                        rest = lines[1]
                        rs = rest.split("Reasoning:")
                        if len(rs) >= 2:
                            tools_used = rs[0].strip()
                            reasoning = rs[1].strip()
                        else:
                            tools_used = rest.strip()
                    routing_parts: list[str] = []
                    if confidence:
                        routing_parts.append(f"<b>Confidence:</b> {confidence}")
                    if tools_used:
                        routing_parts.append(f"<b>Tools Used:</b> {tools_used}")
                    if reasoning:
                        routing_parts.append(f"<b>Reasoning:</b> {reasoning}")
                    if routing_parts:
                        formatted_parts.append(
                            "<br/><br/><b>Tool Selection & Query Routing:</b><br/>"
                            + "<br/>".join(routing_parts)
                        )

                full_text = "".join(formatted_parts)
                full_text = re.sub(r"\s+", " ", full_text)
                return full_text.strip()

            except Exception as exc:
                logger.warning("Error parsing HTML content: %s", exc)
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text().strip()

        # Plain text path
        sections_plain: dict[str, str] = {
            "answer": "",
            "source": "",
            "tool_routing": "",
        }
        current_section: str | None = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("answer"):
                current_section = "answer"
                continue
            elif line.lower().startswith("source"):
                current_section = "source"
                continue
            elif line.lower().startswith("tool selection") or line.lower().startswith(
                "confidence:"
            ):
                current_section = "tool_routing"
                if line.lower().startswith("tool selection"):
                    continue
            if current_section:
                if sections_plain[current_section]:
                    sections_plain[current_section] += " " + line
                else:
                    sections_plain[current_section] = line

        formatted_plain: list[str] = []
        if sections_plain["answer"]:
            formatted_plain.append(sections_plain["answer"])
        if sections_plain["source"]:
            srcs = [
                s.strip()
                for s in sections_plain["source"].split("\n")
                if s.strip()
            ]
            if srcs:
                formatted_plain.append(
                    "<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(srcs)
                )
        if sections_plain["tool_routing"]:
            formatted_plain.append(
                f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>"
                f"{sections_plain['tool_routing']}"
            )
        if not any(sections_plain.values()):
            formatted_plain = [text]

        full_plain = "".join(formatted_plain)
        full_plain = re.sub(r"\s+", " ", full_plain)
        return full_plain.strip()

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        formatted_content = _convert_markdown_to_reportlab(content)

        if role == "user":
            story.append(Paragraph("Doctor Input:", section_header_style))
        elif role == "ai":
            story.append(Paragraph("System Output:", section_header_style))
        else:
            story.append(Paragraph(f"{role.title()}:", section_header_style))

        story.append(Paragraph(formatted_content, content_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    pdf_content = buffer.getvalue()

    azure_url = _upload_pdf_to_azure(
        pdf_content,
        filename,
        "research",
        {
            "doctor_name": str(doctor_name),
            "patient_name": str(patient_name) if patient_name else "",
            "patient_id": str(patient_id) if patient_id else "",
            "patient_problem": str(patient_problem) if patient_problem else "",
            "pdf_type": "research_chat",
            "generated_at": datetime.now().isoformat(),
            "message_count": str(len(messages)),
            "has_json_data": str(bool(json_data)),
        },
    )

    response = make_response(pdf_content)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if azure_url:
        response.headers["X-Azure-URL"] = azure_url
    return response


@pdf_bp.route("/generate_conversation_pdf", methods=["POST"])
@handle_route_errors
def generate_conversation_pdf():
    """Generate PDF for conversation segments with voice diarization results."""
    if PDF_SERVICE_AVAILABLE:
        data = request.get_json(silent=True) or {}
        pdf_bytes, azure_url = pdf_service.generate_conversation_pdf(data)
        doctor_name = data.get("doctorName", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"CONVERSATION-{doctor_name.upper().replace(' ', '') or 'doc'}-{timestamp}.pdf"
        resp = make_response(pdf_bytes)
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        if azure_url:
            resp.headers["X-Azure-URL"] = azure_url
        return resp

    try:
        from reportlab.lib.colors import blue, green
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4  # noqa: F401
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return (
            jsonify({"error": "PDF generation not available. Please install reportlab."}),
            500,
        )

    data = request.json

    doctor_name = data.get("doctorName", "Doctor")
    patient_name = data.get("patientName", "Patient")
    date_time = data.get("dateTime", "")
    segments = data.get("segments", [])
    full_transcript = data.get("fullTranscript", "")
    summary = data.get("summary", "")
    conclusion = data.get("conclusion", "")
    processing_info = data.get("processingInfo", {})
    is_duplicate = data.get("isDuplicate", False)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=10,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName="Helvetica",
        textColor=blue,
    )
    header_style = ParagraphStyle(
        "CustomHeader",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=15,
        fontName="Helvetica-Bold",
    )
    segment_style = ParagraphStyle(
        "SegmentStyle",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=10,
        fontName="Helvetica",
    )
    doctor_segment_style = ParagraphStyle(
        "DoctorSegmentStyle",
        parent=segment_style,
        leftIndent=20,
        textColor=green,
    )
    patient_segment_style = ParagraphStyle(
        "PatientSegmentStyle",
        parent=segment_style,
        leftIndent=20,
        textColor=blue,
    )

    story: list = []

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ClientLogo101.png")
    duplicate_text = " (Duplicate)" if is_duplicate else ""
    main_title = f"Doctor-Patient Conversation Analysis{duplicate_text}"

    if os.path.exists(logo_path):
        try:
            logo_img = Image(logo_path, width=1.5 * inch, height=0.9 * inch)
            title_paragraph = Paragraph(main_title, title_style)
            header_data = [[title_paragraph, logo_img]]
            header_table = Table(header_data, colWidths=[4.5 * inch, 2 * inch])
            header_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )
            story.append(header_table)
            story.append(Spacer(1, 15))
        except Exception as exc:
            logger.warning("Could not add logo to PDF: %s", exc)
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
    else:
        story.append(Paragraph(main_title, title_style))
        story.append(Spacer(1, 15))

    story.append(
        Paragraph("Voice Diarization & Transcription Results", subtitle_style)
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Doctor: {doctor_name}", styles["Normal"]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Processing Information", header_style))

    language_display = {
        "en": "English", "es": "Spanish", "zh": "Chinese", "yue": "Cantonese",
        "tl": "Tagalog", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
        "gu": "Gujarati", "pa": "Punjabi",
    }
    language_used = processing_info.get("language", "English")
    language_name = language_display.get(
        language_used, language_used.title() if language_used else "English"
    )
    total_segments = processing_info.get("totalSegments", len(segments))

    story.append(
        Paragraph(
            "Voice Diarization: OpenAI-based Speaker Separation  "
            "Transcription Engine: Whisper",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            f"Language Majorly Spoken: {language_name}   Total Segments: {total_segments}",
            styles["Normal"],
        )
    )
    display_date = (
        date_time if date_time else datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
    )
    story.append(Paragraph(f"Processing Date: {display_date}", styles["Normal"]))
    story.append(Spacer(1, 15))

    if segments:
        story.append(Paragraph("Conversation Segments", header_style))
        for segment in segments:
            role = segment.get("role", "Unknown")
            text = segment.get("text", "")
            start_time = segment.get("start", "")
            end_time = segment.get("end", "")
            confidence = segment.get("confidence", 0)

            timing_info = f"[{start_time} - {end_time}]" if start_time and end_time else ""
            confidence_info = (
                f"(Confidence: {int(confidence * 100)}%)" if confidence > 0 else ""
            )
            header_text = f"{role} {timing_info} {confidence_info}"

            seg_style = (
                doctor_segment_style
                if role.lower() == "doctor"
                else patient_segment_style
            )
            story.append(Paragraph(f"<b>{header_text}</b>", seg_style))
            story.append(Paragraph(text, seg_style))
            story.append(Spacer(1, 8))

        story.append(Spacer(1, 15))

    story.append(Paragraph("Complete Transcript", header_style))
    story.append(
        Paragraph(
            full_transcript if full_transcript else "No transcript available",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))

    if summary:
        story.append(Paragraph("Medical Summary", header_style))
        story.append(Paragraph(summary, styles["Normal"]))
        story.append(Spacer(1, 15))

    if conclusion:
        story.append(Paragraph("Conclusion & Recommendations", header_style))
        story.append(Paragraph(conclusion, styles["Normal"]))
        story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)
    pdf_content = buffer.getvalue()

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    duplicate_suffix = "-DUPLICATE" if is_duplicate else ""
    filename = (
        f"CONVERSATION-{doctor_name.upper().replace(' ', '')}"
        f"{duplicate_suffix}-{timestamp}.pdf"
    )

    azure_url = _upload_pdf_to_azure(
        pdf_content,
        filename,
        "conversation",
        {
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "date_time": date_time,
            "summary": summary if summary else "",
            "conclusion": conclusion if conclusion else "",
            "pdf_type": "conversation_segments",
            "is_duplicate": is_duplicate,
            "total_segments": len(segments),
            "generated_at": datetime.now().isoformat(),
        },
    )

    response = make_response(pdf_content)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    if azure_url:
        response.headers["X-Azure-URL"] = azure_url
    return response
