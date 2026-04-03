"""
PDF generation service.

Extracts the three PDF-generation functions from main.py
(generate_patient_pdf, generate_chat_pdf, generate_conversation_pdf) into
a reusable ``PDFService`` class that shares header, style, and Azure-upload
helpers.

All ``reportlab`` imports are kept *inside* the methods so that this module
can be safely imported even when ``reportlab`` is not installed.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from utils.error_handlers import get_logger

logger = get_logger(__name__)

# Root directory of the application (one level above this file's package)
_APP_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Azure availability flag — checked at module level so tests can patch it
try:
    from azure_storage import get_storage_manager
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class PDFService:
    """Service for generating medical-report PDFs using ReportLab.

    Each public ``generate_*`` method accepts a plain ``dict`` of report
    data, builds a PDF in memory, optionally uploads it to Azure Blob
    Storage, and returns ``(pdf_bytes, azure_url_or_empty_string)``.

    Azure uploads use the same routing logic as
    ``upload_pdf_to_azure()`` in *main.py* and are silently skipped when
    ``azure_storage`` is unavailable.
    """

    # ------------------------------------------------------------------
    # Shared style factory
    # ------------------------------------------------------------------

    def _get_styles(self) -> dict[str, Any]:
        """Build and return a dict of shared ReportLab ``ParagraphStyle`` objects.

        Returns:
            dict with keys: ``styles`` (the base stylesheet),
            ``title_style``, ``subtitle_style``, ``header_style``,
            ``session_style``, ``content_style``, ``citation_style``,
            ``section_header_style``, ``segment_style``,
            ``doctor_segment_style``, ``patient_segment_style``.
        """
        from reportlab.lib.colors import blue, green
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

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

        return {
            "styles": styles,
            "title_style": title_style,
            "subtitle_style": subtitle_style,
            "header_style": header_style,
            "session_style": session_style,
            "content_style": content_style,
            "citation_style": citation_style,
            "section_header_style": section_header_style,
            "segment_style": segment_style,
            "doctor_segment_style": doctor_segment_style,
            "patient_segment_style": patient_segment_style,
        }

    # ------------------------------------------------------------------
    # Shared header builder
    # ------------------------------------------------------------------

    def _build_header(
        self,
        story: list,
        main_title: str,
        subtitle: str,
        style_map: dict[str, Any],
    ) -> None:
        """Append a logo-title header block to *story*.

        Tries to load ``ClientLogo101.png`` from the application root and
        positions it in a two-column table (title left, logo right).  Falls
        back to a plain title paragraph when the logo file is absent or
        unreadable.

        Args:
            story: ReportLab story list to append elements to.
            main_title: Main heading text rendered on the left.
            subtitle: Subtitle text rendered below the header row.
            style_map: dict returned by :py:meth:`_get_styles`.
        """
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

        title_style = style_map["title_style"]
        subtitle_style = style_map["subtitle_style"]

        logo_path = os.path.join(_APP_DIR, "ClientLogo101.png")

        if os.path.exists(logo_path):
            try:
                logo_img = Image(logo_path, width=1.5 * inch, height=0.9 * inch)
                title_paragraph = Paragraph(main_title, title_style)

                header_data = [[title_paragraph, logo_img]]
                header_table = Table(header_data, colWidths=[4.5 * inch, 2 * inch])
                header_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (0, 0), "LEFT"),
                            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ]
                    )
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

        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # Azure upload helper
    # ------------------------------------------------------------------

    def _upload_to_azure(
        self,
        pdf_bytes: bytes,
        filename: str,
        pdf_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Upload *pdf_bytes* to Azure Blob Storage.

        Silently returns ``None`` when ``azure_storage`` is unavailable or
        the upload fails.

        Args:
            pdf_bytes: Raw PDF content.
            filename: Blob filename (including ``.pdf`` extension).
            pdf_type: One of ``'patient_summary'``, ``'research'``, or
                ``'conversation'`` – controls which storage container and
                helper method is called.
            metadata: Optional dict of string metadata tags.

        Returns:
            The Azure Blob URL as a string, or ``None`` on failure.
        """
        if not AZURE_AVAILABLE:
            return None

        metadata = metadata or {}
        try:
            storage_manager = get_storage_manager()

            if pdf_type == "research":
                return storage_manager.upload_research_pdf(
                    pdf_bytes,
                    filename,
                    metadata.get("patient_problem"),
                    metadata,
                )
            elif pdf_type == "patient_summary":
                patient_data = {
                    "patient_name": metadata.get("patient_name"),
                    "patient_id": metadata.get("patient_id"),
                    "doctor_name": metadata.get("doctor_name"),
                    "session_date": metadata.get("date_time"),
                }
                return storage_manager.upload_patient_summary_pdf(
                    pdf_bytes, filename, patient_data, metadata
                )
            elif pdf_type == "conversation":
                conversation_data = {
                    "doctor_name": metadata.get("doctor_name"),
                    "patient_name": metadata.get("patient_name"),
                    "duration": metadata.get("duration", "Unknown"),
                    "session_date": metadata.get("date_time"),
                }
                return storage_manager.upload_conversation_pdf(
                    pdf_bytes, filename, conversation_data, metadata
                )
        except Exception as exc:
            logger.error("Azure upload failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Markdown → ReportLab converter (used by generate_chat_pdf)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_markdown_to_reportlab(text: str) -> str:
        """Convert the enhanced-tools HTML/markdown response format to
        ReportLab-compatible markup.

        Handles both HTML (``<div>``/``<h4>`` structure produced by
        ``enhanced_tools.py``) and plain-text section-header format.

        Args:
            text: Raw message content string.

        Returns:
            String containing ReportLab XML markup (``<b>``, ``<br/>``, etc.).
        """
        import re

        # ---- HTML path ------------------------------------------------
        if "<div" in text and "<h4" in text:
            try:
                from bs4 import BeautifulSoup

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
                    srcs = [s.strip() for s in sections["source"].split("\n") if s.strip()]
                    if srcs:
                        formatted_parts.append(
                            "<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(srcs)
                        )

                if sections["tool_routing"]:
                    routing_html = sections["tool_routing"]
                    routing_soup = BeautifulSoup(routing_html, "html.parser")
                    routing_text = routing_soup.get_text()

                    confidence = ""
                    tools_used = ""
                    reasoning = ""

                    lines_split = routing_text.replace("\n", " ").split("Tools Used:")
                    if len(lines_split) >= 2:
                        confidence_part = lines_split[0]
                        confidence_match = re.search(
                            r"Confidence:\s*(.+?)$",
                            confidence_part.strip(),
                            re.IGNORECASE,
                        )
                        if confidence_match:
                            confidence = confidence_match.group(1).strip()

                        rest = lines_split[1]
                        reasoning_split = rest.split("Reasoning:")
                        if len(reasoning_split) >= 2:
                            tools_used = reasoning_split[0].strip()
                            reasoning = reasoning_split[1].strip()
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
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(text, "html.parser")
                    return soup.get_text().strip()
                except Exception:
                    return text

        # ---- Plain-text path ------------------------------------------
        sections_pt: dict[str, str] = {
            "answer": "",
            "source": "",
            "tool_routing": "",
        }

        current_section: str | None = None

        for raw_line in text.split("\n"):
            line = raw_line.strip()
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
                if sections_pt[current_section]:
                    sections_pt[current_section] += " " + line
                else:
                    sections_pt[current_section] = line

        formatted_parts_pt: list[str] = []

        if sections_pt["answer"]:
            formatted_parts_pt.append(sections_pt["answer"])

        if sections_pt["source"]:
            srcs = [s.strip() for s in sections_pt["source"].split("\n") if s.strip()]
            if srcs:
                formatted_parts_pt.append(
                    "<br/><br/><b>Sources:</b><br/>• " + "<br/>• ".join(srcs)
                )

        if sections_pt["tool_routing"]:
            formatted_parts_pt.append(
                f"<br/><br/><b>Tool Selection & Query Routing:</b><br/>"
                f"{sections_pt['tool_routing']}"
            )

        # If no structured sections found, treat as plain text
        if not any(sections_pt.values()):
            formatted_parts_pt = [text]

        import re

        full_text_pt = "".join(formatted_parts_pt)
        full_text_pt = re.sub(r"\s+", " ", full_text_pt)
        return full_text_pt.strip()

    # ------------------------------------------------------------------
    # Public generate methods
    # ------------------------------------------------------------------

    def generate_patient_pdf(self, data: dict[str, Any]) -> tuple[bytes, str]:
        """Generate a patient recording notes PDF.

        Args:
            data: dict with keys:

                * ``doctorName`` (str)
                * ``patientName`` (str)
                * ``patientId`` (str)
                * ``dateTime`` (str) – formatted date/time string
                * ``transcription`` (str) – Whisper output
                * ``summary`` (str) – AI medical summary
                * ``conclusion`` (str) – AI conclusion / recommendations
                * ``patientProblem`` (str, optional) – chief complaint

        Returns:
            ``(pdf_bytes, azure_url)`` where *azure_url* is an empty string
            when Azure is unavailable or the upload fails.
        """
        import io

        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        from reportlab.lib.pagesizes import A4

        doctor_name: str = data.get("doctorName", "")
        patient_name: str = data.get("patientName", "")
        patient_id: str = data.get("patientId", "")
        date_time: str = data.get("dateTime", "")
        transcription: str = data.get("transcription", "")
        summary: str = data.get("summary", "")
        conclusion: str = data.get("conclusion", "")
        patient_problem: str = data.get("patientProblem", "")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )

        style_map = self._get_styles()
        styles = style_map["styles"]
        header_style = style_map["header_style"]
        session_style = style_map["session_style"]

        story: list = []

        patient_display_name = patient_name if patient_name else "Patient Name"
        main_title = f"Patient – {patient_display_name} – Recording Notes"

        self._build_header(story, main_title, "Patient Recording Notes", style_map)

        # Patient / doctor info
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

        # Session information
        story.append(Paragraph("Session Information", header_style))
        story.append(
            Paragraph(
                "Transcription Engine: Whisper; Summary Engine: OpenAI",
                session_style,
            )
        )
        display_date = date_time if date_time else datetime.now().strftime(
            "%m/%d/%Y, %I:%M:%S %p"
        )
        story.append(Paragraph(f"Date: {display_date}", session_style))
        story.append(Spacer(1, 15))

        # Patient problem (optional)
        if patient_problem:
            story.append(Paragraph("Patient Problem", header_style))
            story.append(Paragraph(patient_problem, styles["Normal"]))
            story.append(Spacer(1, 15))

        # Original transcription
        story.append(Paragraph("Original Transcription Text", header_style))
        story.append(
            Paragraph(
                transcription if transcription else "No transcription available",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 15))

        # Medical summary
        story.append(Paragraph("Medical Summary", header_style))
        story.append(
            Paragraph(
                summary if summary else "No summary available",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 15))

        # Conclusion & recommendations
        story.append(Paragraph("Conclusion & Recommendations", header_style))
        story.append(
            Paragraph(
                conclusion if conclusion else "No conclusion available",
                styles["Normal"],
            )
        )

        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()

        # Azure upload
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"

        azure_url = self._upload_to_azure(
            pdf_bytes,
            filename,
            "patient_summary",
            metadata={
                "doctor_name": doctor_name,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "date_time": date_time,
                "patient_problem": patient_problem,
                "pdf_type": "patient_notes",
                "generated_at": datetime.now().isoformat(),
            },
        )

        logger.info("Generated patient PDF: %s (azure_url=%s)", filename, azure_url)
        return pdf_bytes, azure_url or ""

    def generate_chat_pdf(self, data: dict[str, Any]) -> tuple[bytes, str]:
        """Generate a chat / research session PDF.

        Accepts the message list under either ``chatHistory`` or ``messages``
        (the legacy field name used in main.py) so the service is compatible
        with existing request data without requiring main.py changes.

        Args:
            data: dict with keys:

                * ``doctorName`` (str)
                * ``patientName`` (str, optional)
                * ``patientId`` (str, optional)
                * ``dateTime`` (str, optional)
                * ``chatHistory`` / ``messages`` – list of
                  ``{"role": str, "content": str}`` dicts
                * ``summary`` (str, optional)
                * ``patientProblem`` (str, optional)
                * ``jsonData`` (str, optional)

        Returns:
            ``(pdf_bytes, azure_url)`` where *azure_url* is an empty string
            when Azure is unavailable or the upload fails.
        """
        import io

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doctor_name: str = data.get("doctorName", "Dr. Name")
        patient_name: str = data.get("patientName", "")
        patient_id: str = data.get("patientId", "")
        patient_problem: str = data.get("patientProblem", "")
        # Accept both field names for backward compatibility
        messages: list[dict] = data.get("chatHistory") or data.get("messages", [])
        json_data: str = data.get("jsonData", "")

        now = datetime.now()
        formatted_date = data.get("dateTime") or now.strftime("%m/%d/%Y, %I:%M:%S %p")
        timestamp = now.strftime("%Y%m%d%H%M")

        if patient_name:
            filename = (
                f"{doctor_name.upper().replace(' ', '')}"
                f"-{patient_name.upper().replace(' ', '')}"
                f"-{timestamp}.pdf"
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

        style_map = self._get_styles()
        styles = style_map["styles"]
        header_style = style_map["header_style"]
        session_style = style_map["session_style"]
        content_style = style_map["content_style"]
        section_header_style = style_map["section_header_style"]

        story: list = []

        patient_display_name = patient_name if patient_name else "Patient Name"
        main_title = f"Patient – {patient_display_name} – Research"

        self._build_header(story, main_title, "Patient Recording Notes", style_map)

        # Patient / doctor info
        if patient_id:
            patient_info = f"Patient Name: {patient_display_name} – Patient ID: {patient_id}"
        else:
            patient_info = f"Patient Name: {patient_display_name}"
        doctor_info = f"Doctor's Name: {doctor_name if doctor_name else 'Unknown Doctor'}"

        story.append(Paragraph(patient_info, styles["Normal"]))
        story.append(Spacer(1, 5))
        story.append(Paragraph(doctor_info, styles["Normal"]))
        story.append(Spacer(1, 15))

        # Session information
        story.append(Paragraph("Session Information", header_style))
        story.append(Paragraph(f"Date: {formatted_date}", session_style))
        story.append(Spacer(1, 12))

        # Patient problem (if provided)
        if patient_problem:
            logger.info("Adding Patient Problem section: '%s'", patient_problem)
            story.append(Paragraph("Patient Problem", header_style))
            story.append(Paragraph(patient_problem, styles["Normal"]))
            story.append(Spacer(1, 12))
        else:
            logger.info("No patient problem provided, skipping section")

        # JSON data section (if provided)
        if json_data:
            story.append(Paragraph("JSON Data:", section_header_style))
            story.append(Paragraph(json_data, content_style))
            story.append(Spacer(1, 12))

        # Conversation messages
        story.append(Paragraph("Conversation", header_style))
        story.append(Spacer(1, 6))

        for message in messages:
            role: str = message.get("role", "")
            content: str = message.get("content", "")

            formatted_content = self._convert_markdown_to_reportlab(content)

            if role == "user":
                story.append(Paragraph("Doctor Input:", section_header_style))
                story.append(Paragraph(formatted_content, content_style))
                story.append(Spacer(1, 8))
            elif role == "ai":
                story.append(Paragraph("System Output:", section_header_style))
                story.append(Paragraph(formatted_content, content_style))
                story.append(Spacer(1, 8))
            else:
                story.append(
                    Paragraph(f"{role.title()}:", section_header_style)
                )
                story.append(Paragraph(formatted_content, content_style))
                story.append(Spacer(1, 8))

        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()

        # Azure upload
        azure_url = self._upload_to_azure(
            pdf_bytes,
            filename,
            "research",
            metadata={
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

        logger.info("Generated chat PDF: %s (azure_url=%s)", filename, azure_url)
        return pdf_bytes, azure_url or ""

    def generate_conversation_pdf(self, data: dict[str, Any]) -> tuple[bytes, str]:
        """Generate a doctor-patient conversation / voice-diarization PDF.

        Accepts both the new spec field names and the legacy field names used
        by the existing ``/generate_conversation_pdf`` route so that no
        changes to main.py are required.

        **Field mapping:**

        * ``conversation`` or ``segments`` → list of speaker segments
        * ``translation`` or ``fullTranscript`` → complete raw transcript text
        * ``conclusion`` → conclusion text (not in spec but used by existing route)
        * ``processingInfo`` → processing metadata dict
        * ``isDuplicate`` → bool flag

        Args:
            data: dict with keys:

                * ``doctorName`` (str)
                * ``patientName`` (str)
                * ``patientId`` (str, optional)
                * ``dateTime`` (str, optional)
                * ``conversation`` / ``segments`` – list of segment dicts
                * ``summary`` (str, optional)
                * ``translation`` / ``fullTranscript`` (str, optional)
                * ``conclusion`` (str, optional)
                * ``processingInfo`` (dict, optional)
                * ``isDuplicate`` (bool, optional)

        Returns:
            ``(pdf_bytes, azure_url)`` where *azure_url* is an empty string
            when Azure is unavailable or the upload fails.
        """
        import io

        from reportlab.lib.colors import blue
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doctor_name: str = data.get("doctorName", "Doctor")
        patient_name: str = data.get("patientName", "Patient")
        date_time: str = data.get("dateTime", "")
        # Accept both spec field names and legacy field names
        segments: list[dict] = data.get("conversation") or data.get("segments", [])
        full_transcript: str = (
            data.get("translation") or data.get("fullTranscript", "")
        )
        summary: str = data.get("summary", "")
        conclusion: str = data.get("conclusion", "")
        processing_info: dict = data.get("processingInfo", {})
        is_duplicate: bool = bool(data.get("isDuplicate", False))

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )

        style_map = self._get_styles()
        styles = style_map["styles"]
        header_style = style_map["header_style"]
        doctor_segment_style = style_map["doctor_segment_style"]
        patient_segment_style = style_map["patient_segment_style"]

        story: list = []

        # Header
        duplicate_text = " (Duplicate)" if is_duplicate else ""
        main_title = f"Doctor-Patient Conversation Analysis{duplicate_text}"
        subtitle_style_conv = style_map["subtitle_style"]
        # Override subtitle textColor to blue for conversation PDF (matches original)
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        subtitle_conv = ParagraphStyle(
            "ConvSubtitle",
            parent=subtitle_style_conv,
            textColor=blue,
        )
        style_map_conv = dict(style_map)
        style_map_conv["subtitle_style"] = subtitle_conv

        self._build_header(
            story,
            main_title,
            "Voice Diarization & Transcription Results",
            style_map_conv,
        )

        # Participant information
        story.append(Paragraph(f"Doctor: {doctor_name}", styles["Normal"]))
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
        story.append(Spacer(1, 15))

        # Processing information
        story.append(Paragraph("Processing Information", header_style))

        total_segments_count = processing_info.get("totalSegments", len(segments))
        language_used: str = processing_info.get("language", "English")

        language_display: dict[str, str] = {
            "en": "English",
            "es": "Spanish",
            "zh": "Chinese",
            "yue": "Cantonese",
            "tl": "Tagalog",
            "hi": "Hindi",
            "te": "Telugu",
            "ta": "Tamil",
            "gu": "Gujarati",
            "pa": "Punjabi",
        }
        language_name = language_display.get(
            language_used,
            language_used.title() if language_used else "English",
        )

        story.append(
            Paragraph(
                "Voice Diarization: OpenAI-based Speaker Separation  "
                "Transcription Engine: Whisper",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                f"Language Majorly Spoken: {language_name}   "
                f"Total Segments: {total_segments_count}",
                styles["Normal"],
            )
        )

        display_date = date_time if date_time else datetime.now().strftime(
            "%m/%d/%Y, %I:%M:%S %p"
        )
        story.append(
            Paragraph(f"Processing Date: {display_date}", styles["Normal"])
        )
        story.append(Spacer(1, 15))

        # Conversation segments
        if segments:
            story.append(Paragraph("Conversation Segments", header_style))

            for segment in segments:
                role: str = segment.get("role", "Unknown")
                text: str = segment.get("text", "")
                start_time = segment.get("start", "")
                end_time = segment.get("end", "")
                confidence = segment.get("confidence", 0)

                timing_info = (
                    f"[{start_time} - {end_time}]"
                    if start_time and end_time
                    else ""
                )
                confidence_info = (
                    f"(Confidence: {int(confidence * 100)}%)"
                    if confidence > 0
                    else ""
                )
                header_text = f"{role} {timing_info} {confidence_info}"

                if role.lower() == "doctor":
                    story.append(
                        Paragraph(f"<b>{header_text}</b>", doctor_segment_style)
                    )
                    story.append(Paragraph(text, doctor_segment_style))
                else:
                    story.append(
                        Paragraph(f"<b>{header_text}</b>", patient_segment_style)
                    )
                    story.append(Paragraph(text, patient_segment_style))

                story.append(Spacer(1, 8))

            story.append(Spacer(1, 15))

        # Complete transcript
        story.append(Paragraph("Complete Transcript", header_style))
        story.append(
            Paragraph(
                full_transcript if full_transcript else "No transcript available",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 15))

        # Medical summary (optional)
        if summary:
            story.append(Paragraph("Medical Summary", header_style))
            story.append(Paragraph(summary, styles["Normal"]))
            story.append(Spacer(1, 15))

        # Conclusion & recommendations (optional)
        if conclusion:
            story.append(Paragraph("Conclusion & Recommendations", header_style))
            story.append(Paragraph(conclusion, styles["Normal"]))
            story.append(Spacer(1, 15))

        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        duplicate_suffix = "-DUPLICATE" if is_duplicate else ""
        filename = (
            f"CONVERSATION-{doctor_name.upper().replace(' ', '')}"
            f"{duplicate_suffix}-{timestamp}.pdf"
        )

        # Azure upload
        azure_url = self._upload_to_azure(
            pdf_bytes,
            filename,
            "conversation",
            metadata={
                "doctor_name": doctor_name,
                "patient_name": patient_name,
                "date_time": date_time,
                "summary": summary if summary else "",
                "conclusion": conclusion if conclusion else "",
                "pdf_type": "conversation_segments",
                "is_duplicate": str(is_duplicate),
                "total_segments": str(len(segments)),
                "generated_at": datetime.now().isoformat(),
            },
        )

        logger.info(
            "Generated conversation PDF: %s (azure_url=%s)", filename, azure_url
        )
        return pdf_bytes, azure_url or ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
pdf_service = PDFService()
