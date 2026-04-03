"""
Azure Blueprint — /check_azure_files, /check_azure_file/<filename>, /azure_storage_info

Guards all routes with AZURE_AVAILABLE flag, set by attempting to import
azure_storage at module load time.
"""

from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import importlib
    import azure_storage as _azure_module

    importlib.reload(_azure_module)
    from azure_storage import get_storage_manager  # type: ignore[import]

    AZURE_AVAILABLE = True
    logger.info("Azure storage module loaded successfully.")
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning(
        "Azure storage not available. Install with: pip install azure-storage-blob"
    )

azure_bp = Blueprint("azure_bp", __name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@azure_bp.route("/check_azure_files", methods=["GET"])
@handle_route_errors
def check_azure_files():
    """Check what files have been uploaded to Azure."""
    if not AZURE_AVAILABLE:
        return jsonify({"error": "Azure storage not available"}), 500

    try:
        storage_manager = get_storage_manager()
        file_type = request.args.get("type", "all")

        files_info: dict = {}

        if file_type in ("research", "all"):
            files_info["research_files"] = storage_manager.list_files_in_container(
                "contoso", "pces/documents/research/"
            )

        if file_type in ("patient_summary", "all"):
            files_info["patient_summary_files"] = storage_manager.list_files_in_container(
                "contoso", "pces/documents/doc-patient-summary/"
            )

        if file_type in ("conversation", "all"):
            files_info["conversation_files"] = storage_manager.list_files_in_container(
                "contoso", "pces/documents/conversation/"
            )

        total_files = sum(len(v) for v in files_info.values())

        return jsonify(
            {
                "success": True,
                "total_files": total_files,
                "files": files_info,
                "message": f"Found {total_files} files in Azure storage",
            }
        )

    except Exception as exc:
        logger.error("Error checking Azure files: %s", exc)
        return jsonify({"error": f"Failed to check Azure files: {exc}"}), 500


@azure_bp.route("/check_azure_file/<path:filename>", methods=["GET"])
@handle_route_errors
def check_azure_file(filename: str):
    """Check if a specific file exists in Azure."""
    if not AZURE_AVAILABLE:
        return jsonify({"error": "Azure storage not available"}), 500

    try:
        storage_manager = get_storage_manager()
        container_name = request.args.get("container", "contoso")
        file_path = request.args.get(
            "path", f"pces/documents/research/{filename}"
        )

        exists = storage_manager.check_file_exists(container_name, file_path)

        if exists:
            file_info = storage_manager.get_file_metadata(container_name, file_path)
            return jsonify({"success": True, "exists": True, "file_info": file_info})
        else:
            return jsonify(
                {
                    "success": True,
                    "exists": False,
                    "message": f"File {filename} not found in Azure",
                }
            )

    except Exception as exc:
        logger.error("Error checking Azure file: %s", exc)
        return jsonify({"error": f"Failed to check Azure file: {exc}"}), 500


@azure_bp.route("/azure_storage_info", methods=["GET"])
@handle_route_errors
def azure_storage_info():
    """Get Azure storage configuration and status."""
    try:
        info: dict = {
            "azure_available": AZURE_AVAILABLE,
            "connection_configured": bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING")),
            "containers": {
                "contoso": {
                    "research_path": "pces/documents/research/",
                    "patient_summary_path": "pces/documents/doc-patient-summary/",
                    "conversations_path": "pces/documents/conversation/",
                }
            },
        }

        if AZURE_AVAILABLE and info["connection_configured"]:
            try:
                storage_manager = get_storage_manager()
                containers = [
                    c.name
                    for c in storage_manager.blob_service_client.list_containers()
                ]
                info["available_containers"] = containers
                info["connection_status"] = "Connected"
            except Exception as exc:
                info["connection_status"] = f"Connection failed: {exc}"
        else:
            info["connection_status"] = "Not configured"

        return jsonify(info)

    except Exception as exc:
        return jsonify({"error": f"Failed to get Azure info: {exc}"}), 500
