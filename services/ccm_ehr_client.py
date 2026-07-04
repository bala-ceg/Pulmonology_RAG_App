"""
CCM / EHR External API Client
==============================
Thin wrapper around the CCM/EHR REST API running at
``CCM_EHR_BASE_URL`` (default: http://4.155.102.23:8888/api/v1).

All search endpoints require a Bearer JWT supplied via ``CCM_EHR_TOKEN``
in the environment.  When the token is absent or expired the client raises
``CCMEHRAuthError`` so callers can return a graceful degradation.

Endpoints implemented:
  GET /api/v1/health                       — liveness (no auth)
  GET /api/v1/patients/search              — patient search
  GET /api/v1/hospitals/search             — hospital search

All methods raise:
  CCMEHRAuthError   — 401 (missing / expired token)
  CCMEHRError       — any other non-200 response or network failure
"""

from __future__ import annotations

import requests
from requests.exceptions import ConnectionError, Timeout

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)

_TIMEOUT = 10  # seconds


class CCMEHRError(Exception):
    """Non-auth error from the CCM/EHR API."""


class CCMEHRAuthError(CCMEHRError):
    """401 Unauthorized — token missing or expired."""


class CCMEHRClient:
    """Stateless HTTP client for the CCM/EHR external API.

    Reads ``CCM_EHR_BASE_URL`` and ``CCM_EHR_TOKEN`` from Config at
    call time so that a token rotation mid-run takes effect immediately.
    """

    def _headers(self) -> dict[str, str]:
        token = Config.CCM_EHR_TOKEN
        if not token:
            raise CCMEHRAuthError(
                "CCM_EHR_TOKEN is not set in the environment. "
                "Add it to .env and restart the server."
            )
        return {"Authorization": f"Bearer {token}"}

    def _base(self) -> str:
        return Config.CCM_EHR_BASE_URL.rstrip("/")

    def _get(self, path: str, params: dict | None = None, auth: bool = True) -> dict:
        url = f"{self._base()}{path}"
        headers = self._headers() if auth else {}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        except (ConnectionError, Timeout) as exc:
            raise CCMEHRError(f"CCM/EHR API unreachable: {exc}") from exc

        if resp.status_code == 401:
            raise CCMEHRAuthError(
                "CCM/EHR API returned 401 — token is missing or expired. "
                "Update CCM_EHR_TOKEN in .env and restart."
            )
        if resp.status_code == 404:
            return {"success": True, "count": 0, "data": []}
        if not resp.ok:
            raise CCMEHRError(
                f"CCM/EHR API error {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Call GET /api/v1/health — no auth required."""
        return self._get("/health", auth=False)

    # ------------------------------------------------------------------
    # Patient search  — GET /api/v1/patients/search
    # ------------------------------------------------------------------

    def search_patients(
        self,
        *,
        first_name: str = "",
        last_name: str = "",
        date_of_birth: str = "",
        phone: str = "",
        email: str = "",
    ) -> list[dict]:
        """Search patients via the CCM/EHR API.

        At least one parameter must be non-empty (API requirement).
        Returns a list of patient dicts normalised to PCES field names:
        ``patient_id``, ``first_name``, ``last_name``, ``full_name``,
        ``date_of_birth``, ``phone``, ``email``, ``is_active``.

        Raises:
            CCMEHRAuthError  — 401 from the API
            CCMEHRError      — other API / network failure
            ValueError       — no search parameters provided
        """
        params: dict[str, str] = {}
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if date_of_birth:
            params["date_of_birth"] = date_of_birth
        if phone:
            params["phone"] = phone
        if email:
            params["email"] = email

        if not params:
            raise ValueError("At least one search parameter is required.")

        logger.debug("CCM/EHR patient search: %s", params)
        payload = self._get("/patients/search", params=params)

        raw: list[dict] = payload.get("data", [])
        results: list[dict] = []
        for item in raw:
            fn = item.get("first_name") or ""
            ln = item.get("last_name") or ""
            results.append({
                "patient_id":    str(item.get("party_id") or item.get("par_row_id") or ""),
                "first_name":    fn,
                "last_name":     ln,
                "full_name":     f"{fn} {ln}".strip(),
                "date_of_birth": item.get("date_of_birth") or "",
                "phone":         item.get("phone") or "",
                "email":         item.get("email") or "",
                "is_active":     item.get("is_active", True),
            })

        logger.info(
            "CCM/EHR patient search %s → %d result(s)", params, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Hospital search  — GET /api/v1/hospitals/search
    # ------------------------------------------------------------------

    def search_hospitals(
        self,
        *,
        firm_name: str = "",
        location: str = "",
        hospital_code: str = "",
        phone: str = "",
        email: str = "",
    ) -> list[dict]:
        """Search hospitals via the CCM/EHR API.

        Returns a list of hospital dicts with keys:
        ``hospital_id``, ``firm_name``, ``location``,
        ``hospital_code``, ``phone``, ``email``.

        Raises:
            CCMEHRAuthError  — 401 from the API
            CCMEHRError      — other API / network failure
            ValueError       — no search parameters provided
        """
        params: dict[str, str] = {}
        if firm_name:
            params["firm_name"] = firm_name
        if location:
            params["location"] = location
        if hospital_code:
            params["hospital_code"] = hospital_code
        if phone:
            params["phone"] = phone
        if email:
            params["email"] = email

        if not params:
            raise ValueError("At least one search parameter is required.")

        logger.debug("CCM/EHR hospital search: %s", params)
        payload = self._get("/hospitals/search", params=params)

        raw: list[dict] = payload.get("data", [])
        results: list[dict] = []
        for item in raw:
            results.append({
                "hospital_id":   str(item.get("party_id") or item.get("par_row_id") or ""),
                "firm_name":     item.get("firm_name") or "",
                "location":      item.get("location") or "",
                "hospital_code": item.get("hospital_code") or "",
                "phone":         item.get("phone") or "",
                "email":         item.get("email") or "",
            })

        logger.info(
            "CCM/EHR hospital search %s → %d result(s)", params, len(results)
        )
        return results


# Module-level singleton
ccm_ehr_client = CCMEHRClient()
