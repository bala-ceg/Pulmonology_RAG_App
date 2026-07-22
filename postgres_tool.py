"""
PostgreSQL Database Tool for Medical Diagnosis Data
===================================================

This module provides functionality to connect to PostgreSQL database and fetch
medical diagnosis data from the p_diagnosis table for the agentic AI system.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgreSQLTool:
    """PostgreSQL database tool for medical diagnosis data retrieval"""
    
    def __init__(self):
        """Initialize the PostgreSQL connection parameters"""
        # Use PostgreSQL tool-specific environment variables
        self.host = os.getenv('PG_TOOL_HOST')
        self.port = os.getenv('PG_TOOL_PORT', '5432')
        self.database = os.getenv('PG_TOOL_NAME')
        self.user = os.getenv('PG_TOOL_USER')
        self.password = os.getenv('PG_TOOL_PASSWORD')
        
        if not all([self.host, self.database, self.user, self.password]):
            raise ValueError("Missing required PostgreSQL tool connection parameters in .env file. Please ensure PG_TOOL_HOST, PG_TOOL_NAME, PG_TOOL_USER, and PG_TOOL_PASSWORD are set.")
    
    def get_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=RealDictCursor
            )
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def test_connection(self) -> Dict[str, str]:
        """Test the database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
                    return {
                        'status': 'success',
                        'message': f"Successfully connected to PostgreSQL: {version['version']}"
                    }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to connect to database: {str(e)}"
            }
    
    def fetch_diagnosis_descriptions(self, search_term: str = None, limit: int = 10) -> Dict[str, str]:
        """
        Fetch diagnosis descriptions from p_diagnosis table with enhanced formatting and LLM summary
        
        Args:
            search_term: Optional search term to filter descriptions
            limit: Maximum number of results to return
            
        Returns:
            Dict containing content, summary, citations, and tool_info
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if search_term:
                        # Search for descriptions containing the search term
                        query = """
                        SELECT diagnosis_id, description, code, hospital_id, encounter_id, created_at, updated_at
                        FROM p_diagnosis 
                        WHERE LOWER(description) LIKE LOWER(%s) AND is_active = TRUE AND deleted_at IS NULL
                        ORDER BY created_at DESC
                        LIMIT %s
                        """
                        cursor.execute(query, (f'%{search_term}%', limit))
                    else:
                        # Fetch all diagnosis descriptions
                        query = """
                        SELECT diagnosis_id, description, code, hospital_id, encounter_id, created_at, updated_at
                        FROM p_diagnosis 
                        WHERE is_active = TRUE AND deleted_at IS NULL
                        ORDER BY created_at DESC
                        LIMIT %s
                        """
                        cursor.execute(query, (limit,))
                    
                    results = cursor.fetchall()
                    
                    if not results:
                        return {
                            'content': f"No diagnosis descriptions found{' for search term: ' + search_term if search_term else ''}",
                            'summary': "No medical diagnosis data found in the database.",
                            'citations': "Database: pces_ehr_ccm.p_diagnosis",
                            'tool_info': "<div class='tool-route'><strong>🗄️ PostgreSQL Database Tool</strong><br>Source: p_diagnosis table</div>"
                        }
                    
                    # Format the results in a more consistent, professional way like other tools
                    content_parts = []
                    diagnosis_codes = []
                    
                    for i, row in enumerate(results, 1):
                        # Collect diagnosis codes for summary
                        if row['code']:
                            diagnosis_codes.append(row['code'])
                        
                        # Format each diagnosis entry more professionally
                        content_parts.append(f"""**{i}. {row['description']} (Code: {row['code'] or 'N/A'})**

📋 **Diagnosis Details:**
- **Diagnosis ID:** {row['diagnosis_id']}
- **Medical Code:** {row['code'] or 'N/A'}
- **Description:** {row['description']}
- **Hospital ID:** {row['hospital_id']}
- **Encounter ID:** {row['encounter_id']}
- **Created:** {row['created_at']}
- **Updated:** {row['updated_at'] or 'N/A'}

""")
                    
                    # Join content with proper spacing
                    raw_content = "\n".join(content_parts)
                    
                    # Generate LLM summary using the enhanced tools pattern
                    summary = self._generate_diagnosis_summary(raw_content, search_term, len(results), diagnosis_codes)
                    
                    # Format citations consistently with other tools
                    citations = f"Database: {self.database}.p_diagnosis (PostgreSQL)"
                    
                    # Format tool routing info consistently with enhanced_tools.py
                    tool_info = self._format_tool_routing_html(
                        primary_tool="PostgreSQL_Diagnosis_Search",
                        confidence="high",
                        tools_used=["PostgreSQL_Diagnosis_Search"],
                        reasoning="Query contains database-specific keywords; PostgreSQL selected for diagnosis data retrieval",
                        result_count=len(results)
                    )
                    
                    return {
                        'content': raw_content,
                        'summary': summary,
                        'citations': citations,
                        'tool_info': tool_info
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching diagnosis descriptions: {str(e)}")
            return {
                'content': f"Error retrieving diagnosis data: {str(e)}",
                'summary': "Database query failed due to technical error.",
                'citations': "Error accessing database",
                'tool_info': "<div class='tool-route'><strong>❌ PostgreSQL Database Tool</strong><br>Status: Error</div>"
            }
    
    def search_diagnosis_by_keyword(self, keyword: str, limit: int = 20) -> Dict[str, str]:
        """
        Search for specific diagnosis descriptions by keyword
        
        Args:
            keyword: Keyword to search for in diagnosis descriptions
            limit: Maximum number of results to return
            
        Returns:
            Dict containing formatted diagnosis data
        """
        return self.fetch_diagnosis_descriptions(search_term=keyword, limit=limit)
    
    def get_diagnosis_by_code(self, code: str) -> Dict[str, str]:
        """
        Get diagnosis information by specific code
        
        Args:
            code: Diagnosis code to search for
            
        Returns:
            Dict containing diagnosis information
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                    SELECT diagnosis_id, description, code, hospital_id, encounter_id, created_at, updated_at
                    FROM p_diagnosis 
                    WHERE LOWER(code) = LOWER(%s) AND is_active = TRUE AND deleted_at IS NULL
                    """
                    cursor.execute(query, (code,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return {
                            'content': f"No diagnosis found with code: {code}",
                            'summary': f"Diagnosis code '{code}' not found in the database.",
                            'citations': f"Database: {self.database}.p_diagnosis",
                            'tool_info': "<div class='tool-route'><strong>🗄️ PostgreSQL Database Tool</strong><br>Query: Diagnosis by code</div>"
                        }
                    
                    content = f"""
**Diagnosis Code:** {result['code']}
**Diagnosis ID:** {result['diagnosis_id']}
**Description:** {result['description']}
**Hospital ID:** {result['hospital_id']}
**Encounter ID:** {result['encounter_id']}
**Created:** {result['created_at']}
**Updated:** {result['updated_at'] or 'N/A'}
"""
                    
                    summary = f"Found diagnosis with code '{code}': {result['description']}"
                    citations = f"Database: {self.database}.p_diagnosis (PostgreSQL)"
                    tool_info = "<div class='tool-route'><strong>🗄️ PostgreSQL Database Tool</strong><br>Query: Diagnosis by specific code</div>"
                    
                    return {
                        'content': content,
                        'summary': summary,
                        'citations': citations,
                        'tool_info': tool_info
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching diagnosis by code: {str(e)}")
            return {
                'content': f"Error retrieving diagnosis by code '{code}': {str(e)}",
                'summary': "Database query failed due to technical error.",
                'citations': "Error accessing database",
                'tool_info': "<div class='tool-route'><strong>❌ PostgreSQL Database Tool</strong><br>Status: Error</div>"
            }

    def fetch_patient_history(
        self,
        patient_id: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Fetch full patient medical history via p_party ↔ p_encounter ↔ p_diagnosis.

        Looks up encounters (hospital visits) and associated diagnoses /
        medications for a specific patient.  Accepts either party_id or
        first_name + last_name (at least one must be supplied).

        Returns:
            Dict with keys: content, summary, citations, tool_info  (same
            contract as fetch_diagnosis_descriptions so callers are uniform).
        """
        if not patient_id and not (first_name or last_name):
            return {
                'content': "No patient identifier supplied — provide party_id or first/last name.",
                'summary': "Unable to fetch patient history: no patient identifier.",
                'citations': f"Database: {self.database}.p_encounter + p_diagnosis",
                'tool_info': "<div class='tool-route'><strong>⚠️ Patient History Tool</strong><br>Error: no identifier</div>",
            }

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Build WHERE clause dynamically
                    if patient_id:
                        where_clause = "A.party_id = %s"
                        params: tuple = (patient_id, limit)
                    else:
                        # Name-based lookup (case-insensitive)
                        where_parts = []
                        param_list: list = []
                        if first_name:
                            where_parts.append("LOWER(A.first_name) = LOWER(%s)")
                            param_list.append(first_name)
                        if last_name:
                            where_parts.append("LOWER(A.last_name) = LOWER(%s)")
                            param_list.append(last_name)
                        where_clause = " AND ".join(where_parts)
                        param_list.append(limit)
                        params = tuple(param_list)

                    sql = f"""
                        SELECT
                            A.party_id,
                            A.first_name,
                            COALESCE(A.middle_name, '') AS middle_name,
                            A.last_name,
                            A.date_of_birth,
                            A.gender,
                            B.encounter_id,
                            B.encounter_type,
                            B.notes AS encounter_notes,
                            C.code          AS diagnosis_code,
                            C.description   AS diagnosis_description
                        FROM p_party A
                        JOIN p_encounter B ON A.party_id = B.patient_id
                        JOIN p_diagnosis C ON B.encounter_id = C.encounter_id
                        WHERE {where_clause}
                          AND A.party_type = 'PATIENT'
                        ORDER BY B.encounter_id DESC
                        LIMIT %s
                    """
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()

                    if not rows:
                        label = f"patient_id={patient_id}" if patient_id else f"{first_name} {last_name}"
                        return {
                            'content': f"No medical history records found for {label}.",
                            'summary': f"No encounter or diagnosis history found for {label} in the database.",
                            'citations': f"Database: {self.database}.p_encounter + p_diagnosis",
                            'tool_info': "<div class='tool-route'><strong>🗄️ Patient History Tool</strong><br>Result: No records</div>",
                        }

                    # Build human-readable content grouped by encounter
                    seen_encounters: dict = {}
                    for row in rows:
                        enc_id = row['encounter_id']
                        if enc_id not in seen_encounters:
                            seen_encounters[enc_id] = {
                                'patient_name': f"{row['first_name']} {row['middle_name']} {row['last_name']}".strip(),
                                'dob': row['date_of_birth'],
                                'gender': row['gender'],
                                'encounter_type': row['encounter_type'],
                                'notes': row['encounter_notes'],
                                'diagnoses': [],
                            }
                        seen_encounters[enc_id]['diagnoses'].append({
                            'code': row['diagnosis_code'],
                            'description': row['diagnosis_description'],
                        })

                    content_parts = []
                    patient_name_label = list(seen_encounters.values())[0]['patient_name']
                    for i, (enc_id, enc) in enumerate(seen_encounters.items(), 1):
                        diag_lines = "\n".join(
                            f"    • [{d['code']}] {d['description']}"
                            for d in enc['diagnoses']
                        )
                        content_parts.append(
                            f"**Encounter {i} (ID: {enc_id}) — {enc['encounter_type'] or 'General Visit'}**\n"
                            f"  Visit Notes: {enc['notes'] or 'No notes recorded'}\n"
                            f"  Diagnoses & Medications:\n{diag_lines}"
                        )

                    full_content = (
                        f"**Patient:** {patient_name_label}\n"
                        f"**DOB:** {list(seen_encounters.values())[0]['dob']}  "
                        f"**Gender:** {list(seen_encounters.values())[0]['gender']}\n\n"
                        + "\n\n".join(content_parts)
                    )

                    summary = (
                        f"Found {len(seen_encounters)} encounter(s) for {patient_name_label} "
                        f"with a total of {len(rows)} diagnosis/medication entries. "
                        "Records include visit notes, diagnosis codes, and medication descriptions "
                        "retrieved from the PCES EHR system."
                    )

                    return {
                        'content': full_content,
                        'summary': summary,
                        'citations': f"Database: {self.database} (p_party + p_encounter + p_diagnosis)",
                        'tool_info': (
                            f"<div class='tool-route'><strong>🏥 Patient History Tool</strong><br>"
                            f"Patient: {patient_name_label} | "
                            f"Encounters: {len(seen_encounters)} | "
                            f"Diagnoses: {len(rows)}</div>"
                        ),
                    }

        except Exception as exc:
            logger.error("fetch_patient_history error: %s", exc)
            return {
                'content': f"Error retrieving patient history: {exc}",
                'summary': "Patient history query failed due to a database error.",
                'citations': "Database error",
                'tool_info': "<div class='tool-route'><strong>❌ Patient History Tool</strong><br>Status: Error</div>",
            }


        """
        Generate an LLM summary of diagnosis data consistent with other tools
        
        Args:
            content: Raw diagnosis content
            search_term: Optional search term used
            result_count: Number of results found
            diagnosis_codes: List of diagnosis codes found
            
        Returns:
            Professional summary string
        """
        try:
            # Try to import and use the enhanced tools LLM summary function
            from enhanced_tools import get_llm_instance
            
            llm = get_llm_instance()
            if not llm:
                # Fallback to basic summary if LLM not available
                return self._generate_basic_summary(search_term, result_count, diagnosis_codes)
            
            # Create a concise query description for the LLM
            if search_term:
                query_desc = f"diagnosis information for '{search_term}'"
            else:
                query_desc = "available diagnoses in the medical database"
            
            # Prepare content for LLM (limit to avoid token issues)
            content_preview = content[:1500] + "..." if len(content) > 1500 else content
            
            system_message = "You are a medical AI assistant providing concise, professional summaries of medical database information."
            
            user_prompt = f"""Please provide a concise, professional summary of the following diagnosis database query results for: {query_desc}

Focus on:
- Number of diagnoses found
- Range of diagnosis codes if available
- Brief mention of the types of medical conditions represented
- Professional, clinical language appropriate for medical context

Diagnosis data:
{content_preview}

Provide a 2-3 sentence summary that is informative and professional."""
            
            # Use LangChain message format
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=user_prompt)
            ])
            
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}. Using basic summary.")
            return self._generate_basic_summary(search_term, result_count, diagnosis_codes)
    
    def _generate_basic_summary(self, search_term: str = None, result_count: int = 0, diagnosis_codes: List[str] = None) -> str:
        """Generate a basic summary when LLM is not available"""
        summary_parts = [f"Retrieved {result_count} diagnosis records from the medical database"]
        
        if search_term:
            summary_parts.append(f"matching the search criteria '{search_term}'")
        
        if diagnosis_codes and len(diagnosis_codes) > 0:
            code_range = f"ranging from {min(diagnosis_codes)} to {max(diagnosis_codes)}" if len(diagnosis_codes) > 1 else f"with code {diagnosis_codes[0]}"
            summary_parts.append(f"with diagnosis codes {code_range}")
        
        summary_parts.append("These records include comprehensive medical diagnosis information with associated clinical details.")
        
        return ". ".join(summary_parts) + "."
    
    def _format_tool_routing_html(self, primary_tool: str, confidence: str, tools_used: List[str], reasoning: str = "", result_count: int = 0) -> str:
        """Format tool routing information as HTML content consistent with enhanced_tools.py"""
        
        # Use consistent styling with other tools
        confidence_color = '#000'
        
        parts = []
        parts.append(f'<span style="color: #495057;">Confidence:</span> <span style="color: {confidence_color}; font-weight: bold;">{confidence}</span><br>')
        parts.append(f'<span style="color: #495057;">Tools Used:</span> {", ".join(tools_used)}<br>')
        if reasoning:
            parts.append(f'<span style="color: #495057;">Reasoning:</span> {reasoning}<br>')
        if result_count > 0:
            parts.append(f'<span style="color: #495057;">Results Found:</span> {result_count} diagnosis records')
        
        return "".join(parts)


# Initialize the PostgreSQL tool instance
postgres_tool = PostgreSQLTool()


def enhanced_postgres_search(query: str, patient_context: str = None) -> Dict[str, str]:
    """
    Enhanced PostgreSQL search function that integrates with the existing tool system
    
    Args:
        query: Search query for diagnosis descriptions
        patient_context: Optional patient context (for future use)
        
    Returns:
        Dict containing content, summary, citations, and tool_info
    """
    logger.info(f"Enhanced PostgreSQL Search: Searching diagnosis database for '{query}'")
    
    try:
        # Smart query processing: detect if user wants to list all diagnoses vs search for specific terms
        query_lower = query.lower()
        
        # Keywords that indicate user wants to see all/list diagnoses
        list_keywords = [
            'what diagnoses are available',
            'show me diagnoses',
            'list diagnoses',
            'diagnoses available',
            'show diagnosis codes',
            'what diagnosis codes',
            'list diagnosis codes',
            'all diagnoses',
            'available diagnoses',
            'diagnosis codes from database',
            'what diagnoses',
            'show me diagnosis'
        ]
        
        # Keywords that indicate specific search
        search_keywords = [
            'search for',
            'find diagnosis',
            'look for',
            'diagnosis containing'
        ]
        
        # Check if this is a general "list all" request
        is_list_request = any(keyword in query_lower for keyword in list_keywords)
        is_search_request = any(keyword in query_lower for keyword in search_keywords)
        
        if is_list_request and not is_search_request:
            # User wants to see all diagnoses - don't pass search term
            logger.info("Detected general diagnosis listing request - fetching all diagnoses")
            result = postgres_tool.fetch_diagnosis_descriptions(search_term=None, limit=30)
        else:
            # Extract meaningful search terms from the query
            # Remove common question words and database-related words
            words_to_remove = [
                'what', 'show', 'me', 'from', 'the', 'database', 'table', 'available',
                'are', 'in', 'diagnoses', 'diagnosis', 'codes', 'code', 'search',
                'find', 'look', 'for', 'containing', 'with', 'that', 'have'
            ]
            
            # Extract meaningful terms
            words = query_lower.split()
            search_terms = [word for word in words if word not in words_to_remove and len(word) > 2]
            
            if search_terms:
                # Use the first meaningful term for search
                search_term = search_terms[0]
                logger.info(f"Extracted search term: '{search_term}' from query: '{query}'")
                result = postgres_tool.search_diagnosis_by_keyword(search_term)
            else:
                # Fallback to showing all diagnoses if no meaningful terms found
                logger.info("No meaningful search terms found - showing all diagnoses")
                result = postgres_tool.fetch_diagnosis_descriptions(search_term=None, limit=30)
        
        # Add patient context consideration if provided
        if patient_context and result['content']:
            result['summary'] += f" This information may be relevant to the patient context: {patient_context[:100]}..."
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced PostgreSQL search error: {str(e)}")
        return {
            'content': f"Error accessing diagnosis database: {str(e)}",
            'summary': "Unable to retrieve diagnosis information from the medical database due to technical error.",
            'citations': "Database connection error",
            'tool_info': "<div class='tool-route'><strong>❌ PostgreSQL Database Tool</strong><br>Status: Connection Error</div>"
        }


def get_all_diagnosis_descriptions(limit: int = 50) -> Dict[str, str]:
    """
    Get all diagnosis descriptions from the database
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        Dict containing all diagnosis descriptions
    """
    return postgres_tool.fetch_diagnosis_descriptions(limit=limit)


def get_diagnosis_by_code(code: str) -> Dict[str, str]:
    """
    Get diagnosis by specific code
    
    Args:
        code: Diagnosis code to search for
        
    Returns:
        Dict containing diagnosis information
    """
    return postgres_tool.get_diagnosis_by_code(code)


def get_patient_history(
    patient_id: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    limit: int = 20,
) -> Dict[str, str]:
    """Fetch patient medical history (encounters + diagnoses) from pces_ehr_ccm.

    Args:
        patient_id:  p_party.party_id  (e.g. '101')
        first_name:  Patient first name (used when party_id unknown)
        last_name:   Patient last name
        limit:       Max encounter rows to return

    Returns:
        Dict with content, summary, citations, tool_info
    """
    return postgres_tool.fetch_patient_history(
        patient_id=patient_id,
        first_name=first_name,
        last_name=last_name,
        limit=limit,
    )
