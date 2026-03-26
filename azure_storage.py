"""
Azure Blob Storage utilities for PCES application
Handles encrypted PDF uploads to specific containers
"""

import os
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import AzureError
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=sauslocumservices;AccountKey=Sa4Do30I009CwG89FfTrmQ1lI/2/A4qgwBMQwM778Pi1099b+LQytd3YaXV9VytpVr8M3Bv6HP+M+AStLm/tTQ==;EndpointSuffix=core.windows.net"


class AzureStorageManager:
    def __init__(self):
        """Initialize Azure Storage client"""
        try:
            # Use the hardcoded connection string directly (same as Azure_Blob.py)
            self.connection_string = AZURE_STORAGE_CONNECTION_STRING
            
            if not self.connection_string:
                raise ValueError("Connection string is not defined")
            
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            logger.info("Azure Storage client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage client: {e}")
            raise
            raise

    def upload_research_pdf(self, pdf_content, filename, patient_problem=None, metadata=None):
        """
        Upload research PDF to contoso container under pces/documents/research/
        
        Args:
            pdf_content: PDF file content (bytes)
            filename: Name of the PDF file
            patient_problem: Patient problem text for metadata
            metadata: Additional metadata dictionary
        
        Returns:
            str: Blob URL if successful, None if failed
        """
        try:
            container_name = "contoso"
            blob_path = f"pces/documents/research/{filename}"
            
            # Prepare metadata
            blob_metadata = {
                'upload_date': datetime.now().isoformat(),
                'file_type': 'research_pdf',
                'patient_problem': patient_problem or 'Not specified',
                'content_length': str(len(pdf_content))
            }
            
            # Add custom metadata if provided
            if metadata:
                # Convert all metadata values to strings for Azure compatibility
                for key, value in metadata.items():
                    blob_metadata[key] = str(value) if value is not None else ''
            
            # Create container if it doesn't exist
            container_client = self.blob_service_client.get_container_client(container_name)
            try:
                container_client.create_container()
                logger.info(f"Created container: {container_name}")
            except AzureError:
                # Container already exists
                pass
            
            # Upload with encryption (server-side encryption is enabled by default)
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_path
            )
            
            # Set content settings
            content_settings = ContentSettings(content_type='application/pdf')
            
            # Upload the blob
            blob_client.upload_blob(
                data=pdf_content,
                overwrite=True,
                metadata=blob_metadata,
                content_settings=content_settings
            )
            
            blob_url = blob_client.url
            logger.info(f"Research PDF uploaded successfully: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to upload research PDF: {e}")
            return None

    def upload_patient_summary_pdf(self, pdf_content, filename, patient_data=None, metadata=None):
        """
        Upload patient summary PDF to contoso container under pces/documents/doc-patient-summary/
        
        Args:
            pdf_content: PDF file content (bytes)
            filename: Name of the PDF file
            patient_data: Patient information dictionary
            metadata: Additional metadata dictionary
        
        Returns:
            str: Blob URL if successful, None if failed
        """
        try:
            container_name = "contoso"
            blob_path = f"pces/documents/doc-patient-summary/{filename}"
            
            # Prepare metadata
            blob_metadata = {
                'upload_date': datetime.now().isoformat(),
                'file_type': 'patient_summary_pdf',
                'content_length': str(len(pdf_content))
            }
            
            # Add patient data to metadata if provided
            if patient_data:
                blob_metadata.update({
                    'patient_name': str(patient_data.get('patient_name', 'Unknown')),
                    'patient_id': str(patient_data.get('patient_id', 'Unknown')),
                    'doctor_name': str(patient_data.get('doctor_name', 'Unknown')),
                    'session_date': str(patient_data.get('session_date', ''))
                })
            
            # Add custom metadata if provided
            if metadata:
                # Convert all metadata values to strings for Azure compatibility
                for key, value in metadata.items():
                    blob_metadata[key] = str(value) if value is not None else ''
            
            # Create container if it doesn't exist
            container_client = self.blob_service_client.get_container_client(container_name)
            try:
                container_client.create_container()
                logger.info(f"Created container: {container_name}")
            except AzureError:
                # Container already exists
                pass
            
            # Upload with encryption
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_path
            )
            
            # Set content settings
            content_settings = ContentSettings(content_type='application/pdf')
            
            # Upload the blob
            blob_client.upload_blob(
                data=pdf_content,
                overwrite=True,
                metadata=blob_metadata,
                content_settings=content_settings
            )
            
            blob_url = blob_client.url
            logger.info(f"Patient summary PDF uploaded successfully: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to upload patient summary PDF: {e}")
            return None

    def upload_conversation_pdf(self, pdf_content, filename, conversation_data=None, metadata=None):
        """
        Upload doctor-patient conversation PDF to contoso container under pces/documents/conversation/
        
        Args:
            pdf_content: PDF file content (bytes)
            filename: Name of the PDF file
            conversation_data: Conversation information dictionary
            metadata: Additional metadata dictionary
        
        Returns:
            str: Blob URL if successful, None if failed
        """
        try:
            container_name = "contoso"
            blob_path = f"pces/documents/conversation/{filename}"
            
            # Prepare metadata
            blob_metadata = {
                'upload_date': datetime.now().isoformat(),
                'file_type': 'conversation_pdf',
                'content_length': str(len(pdf_content))
            }
            
            # Add conversation data to metadata if provided
            if conversation_data:
                blob_metadata.update({
                    'doctor_name': str(conversation_data.get('doctor_name', 'Unknown')),
                    'patient_name': str(conversation_data.get('patient_name', 'Unknown')),
                    'conversation_duration': str(conversation_data.get('duration', 'Unknown')),
                    'session_date': str(conversation_data.get('session_date', ''))
                })
            
            # Add custom metadata if provided
            if metadata:
                # Convert all metadata values to strings for Azure compatibility
                for key, value in metadata.items():
                    blob_metadata[key] = str(value) if value is not None else ''
            
            # Create container if it doesn't exist
            container_client = self.blob_service_client.get_container_client(container_name)
            try:
                container_client.create_container()
                logger.info(f"Created container: {container_name}")
            except AzureError:
                # Container already exists
                pass
            
            # Upload with encryption
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_path
            )
            
            # Set content settings
            content_settings = ContentSettings(content_type='application/pdf')
            
            # Upload the blob
            blob_client.upload_blob(
                data=pdf_content,
                overwrite=True,
                metadata=blob_metadata,
                content_settings=content_settings
            )
            
            blob_url = blob_client.url
            logger.info(f"Conversation PDF uploaded successfully: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to upload conversation PDF: {e}")
            return None

    def save_metadata_json(self, data, filename, folder_type="research"):
        """
        Save metadata as JSON file to Azure storage
        
        Args:
            data: Dictionary containing metadata
            filename: Name of the JSON file
            folder_type: Type of folder (research, patient-summary, conversation)
        
        Returns:
            str: Blob URL if successful, None if failed
        """
        try:
            container_name = "contoso"
            
            # Determine folder path based on type
            folder_paths = {
                "research": "pces/documents/research/metadata",
                "patient-summary": "pces/documents/doc-patient-summary/metadata",
                "conversation": "pces/documents/conversation/metadata"
            }
            
            folder_path = folder_paths.get(folder_type, "pces/documents/metadata")
            blob_path = f"{folder_path}/{filename}"
            
            # Convert data to JSON
            json_content = json.dumps(data, indent=2, default=str).encode('utf-8')
            
            # Upload JSON file
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_path
            )
            
            # Set content settings for JSON
            content_settings = ContentSettings(content_type='application/json')
            
            # Upload the blob
            blob_client.upload_blob(
                data=json_content,
                overwrite=True,
                content_settings=content_settings
            )
            
            blob_url = blob_client.url
            logger.info(f"Metadata JSON uploaded successfully: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to upload metadata JSON: {e}")
            return None

    def check_file_exists(self, container_name, blob_path):
        """
        Check if a file exists in Azure Blob Storage
        
        Args:
            container_name: Name of the container
            blob_path: Path to the blob (including folders)
        
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_path
            )
            
            # Check if blob exists
            exists = blob_client.exists()
            logger.info(f"File {blob_path} exists: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

    def list_files_in_container(self, container_name, prefix=None):
        """
        List all files in a container or with a specific prefix
        
        Args:
            container_name: Name of the container
            prefix: Optional prefix to filter files (e.g., "pces/documents/research/")
        
        Returns:
            list: List of blob information dictionaries
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blob_info = {
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified,
                    'url': f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}",
                    'metadata': blob.metadata if hasattr(blob, 'metadata') else {}
                }
                blobs.append(blob_info)
            
            logger.info(f"Found {len(blobs)} files in container '{container_name}' with prefix '{prefix}'")
            return blobs
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def get_file_metadata(self, container_name, blob_path):
        """
        Get metadata for a specific file
        
        Args:
            container_name: Name of the container
            blob_path: Path to the blob
        
        Returns:
            dict: File metadata and properties
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_path
            )
            
            properties = blob_client.get_blob_properties()
            
            file_info = {
                'name': blob_path,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'content_type': properties.content_settings.content_type,
                'url': blob_client.url,
                'metadata': properties.metadata,
                'etag': properties.etag
            }
            
            logger.info(f"Retrieved metadata for {blob_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return None

# Initialize global storage manager
storage_manager = None

def get_storage_manager():
    """Get or create storage manager instance"""
    global storage_manager
    if storage_manager is None:
        storage_manager = AzureStorageManager()
    return storage_manager
