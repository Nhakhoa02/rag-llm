"""
CSV Database Manager for storing and querying CSV data.
"""

import sqlite3
import pandas as pd
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

from core.models.base import BaseDocument
from core.models.csv_index import CSVIndex
from core.utils.logging import get_logger, LoggerMixin
from core.utils.metrics import monitor_function
from config.config import settings

logger = get_logger(__name__)

class CSVDatabaseManager:
    """Manages SQLite databases for CSV data storage and querying."""
    
    def __init__(self, db_directory: str = "csv_databases"):
        self.db_directory = Path(db_directory)
        self.db_directory.mkdir(exist_ok=True)
        self.connections: Dict[str, sqlite3.Connection] = {}
        
    def store_csv_data(self, csv_index: CSVIndex, csv_file_path: str) -> str:
        """
        Store CSV data in SQLite database.
        
        Args:
            csv_index: CSV index with metadata
            csv_file_path: Path to the CSV file
            
        Returns:
            Database file path
        """
        try:
            # Create database file path using CSV filename
            # Remove file extension and replace invalid characters
            safe_filename = csv_index.csv_filename.replace('.csv', '').replace('.CSV', '')
            safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')
            
            db_filename = f"{safe_filename}_{csv_index.csv_file_id}.db"
            db_path = self.db_directory / db_filename
            
            # Read CSV data
            df = pd.read_csv(csv_file_path)
            
            # Create SQLite database
            conn = sqlite3.connect(str(db_path))
            
            # Store data in table named after the CSV file
            table_name = f"csv_data_{csv_index.csv_file_id.replace('-', '_')}"
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Store metadata
            metadata = {
                "csv_filename": csv_index.csv_filename,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_headers": json.dumps(df.columns.tolist()),  # Convert list to JSON string
                "table_name": table_name
            }
            
            # Create metadata table
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_sql("csv_metadata", conn, if_exists='replace', index=False)
            
            conn.close()
            
            logger.info(f"Stored CSV data in database: {db_path}")
            return str(db_path)
            
        except Exception as e:
            logger.error(f"Failed to store CSV data: {e}")
            raise
    
    def execute_query(self, csv_file_id: str, sql_query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute SQL query on CSV data.
        
        Args:
            csv_file_id: ID of the CSV file
            sql_query: SQL query to execute
            
        Returns:
            Tuple of (results, column_names)
        """
        try:
            # Find database file by searching for the csv_file_id in the filename
            db_path = None
            available_dbs = []
            
            for db_file in self.db_directory.glob("*.db"):
                available_dbs.append(db_file.name)
                if csv_file_id in db_file.name:
                    db_path = db_file
                    break
            
            if not db_path or not db_path.exists():
                error_msg = f"Database not found for CSV file ID: {csv_file_id}"
                error_msg += f"\nAvailable databases: {available_dbs}"
                error_msg += f"\nSearched directory: {self.db_directory}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Connect to database
            conn = sqlite3.connect(str(db_path))
            
            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get results
            results = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            
            # Convert to list of dictionaries
            result_dicts = []
            for row in results:
                result_dicts.append(dict(zip(column_names, row)))
            
            conn.close()
            
            logger.info(f"Executed query on {csv_file_id}: {len(result_dicts)} results")
            return result_dicts, column_names
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    def get_csv_metadata(self, csv_file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a CSV file."""
        try:
            # Find database file by searching for the csv_file_id in the filename
            db_path = None
            for db_file in self.db_directory.glob("*.db"):
                if csv_file_id in db_file.name:
                    db_path = db_file
                    break
            
            if not db_path or not db_path.exists():
                return None
            
            conn = sqlite3.connect(str(db_path))
            metadata_df = pd.read_sql("SELECT * FROM csv_metadata", conn)
            conn.close()
            
            if len(metadata_df) > 0:
                metadata = metadata_df.iloc[0].to_dict()
                # Parse column_headers back to list
                if 'column_headers' in metadata and isinstance(metadata['column_headers'], str):
                    try:
                        metadata['column_headers'] = json.loads(metadata['column_headers'])
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        metadata['column_headers'] = []
                return metadata
            return None
            
        except Exception as e:
            logger.error(f"Failed to get CSV metadata: {e}")
            return None
    
    def list_csv_databases(self) -> List[Dict[str, Any]]:
        """List all CSV databases."""
        databases = []
        
        for db_file in self.db_directory.glob("*.db"):
            try:
                # Extract CSV file ID from filename (format: filename_csvfileid.db)
                filename_parts = db_file.stem.split('_')
                if len(filename_parts) >= 2:
                    # The last part should be the CSV file ID
                    csv_file_id = filename_parts[-1]
                    metadata = self.get_csv_metadata(csv_file_id)
                    
                    if metadata:
                        databases.append({
                            "csv_file_id": csv_file_id,
                            "db_path": str(db_file),
                            "db_filename": db_file.name,
                            "metadata": metadata
                        })
                    else:
                        # Add database even if metadata is not available
                        databases.append({
                            "csv_file_id": csv_file_id,
                            "db_path": str(db_file),
                            "db_filename": db_file.name,
                            "metadata": None,
                            "error": "Metadata not available"
                        })
            except Exception as e:
                logger.warning(f"Error reading database {db_file}: {e}")
                databases.append({
                    "csv_file_id": "unknown",
                    "db_path": str(db_file),
                    "db_filename": db_file.name,
                    "metadata": None,
                    "error": str(e)
                })
        
        return databases
    
    def get_database_mapping(self) -> Dict[str, str]:
        """Get mapping of CSV file IDs to database filenames for debugging."""
        mapping = {}
        for db_file in self.db_directory.glob("*.db"):
            try:
                filename_parts = db_file.stem.split('_')
                if len(filename_parts) >= 2:
                    csv_file_id = filename_parts[-1]
                    mapping[csv_file_id] = db_file.name
            except Exception as e:
                logger.warning(f"Error parsing database filename {db_file}: {e}")
        return mapping
    
    def delete_csv_database(self, csv_file_id: str) -> bool:
        """Delete CSV database."""
        try:
            # Find database file by searching for the csv_file_id in the filename
            db_path = None
            for db_file in self.db_directory.glob("*.db"):
                if csv_file_id in db_file.name:
                    db_path = db_file
                    break
            
            if db_path and db_path.exists():
                db_path.unlink()
                logger.info(f"Deleted CSV database: {db_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete CSV database: {e}")
            return False

# Global instance
csv_db_manager = CSVDatabaseManager() 