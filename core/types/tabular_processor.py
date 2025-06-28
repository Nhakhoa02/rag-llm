"""
Tabular processing module for handling CSV, Excel, and other tabular data formats.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd

from core.models.base import BaseDocument, DataType, ProcessingStatus
from core.models.csv_index import CSVIndex, CSVIndexDocument
from core.utils.logging import LoggerMixin
from core.utils.validation import data_validator
from core.utils.metrics import monitor_function


class TabularProcessor(LoggerMixin):
    """Processor for tabular data (CSV, Excel, etc.) with CSV indexing support."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.csv', '.xlsx', '.xls', '.parquet', '.json'}
    
    @monitor_function("tabular_processor", "process_tabular", "tabular")
    def process_tabular(self, file_path: Union[str, Path], 
                       metadata: Optional[Dict[str, Any]] = None) -> BaseDocument:
        """
        Process tabular data and extract structured information.
        
        Args:
            file_path: Path to the tabular file
            metadata: Additional metadata for the data
            
        Returns:
            BaseDocument object with processed content
        """
        try:
            file_path = Path(file_path)
            
            # Validate file
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported tabular format: {extension}")
            
            # Load data
            df = self._load_tabular_data(file_path)
            
            # Extract structured information
            structure_info = self._extract_structure_info(df)
            
            # Convert to text representation
            text_content = self._convert_to_text(df)
            
            # Create document object
            doc = BaseDocument(
                type=DataType.TABULAR,
                content=text_content,
                metadata=metadata or {}
            )
            
            # Add tabular metadata
            doc.update_metadata("file_path", str(file_path))
            doc.update_metadata("row_count", len(df))
            doc.update_metadata("column_count", len(df.columns))
            doc.update_metadata("columns", df.columns.tolist())
            doc.update_metadata("data_types", df.dtypes.astype(str).to_dict())
            doc.update_metadata("structure_info", structure_info)
            doc.update_metadata("file_hash", data_validator.calculate_file_hash(file_path))
            
            # For CSV files, create and store CSV index
            if extension == '.csv':
                csv_index = self._create_csv_index(df, doc.id, file_path.name)
                doc.update_metadata("csv_index", csv_index.to_dict())
                self.logger.info(f"Created CSV index for {file_path.name}")
            
            doc.status = ProcessingStatus.COMPLETED
            self.logger.info("Tabular data processed successfully", document_id=doc.id, file_path=str(file_path))
            
            return doc
            
        except Exception as e:
            self.logger.error("Tabular processing failed", error=str(e), file_path=str(file_path))
            raise
    
    def _load_tabular_data(self, file_path: Path) -> pd.DataFrame:
        """Load tabular data from file."""
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif extension == '.parquet':
            return pd.read_parquet(file_path)
        elif extension == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported tabular format: {extension}")
    
    def _extract_structure_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract structural information from dataframe."""
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns},
        }
        
        # Add statistical information for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        return info
    
    def _create_csv_index(self, df: pd.DataFrame, csv_file_id: str, filename: str) -> CSVIndex:
        """Create a CSV index from the first and second rows of the dataframe."""
        try:
            # Extract first row (column headers)
            column_headers = df.columns.tolist()
            
            # Extract second row (sample data) if available
            sample_data = None
            if len(df) > 0:
                second_row = df.iloc[0].tolist()  # First data row (index 0)
                sample_data = [str(val) if pd.notna(val) else "" for val in second_row]
            
            # Infer data types from sample
            inferred_types = {}
            if sample_data:
                for header, value in zip(column_headers, sample_data):
                    if value:
                        try:
                            # Try to infer type
                            if value.replace('.', '').replace('-', '').isdigit():
                                if '.' in value:
                                    inferred_types[header] = 'float'
                                else:
                                    inferred_types[header] = 'int'
                            else:
                                inferred_types[header] = 'string'
                        except:
                            inferred_types[header] = 'string'
            
            # Create CSV index
            csv_index = CSVIndex(
                csv_file_id=csv_file_id,
                csv_filename=filename,
                column_headers=column_headers,
                sample_data=sample_data,
                total_rows=len(df),
                total_columns=len(df.columns),
                inferred_types=inferred_types
            )
            
            # Generate index content
            csv_index.update_index_content()
            
            return csv_index
            
        except Exception as e:
            self.logger.error(f"Failed to create CSV index: {e}")
            # Return a minimal index
            return CSVIndex(
                csv_file_id=csv_file_id,
                csv_filename=filename,
                column_headers=df.columns.tolist(),
                total_rows=len(df),
                total_columns=len(df.columns)
            )
    
    def create_csv_index_document(self, csv_index: CSVIndex) -> CSVIndexDocument:
        """Create a CSV index document for vector storage."""
        if csv_index is None:
            raise ValueError("CSV index cannot be None")
        return CSVIndexDocument(csv_index=csv_index)
    
    def _convert_to_text(self, df: pd.DataFrame) -> str:
        """Convert dataframe to text representation that's AI-friendly."""
        # Create a more structured, readable format for AI processing
        text_parts = []
        
        # Add dataset overview
        text_parts.append(f"Dataset Overview:")
        text_parts.append(f"- Total rows: {len(df)}")
        text_parts.append(f"- Total columns: {len(df.columns)}")
        text_parts.append(f"- Column names: {', '.join(df.columns.tolist())}")
        text_parts.append("")
        
        # Add data types information
        text_parts.append("Column Data Types:")
        for col, dtype in df.dtypes.items():
            text_parts.append(f"- {col}: {dtype}")
        text_parts.append("")
        
        # Add missing values summary
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            text_parts.append("Missing Values Summary:")
            for col, missing_count in missing_values.items():
                if missing_count > 0:
                    percentage = (missing_count / len(df)) * 100
                    text_parts.append(f"- {col}: {missing_count} missing values ({percentage:.1f}%)")
            text_parts.append("")
        
        # Add sample data in a more readable format
        text_parts.append("Sample Data:")
        
        # Limit the number of rows for readability
        if len(df) > 50:
            # Show first 25 and last 25 rows
            sample_df = pd.concat([df.head(25), df.tail(25)], ignore_index=True)
            text_parts.append(f"(Showing first 25 and last 25 rows out of {len(df)} total rows)")
        else:
            sample_df = df
        
        # Convert to a more readable format
        for i in range(len(sample_df)):
            row = sample_df.iloc[i]
            row_text = f"Row {i + 1}: "
            row_data = []
            for col in df.columns:
                value = row[col]
                # Handle different data types
                if pd.isna(value):
                    row_data.append(f"{col}=NULL")
                elif isinstance(value, (int, float)):
                    row_data.append(f"{col}={value}")
                else:
                    # Truncate long string values
                    str_value = str(value)
                    if len(str_value) > 50:
                        str_value = str_value[:47] + "..."
                    row_data.append(f"{col}='{str_value}'")
            row_data.append(f"{col}='{str_value}'")
            row_text += ", ".join(row_data)
            text_parts.append(row_text)
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text_parts.append("")
            text_parts.append("Numeric Column Statistics:")
            for col in numeric_cols:
                stats = df[col].describe()
                text_parts.append(f"- {col}:")
                text_parts.append(f"  * Count: {stats['count']:.0f}")
                text_parts.append(f"  * Mean: {stats['mean']:.2f}")
                text_parts.append(f"  * Min: {stats['min']:.2f}")
                text_parts.append(f"  * Max: {stats['max']:.2f}")
                text_parts.append(f"  * Std: {stats['std']:.2f}")
        
        return "\n".join(text_parts) 