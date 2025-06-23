"""
Document processing module for handling different file types.
"""

import os
import io
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import json
import xml.etree.ElementTree as ET
from pypdf import PdfReader
import cv2
import numpy as np

from ..models.base import BaseDocument, DataType, ProcessingStatus
from ..utils.logging import LoggerMixin
from ..utils.validation import data_validator
from ..utils.metrics import monitor_function


class DocumentProcessor(LoggerMixin):
    """Base document processor for handling different file types."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.txt': self._process_text,
            '.json': self._process_json,
            '.xml': self._process_xml,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
        }
    
    @monitor_function("document_processor", "process_document", "document")
    def process_document(self, file_path: Union[str, Path], 
                        metadata: Optional[Dict[str, Any]] = None) -> BaseDocument:
        """
        Process a document and return a BaseDocument object.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            BaseDocument object with processed content
        """
        try:
            file_path = Path(file_path)
            
            # Validate file
            is_valid, error_msg = data_validator.validate_file_type(file_path)
            if not is_valid:
                raise ValueError(f"File validation failed: {error_msg}")
            
            is_valid, error_msg = data_validator.validate_file_size(file_path)
            if not is_valid:
                raise ValueError(f"File size validation failed: {error_msg}")
            
            # Determine file type
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Process file
            content = self.supported_extensions[extension](file_path)
            
            # Create document object
            doc = BaseDocument(
                type=DataType.DOCUMENT,
                content=content,
                metadata=metadata or {}
            )
            
            # Add file metadata
            doc.update_metadata("file_path", str(file_path))
            doc.update_metadata("file_size", file_path.stat().st_size)
            doc.update_metadata("file_extension", extension)
            doc.update_metadata("file_hash", data_validator.calculate_file_hash(file_path))
            
            doc.status = ProcessingStatus.COMPLETED
            self.logger.info("Document processed successfully", document_id=doc.id, file_path=str(file_path))
            
            return doc
            
        except Exception as e:
            self.logger.error("Document processing failed", error=str(e), file_path=str(file_path))
            raise
    
    def _process_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception:
            # Fallback to PyPDF
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text.strip()
    
    def _process_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _process_text(self, file_path: Path) -> str:
        """Read text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def _process_json(self, file_path: Path) -> str:
        """Process JSON file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, indent=2)
    
    def _process_xml(self, file_path: Path) -> str:
        """Process XML file."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode')
    
    def _process_csv(self, file_path: Path) -> str:
        """Process CSV file."""
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    
    def _process_excel(self, file_path: Path) -> str:
        """Process Excel file."""
        df = pd.read_excel(file_path)
        return df.to_string(index=False)


class ImageProcessor(LoggerMixin):
    """Processor for image files with OCR and feature extraction."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    @monitor_function("image_processor", "process_image", "image")
    def process_image(self, file_path: Union[str, Path], 
                     metadata: Optional[Dict[str, Any]] = None) -> BaseDocument:
        """
        Process an image and extract text and features.
        
        Args:
            file_path: Path to the image file
            metadata: Additional metadata for the image
            
        Returns:
            BaseDocument object with processed content
        """
        try:
            file_path = Path(file_path)
            
            # Validate file
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported image format: {extension}")
            
            # Load image
            image = Image.open(file_path)
            
            # Extract text using OCR (if available)
            text_content = self._extract_text_from_image(image)
            
            # Extract image features
            features = self._extract_image_features(image)
            
            # Create document object
            doc = BaseDocument(
                type=DataType.IMAGE,
                content=text_content,
                metadata=metadata or {}
            )
            
            # Add image metadata
            doc.update_metadata("file_path", str(file_path))
            doc.update_metadata("image_size", image.size)
            doc.update_metadata("image_mode", image.mode)
            doc.update_metadata("image_format", image.format)
            doc.update_metadata("image_features", features)
            doc.update_metadata("file_hash", data_validator.calculate_file_hash(file_path))
            
            doc.status = ProcessingStatus.COMPLETED
            self.logger.info("Image processed successfully", document_id=doc.id, file_path=str(file_path))
            
            return doc
            
        except Exception as e:
            self.logger.error("Image processing failed", error=str(e), file_path=str(file_path))
            raise
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            self.logger.warning("pytesseract not available, skipping OCR")
            return ""
        except Exception as e:
            self.logger.warning("OCR failed", error=str(e))
            return ""
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image features."""
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Basic features
            features = {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height,
                "mode": image.mode,
                "format": image.format,
            }
            
            # Color features (if RGB)
            if image.mode == 'RGB':
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Color histogram
                hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features["color_histogram"] = hist.tolist()
                
                # Dominant colors
                pixels = img_cv.reshape(-1, 3)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42)
                kmeans.fit(pixels)
                features["dominant_colors"] = kmeans.cluster_centers_.tolist()
            
            return features
            
        except Exception as e:
            self.logger.warning("Feature extraction failed", error=str(e))
            return {"width": image.width, "height": image.height, "mode": image.mode}


class TabularProcessor(LoggerMixin):
    """Processor for tabular data (CSV, Excel, etc.)."""
    
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        return info
    
    def _convert_to_text(self, df: pd.DataFrame) -> str:
        """Convert dataframe to text representation."""
        # Limit the size for text representation
        if len(df) > 1000:
            df_sample = df.head(500).append(df.tail(500))
            text = f"DataFrame with {len(df)} rows and {len(df.columns)} columns.\n"
            text += "Showing first 500 and last 500 rows:\n\n"
            text += df_sample.to_string(index=False)
        else:
            text = df.to_string(index=False)
        
        return text 