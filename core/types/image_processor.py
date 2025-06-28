"""
Image processing module for handling image files with OCR and feature extraction.
"""

import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import google.generativeai as genai

from core.models.base import BaseDocument, DataType, ProcessingStatus
from core.utils.logging import LoggerMixin
from core.utils.validation import data_validator
from core.utils.metrics import monitor_function


class ImageProcessor(LoggerMixin):
    """Processor for image files with OCR and feature extraction."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    @monitor_function("image_processor", "process_image", "image")
    def process_image(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> BaseDocument:
        """
        Process an image and generate a caption using Gemini 2.5 Flash.
        """
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            image = Image.open(file_path).convert("RGB")
            # Use Gemini Vision for caption
            caption = self.get_gemini_image_caption(str(file_path))
            self.logger.info("Gemini Vision caption generated for image", caption=caption)
            features = self._extract_image_features(image)
            doc = BaseDocument(
                type=DataType.IMAGE,
                content=caption,
                metadata=metadata or {}
            )
            
            # Add image metadata
            doc.update_metadata("file_path", str(file_path))
            doc.update_metadata("file_size", file_path.stat().st_size)
            doc.update_metadata("file_extension", extension)
            doc.update_metadata("image_features", features)
            doc.update_metadata("file_hash", data_validator.calculate_file_hash(file_path))
            
            doc.status = ProcessingStatus.COMPLETED
            self.logger.info("Image processed successfully", document_id=doc.id, file_path=str(file_path))
            
            return doc
            
        except Exception as e:
            self.logger.error("Image processing failed", error=str(e), file_path=str(file_path))
            raise
    
    def get_gemini_image_caption(self, image_path: str) -> str:
        """Generate image caption using Gemini Vision."""
        try:
            # Configure Gemini
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Load and process image
            image = Image.open(image_path)
            
            # Generate caption
            response = model.generate_content([
                "Describe this image in detail, focusing on the main subjects, actions, and context. Be specific about what you see.",
                image
            ])
            
            # Extract text from response
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            
            # Fallback to parts if text doesn't work
            if hasattr(response, "parts") and response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                return " ".join(text_parts)
        
        except Exception as e:
            self.logger.warning(f"Gemini Vision caption generation failed: {e}")
            return f"Image file: {os.path.basename(image_path)}"
        
        # Fallback to candidates if parts don't work
        if hasattr(response, "candidates") and response.candidates:
            text_parts = []
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
            return " ".join(text_parts)
        
        return ""

    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        try:
            features = {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height,
                "mode": image.mode,
                "format": image.format,
            }
            return features
        except Exception as e:
            self.logger.warning("Feature extraction failed", error=str(e))
            return {"width": image.width, "height": image.height, "mode": image.mode} 