"""
Gemini API Fallback Module for Video Summarization
Provides fallback summarization when GNN processing fails
"""
import os
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiVideoSummarizer:
    """Fallback video summarization using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google Generative AI API key (defaults to GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment or provided as argument")
            self.available = False
        else:
            self.available = True
            try:
                from google import genai
                self.genai = genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("✓ GAPI")
            except ImportError:
                logger.error("google-genai package not installed. Install with: pip install google-genai")
                self.available = False
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.available = False
    
    def summarize_video_from_path(
        self, 
        video_path: str, 
        summary_type: str = "balanced",
        text_length: str = "medium",
        summary_format: str = "bullet"
    ) -> Tuple[Optional[str], Dict]:
        """
        Summarize a video using Gemini API
        
        Args:
            video_path: Path to video file (must be ≤20MB for inline upload)
            summary_type: Type of summary ('balanced', 'visual', 'audio', 'highlight')
            text_length: Summary length ('short', 'medium', 'long')
            summary_format: Output format ('bullet', 'structured', 'plain')
        
        Returns:
            Tuple of (summary_text, metadata_dict)
            Returns (None, {}) if summarization fails
        """
        if not self.available:
            logger.error("GAPI not available")
            return None, {"error": "GAPI not initialized"}
        
        try:
            # Check file size
            file_size = Path(video_path).stat().st_size
            if file_size > 20 * 1024 * 1024:  # 20MB limit for inline upload
                logger.warning(f"Video file {file_size / 1024 / 1024:.1f}MB exceeds 20MB limit for GAPI inline upload")
                return None, {"error": f"Video too large ({file_size / 1024 / 1024:.1f}MB > 20MB)"}
            
            logger.info(f"[FALLBACK] Summarizing video via Gemini API: {Path(video_path).name} ({file_size / 1024 / 1024:.1f}MB)")
            
            # Read video file
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            # Determine prompt based on summary type (support legacy + current vocab)
            prompt_map = {
                "visual": "Focus on visual elements, scene changes, and visual effects. Provide a summary emphasizing what viewers see.",
                "visual_priority": "Focus on visual elements, scene changes, and visual effects. Provide a summary emphasizing what viewers see.",
                "audio": "Focus on dialogue, narration, and sound. Provide a summary emphasizing what viewers hear.",
                "audio_priority": "Focus on dialogue, narration, and sound. Provide a summary emphasizing what viewers hear.",
                "highlight": "Identify and summarize the most important/exciting moments in the video.",
                "highlights": "Identify and summarize the most important/exciting moments in the video.",
                "balanced": "Provide a comprehensive summary of the video content, balancing visual and audio elements."
            }
            prompt_base = prompt_map.get(summary_type, prompt_map["balanced"])
            
            # Add length guidance
            length_map = {
                "short": "Keep it very concise (2-3 sentences).",
                "medium": "Keep it moderate length (4-6 sentences).",
                "long": "Provide a detailed summary (8-12 sentences)."
            }
            length_hint = length_map.get(text_length, length_map["medium"])
            
            # Format specification
            format_map = {
                "bullet": "Format as bullet points (each point on a new line starting with '•')",
                "structured": "Format as a structured summary with clear sections",
                "plain": "Format as plain text paragraphs"
            }
            format_hint = format_map.get(summary_format, format_map["bullet"])
            
            # Construct full prompt
            full_prompt = f"""Summarize this video. {prompt_base} {length_hint} {format_hint}"""
            
            logger.debug(f"[FALLBACK] Gemini Prompt: {full_prompt}")
            
            # Call Gemini API with video
            from google.genai import types
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=types.Content(
                    parts=[
                        types.Part(text=full_prompt),
                        types.Part(inline_data=types.Blob(
                            data=video_bytes, 
                            mime_type="video/mp4"
                        )),
                    ]
                ),
            )
            
            summary_text = response.text
            logger.info("[FALLBACK] ✓ Gemini API summarization successful")
            
            return summary_text, {
                "source": "gemini",
                "model": "gemini-2.0-flash",
                "summary_type": summary_type,
                "text_length": text_length,
                "summary_format": summary_format,
                "video_size_mb": file_size / 1024 / 1024
            }
        
        except FileNotFoundError:
            logger.error(f"Video file not found: {video_path}")
            return None, {"error": f"Video file not found: {video_path}"}
        
        except ImportError:
            logger.error("google-genai package not installed")
            return None, {"error": "google-genai package not installed"}
        
        except Exception as e:
            logger.error(f"[FALLBACK] Gemini API error: {e}", exc_info=True)
            return None, {"error": f"Gemini API failed: {str(e)}"}
    
    def is_available(self) -> bool:
        """Check if Gemini API is available and configured"""
        return self.available


def get_gemini_summarizer() -> GeminiVideoSummarizer:
    """Get or create a Gemini summarizer instance (singleton-ish)"""
    if not hasattr(get_gemini_summarizer, "_instance"):
        get_gemini_summarizer._instance = GeminiVideoSummarizer()
    return get_gemini_summarizer._instance
