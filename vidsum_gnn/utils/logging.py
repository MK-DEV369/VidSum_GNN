"""
Structured logging utilities for VidSum GNN pipeline.
Provides progress tracking, batch-level logging, and real-time updates.
"""
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import json


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PipelineStage(str, Enum):
    UPLOAD = "upload"
    SHOT_DETECTION = "shot_detection"
    FEATURE_EXTRACTION = "feature_extraction"
    GRAPH_CONSTRUCTION = "graph_construction"
    GNN_INFERENCE = "gnn_inference"
    SUMMARY_ASSEMBLY = "summary_assembly"
    COMPLETED = "completed"
    FAILED = "failed"


class StructuredLogger:
    """Structured logger with progress tracking and database integration."""
    
    def __init__(self, name: str, video_id: Optional[str] = None):
        self.name = name
        self.video_id = video_id
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        stage: Optional[PipelineStage] = None,
        progress: Optional[float] = None,
        batch_info: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """
        Log a structured message.
        
        Args:
            level: Log level
            message: Log message
            stage: Current pipeline stage
            progress: Progress percentage (0.0 - 1.0)
            batch_info: Batch processing information
            extra: Additional metadata
            exc_info: Whether to include exception information
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "video_id": self.video_id,
            "message": message,
            "stage": stage.value if stage else None,
            "progress": progress,
            "batch_info": batch_info or {},
            "extra": extra or {}
        }
        
        # Log to standard output
        log_func = getattr(self.logger, level.value.lower())
        formatted_msg = f"{message} | Stage: {stage} | Progress: {progress:.2%}" if progress else message
        log_func(formatted_msg, exc_info=exc_info)
        
        return log_data
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        return self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        return self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        return self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        return self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def batch_start(self, batch_num: int, total_batches: int, stage: PipelineStage):
        """Log batch processing start."""
        return self.info(
            f"Starting batch {batch_num}/{total_batches}",
            stage=stage,
            progress=batch_num / total_batches,
            batch_info={"current_batch": batch_num, "total_batches": total_batches}
        )
    
    def batch_complete(self, batch_num: int, total_batches: int, stage: PipelineStage, duration: float):
        """Log batch processing completion."""
        return self.info(
            f"Completed batch {batch_num}/{total_batches} in {duration:.2f}s",
            stage=stage,
            progress=batch_num / total_batches,
            batch_info={
                "current_batch": batch_num,
                "total_batches": total_batches,
                "duration": duration
            }
        )
    
    def stage_start(self, stage: PipelineStage):
        """Log pipeline stage start."""
        return self.info(f"Starting stage: {stage.value}", stage=stage, progress=0.0)
    
    def stage_complete(self, stage: PipelineStage, duration: float):
        """Log pipeline stage completion."""
        return self.info(
            f"Completed stage: {stage.value} in {duration:.2f}s",
            stage=stage,
            progress=1.0,
            extra={"duration": duration}
        )
    
    def memory_alert(self, gpu_used_mb: float, cpu_used_mb: float):
        """Log memory usage alert."""
        return self.warning(
            f"Memory usage: GPU={gpu_used_mb:.1f}MB, CPU={cpu_used_mb:.1f}MB",
            extra={"gpu_memory_mb": gpu_used_mb, "cpu_memory_mb": cpu_used_mb}
        )


def get_logger(name: str, video_id: Optional[str] = None) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        video_id: Optional video ID for context
    
    Returns:
        Structured logger instance
    """
    return StructuredLogger(name, video_id)


# Global logger instances
def setup_logging(log_level: str = "INFO"):
    """Setup global logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Progress tracking utilities
class ProgressTracker:
    """Track progress across pipeline stages."""
    
    STAGE_WEIGHTS = {
        PipelineStage.SHOT_DETECTION: 0.15,
        PipelineStage.FEATURE_EXTRACTION: 0.40,
        PipelineStage.GRAPH_CONSTRUCTION: 0.15,
        PipelineStage.GNN_INFERENCE: 0.20,
        PipelineStage.SUMMARY_ASSEMBLY: 0.10,
    }
    
    def __init__(self):
        self.stage_progress: Dict[PipelineStage, float] = {}
    
    def update_stage(self, stage: PipelineStage, progress: float):
        """Update progress for a stage (0.0 - 1.0)."""
        self.stage_progress[stage] = max(0.0, min(1.0, progress))
    
    def get_overall_progress(self) -> float:
        """Calculate overall pipeline progress."""
        total_progress = 0.0
        for stage, weight in self.STAGE_WEIGHTS.items():
            stage_prog = self.stage_progress.get(stage, 0.0)
            total_progress += stage_prog * weight
        return total_progress
    
    def get_current_stage(self) -> Optional[PipelineStage]:
        """Get the currently active stage."""
        for stage in PipelineStage:
            if stage in self.stage_progress and self.stage_progress[stage] < 1.0:
                return stage
        return None
