from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import datetime

Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        Index('idx_video_status', 'status'),
        Index('idx_video_uploaded_at', 'uploaded_at'),
        Index('idx_video_created_compound', 'uploaded_at', 'status'),
    )

    video_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    duration_seconds = Column(Float)
    fps = Column(Float)
    target_duration = Column(Integer, default=60)
    selection_method = Column(String, default="greedy")  # greedy, knapsack
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="queued", index=True)  # queued, processing, completed, failed

    shots = relationship("Shot", back_populates="video", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="video", cascade="all, delete-orphan")

class Shot(Base):
    __tablename__ = "shots"
    __table_args__ = (
        Index('idx_shot_video_id', 'video_id'),
        Index('idx_shot_time_range', 'video_id', 'start_sec', 'end_sec'),
        Index('idx_shot_importance', 'importance_score'),
        Index('idx_shot_created_at', 'created_at'),
    )

    shot_id = Column(String, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    duration_sec = Column(Float, nullable=False)
    keyframe_path = Column(String)
    importance_score = Column(Float, default=0.0, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    video = relationship("Video", back_populates="shots")
    embeddings = relationship("Embedding", back_populates="shot", cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        Index('idx_embedding_video_shot', 'video_id', 'shot_id'),
        Index('idx_embedding_modality', 'modality'),
        Index('idx_embedding_created_at', 'created_at'),
    )

    embedding_id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    shot_id = Column(String, ForeignKey("shots.shot_id"), nullable=False, index=True)
    modality = Column(String, nullable=False, index=True) # visual, audio, text
    vector_ref_path = Column(String) # Path to .pt or .npy file
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    shot = relationship("Shot", back_populates="embeddings")

class Summary(Base):
    __tablename__ = "summaries"
    __table_args__ = (
        Index('idx_summary_video_id', 'video_id'),
        Index('idx_summary_type', 'type'),
        Index('idx_summary_generated_at', 'generated_at'),
    )

    summary_id = Column(String, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)
    type = Column(String, index=True) # keyframes, clips
    duration = Column(Float)
    path = Column(String)
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    config_json = Column(JSON)

    video = relationship("Video", back_populates="summaries")

