-- TimescaleDB Initialization and Optimization Script
-- Run this script after creating the base tables to enable TimescaleDB features

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert videos table to hypertable for time-series optimization
-- (only needed if you want to partition by uploaded_at)
-- SELECT create_hypertable('videos', 'uploaded_at', if_not_exists => TRUE);

-- Convert shots table to hypertable for time-series optimization
SELECT create_hypertable('shots', 'created_at', if_not_exists => TRUE);

-- Convert embeddings table to hypertable
SELECT create_hypertable('embeddings', 'created_at', if_not_exists => TRUE);

-- Convert summaries table to hypertable
SELECT create_hypertable('summaries', 'generated_at', if_not_exists => TRUE);

-- Enable compression on older chunks (data older than 7 days)
ALTER TABLE shots SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'created_at DESC, shot_id'
);

ALTER TABLE embeddings SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'created_at DESC, embedding_id'
);

ALTER TABLE summaries SET (
  timescaledb.compress,
  timescaledb.compress_orderby = 'generated_at DESC, summary_id'
);

-- Add compression policy for automatic compression of older chunks
SELECT add_compression_policy('shots', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('embeddings', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('summaries', INTERVAL '7 days', if_not_exists => TRUE);

-- Create continuous aggregates for fast queries
CREATE MATERIALIZED VIEW IF NOT EXISTS videos_per_day AS
SELECT
  time_bucket('1 day', videos.uploaded_at) AS day,
  COUNT(DISTINCT videos.video_id) AS video_count,
  AVG(videos.duration_seconds) AS avg_duration
FROM videos
GROUP BY day;

-- Create index on continuous aggregate
CREATE INDEX IF NOT EXISTS idx_videos_per_day_day ON videos_per_day (day DESC);

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('videos_per_day',
  start_offset => INTERVAL '1 day',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour',
  if_not_exists => TRUE
);

-- Create stats view for monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS processing_stats AS
SELECT
  videos.video_id,
  videos.filename,
  videos.status,
  COUNT(DISTINCT shots.shot_id) AS shot_count,
  COUNT(DISTINCT embeddings.embedding_id) AS embedding_count,
  MAX(shots.importance_score) AS max_importance_score,
  AVG(shots.importance_score) AS avg_importance_score,
  videos.uploaded_at,
  MAX(shots.created_at) AS last_shot_created
FROM videos
LEFT JOIN shots ON videos.video_id = shots.video_id
LEFT JOIN embeddings ON videos.video_id = embeddings.video_id
GROUP BY videos.video_id, videos.filename, videos.status, videos.uploaded_at;

CREATE INDEX IF NOT EXISTS idx_processing_stats_video_id ON processing_stats (video_id);
CREATE INDEX IF NOT EXISTS idx_processing_stats_status ON processing_stats (status);

-- Add retention policy to automatically drop old data (optional, adjust as needed)
-- SELECT add_retention_policy('shots', INTERVAL '90 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('embeddings', INTERVAL '90 days', if_not_exists => TRUE);
-- SELECT add_retention_policy('summaries', INTERVAL '90 days', if_not_exists => TRUE);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_shots_video_importance ON shots (video_id, importance_score DESC)
  WHERE importance_score > 0;

CREATE INDEX IF NOT EXISTS idx_embeddings_video_modality ON embeddings (video_id, modality);

-- Analyze the database to gather statistics
ANALYZE videos;
ANALYZE shots;
ANALYZE embeddings;
ANALYZE summaries;

-- Log successful initialization
SELECT 'TimescaleDB initialization complete' AS status;
