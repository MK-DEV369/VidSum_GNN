"""
Database Optimization Script
Creates indexes and optimized queries for VideoSum-GNN application
"""
from sqlalchemy import text
from vidsum_gnn.db.client import AsyncSessionLocal


OPTIMIZATION_SQL = """
-- Create composite index for text summary lookup
CREATE INDEX IF NOT EXISTS idx_summary_video_type 
ON summaries(video_id, type);

-- Create index for shot importance score queries (top-K selection)
CREATE INDEX IF NOT EXISTS idx_shot_importance 
ON shots(video_id, importance_score DESC);

-- Create index for video status filtering
CREATE INDEX IF NOT EXISTS idx_video_status 
ON videos(status);

-- Create index for timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_video_created 
ON videos(created_at DESC);

-- Analyze tables for query planner optimization
ANALYZE videos;
ANALYZE shots;
ANALYZE summaries;
ANALYZE embeddings;
"""


async def apply_optimizations():
    """Apply database optimizations"""
    async with AsyncSessionLocal() as db:
        try:
            print("Applying database optimizations...")
            
            # Split and execute each statement
            statements = [s.strip() for s in OPTIMIZATION_SQL.split(';') if s.strip()]
            
            for statement in statements:
                print(f"Executing: {statement[:50]}...")
                await db.execute(text(statement))
                await db.commit()
            
            print("✓ Database optimizations applied successfully!")
            
        except Exception as e:
            print(f"✗ Error applying optimizations: {e}")
            await db.rollback()
            raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(apply_optimizations())
