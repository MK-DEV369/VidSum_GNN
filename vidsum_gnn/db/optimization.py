"""
Database initialization and optimization utilities for TimescaleDB.
Provides functions to set up hypertables, compression, and aggregates.
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from vidsum_gnn.utils import get_logger

logger = get_logger(__name__)

async def init_timescaledb(session: AsyncSession) -> bool:
    """
    Initialize TimescaleDB extensions and optimizations.
    
    This should be called once during application startup.
    
    Args:
        session: SQLAlchemy async session
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing TimescaleDB extensions...")
        
        # Enable TimescaleDB extension
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        await session.commit()
        logger.info("TimescaleDB extension enabled")
        
        # Create hypertables
        hypertables = [
            ("shots", "created_at"),
            ("embeddings", "created_at"),
            ("summaries", "generated_at"),
        ]
        
        for table_name, time_column in hypertables:
            try:
                await session.execute(
                    text(f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE)")
                )
                logger.info(f"Hypertable created for {table_name} on {time_column}")
            except Exception as e:
                logger.warning(f"Could not create hypertable for {table_name}: {e}")
        
        await session.commit()
        
        # Enable compression
        compression_tables = [
            ("shots", "created_at DESC, shot_id"),
            ("embeddings", "created_at DESC, embedding_id"),
            ("summaries", "generated_at DESC, summary_id"),
        ]
        
        for table_name, order_by in compression_tables:
            try:
                await session.execute(
                    text(f"""
                        ALTER TABLE {table_name} SET (
                            timescaledb.compress,
                            timescaledb.compress_orderby = '{order_by}'
                        )
                    """)
                )
                logger.info(f"Compression enabled for {table_name}")
            except Exception as e:
                logger.warning(f"Could not enable compression for {table_name}: {e}")
        
        await session.commit()
        
        # Add compression policies
        for table_name in ["shots", "embeddings", "summaries"]:
            try:
                await session.execute(
                    text(f"""
                        SELECT add_compression_policy('{table_name}', INTERVAL '7 days', if_not_exists => TRUE)
                    """)
                )
                logger.info(f"Compression policy added for {table_name}")
            except Exception as e:
                logger.warning(f"Could not add compression policy for {table_name}: {e}")
        
        await session.commit()
        
        # Analyze tables for query optimization
        for table_name in ["videos", "shots", "embeddings", "summaries"]:
            try:
                await session.execute(text(f"ANALYZE {table_name}"))
                logger.info(f"Analyzed table: {table_name}")
            except Exception as e:
                logger.warning(f"Could not analyze {table_name}: {e}")
        
        await session.commit()
        
        logger.info("TimescaleDB initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing TimescaleDB: {e}")
        return False


async def get_database_stats(session: AsyncSession) -> dict:
    """
    Get comprehensive database statistics.
    
    Args:
        session: SQLAlchemy async session
        
    Returns:
        Dictionary with database statistics
    """
    try:
        stats = {}
        
        # Count records
        for table in ["videos", "shots", "embeddings", "summaries"]:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            stats[f"{table}_count"] = count or 0
        
        # Get database size
        result = await session.execute(
            text("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
        )
        row = result.first()
        stats["database_size"] = row[0] if row else "Unknown"
        
        # Get hypertable info
        result = await session.execute(
            text("""
                SELECT table_name, num_chunks, total_size
                FROM timescaledb_information.hypertable
            """)
        )
        rows = result.fetchall()
        stats["hypertables"] = [
            {"name": row[0], "chunks": row[1], "size": row[2]}
            for row in rows
        ]
        
        await session.commit()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}


async def optimize_queries(session: AsyncSession) -> bool:
    """
    Run query optimization and statistics gathering.
    
    Args:
        session: SQLAlchemy async session
        
    Returns:
        True if optimization successful
    """
    try:
        logger.info("Running database optimization...")
        
        # Vacuum and analyze
        for table_name in ["videos", "shots", "embeddings", "summaries"]:
            await session.execute(text(f"VACUUM ANALYZE {table_name}"))
            logger.info(f"Vacuumed and analyzed {table_name}")
        
        await session.commit()
        
        logger.info("Database optimization completed")
        return True
        
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        return False


async def get_processing_stats(session: AsyncSession, video_id: str) -> dict:
    """
    Get processing statistics for a specific video.
    
    Args:
        session: SQLAlchemy async session
        video_id: Video ID to get stats for
        
    Returns:
        Dictionary with processing statistics
    """
    try:
        result = await session.execute(
            text("""
                SELECT
                    videos.video_id,
                    videos.filename,
                    videos.status,
                    videos.uploaded_at,
                    COUNT(DISTINCT shots.shot_id) AS shot_count,
                    COUNT(DISTINCT embeddings.embedding_id) AS embedding_count,
                    MAX(shots.importance_score) AS max_importance_score,
                    AVG(shots.importance_score) AS avg_importance_score,
                    MAX(shots.created_at) AS last_shot_created
                FROM videos
                LEFT JOIN shots ON videos.video_id = shots.video_id
                LEFT JOIN embeddings ON videos.video_id = embeddings.video_id
                WHERE videos.video_id = :video_id
                GROUP BY videos.video_id, videos.filename, videos.status, videos.uploaded_at
            """),
            {"video_id": video_id}
        )
        
        row = result.first()
        if row:
            return {
                "video_id": row[0],
                "filename": row[1],
                "status": row[2],
                "uploaded_at": row[3],
                "shot_count": row[4] or 0,
                "embedding_count": row[5] or 0,
                "max_importance_score": row[6] or 0.0,
                "avg_importance_score": row[7] or 0.0,
                "last_shot_created": row[8]
            }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting processing stats for {video_id}: {e}")
        return {}
