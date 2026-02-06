# API Endpoints Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Upload Video
Upload a video file for summarization processing.

**Endpoint:** `POST /api/upload`

**Content-Type:** `multipart/form-data`

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | Video file (MP4, AVI, MOV, etc.) |
| target_duration | int | No | Target summary duration in seconds (default: 60) |
| selection_method | string | No | Shot selection strategy: "greedy" or "knapsack" (default: "greedy") |
| summary_type | string | No | Summary focus: "balanced", "visual", "audio", "highlight" (default: "balanced") |

**Response:**
```json
{
  "video_id": "abc123def456",
  "message": "Upload successful, processing started",
  "filename": "my_video.mp4"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@video.mp4" \
  -F "target_duration=45" \
  -F "selection_method=knapsack" \
  -F "summary_type=balanced"
```

---

### 2. Get Video Status
Check the processing status of an uploaded video.

**Endpoint:** `GET /api/status/{video_id}`

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| video_id | string | Unique video identifier |

**Response:**
```json
{
  "video_id": "abc123def456",
  "status": "processing",
  "target_duration": 45,
  "selection_method": "greedy"
}
```

**Status Values:**
- `preprocessing`: Transcoding and preparing video
- `shot_detection`: Detecting shot boundaries
- `feature_extraction`: Extracting visual and audio features
- `gnn_inference`: Running GNN model for importance scoring
- `assembling`: Selecting and concatenating shots
- `completed`: Processing finished successfully
- `failed`: Processing encountered an error

**Example:**
```bash
curl http://localhost:8000/api/status/abc123def456
```

---

### 3. Get Text Summary
Retrieve the generated text summary for a processed video.

**Endpoint:** `GET /api/summary/{video_id}/text`

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| video_id | string | Unique video identifier |

**Response:**
```json
{
  "video_id": "abc123def456",
  "bullet": "• Introduction to the topic\n• Main argument presented\n• Supporting evidence discussed\n• Conclusion and key takeaways",
  "structured": "Title: Video Summary\nDuration: 45 seconds\nType: balanced\n\nKey Points:\n1. Introduction to the topic\n2. Main argument presented\n3. Supporting evidence discussed\n4. Conclusion and key takeaways",
  "plain": "This video introduces the main topic, presents the key arguments with supporting evidence, and concludes with important takeaways for the audience.",
  "style": "balanced"
}
```

**Error Responses:**
- `404`: Text summary not found (video may not be processed yet)

**Example:**
```bash
curl http://localhost:8000/api/summary/abc123def456/text
```

---

### 4. Download Summary Video
Download the generated summary video file.

**Endpoint:** `GET /api/download/{video_id}`

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| video_id | string | Unique video identifier |

**Response:**
- Content-Type: `video/mp4`
- Disposition: `attachment; filename="summary_{video_id}.mp4"`

**Error Responses:**
- `404`: Summary video not found

**Example:**
```bash
# Download directly
curl -O -J http://localhost:8000/api/download/abc123def456

# Or use wget
wget http://localhost:8000/api/download/abc123def456
```

---

### 5. Get Summary Results
Get all summaries associated with a video.

**Endpoint:** `GET /api/results/{video_id}`

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| video_id | string | Unique video identifier |

**Response:**
```json
[
  {
    "summary_id": "sum_abc123",
    "video_id": "abc123def456",
    "type": "clips",
    "duration": 45.2,
    "video_path": "/app/data/outputs/summary_abc123.mp4",
    "text_summary_bullet": "• Point 1\n• Point 2",
    "text_summary_structured": "Title: Summary\n...",
    "text_summary_plain": "This video...",
    "summary_style": "balanced",
    "created_at": "2024-01-15T10:30:00Z"
  }
]
```

**Example:**
```bash
curl http://localhost:8000/api/results/abc123def456
```

---

### 6. Get Shot Scores
Retrieve importance scores for individual shots.

**Endpoint:** `GET /api/shot-scores/{video_id}`

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| video_id | string | Unique video identifier |

**Response:**
```json
[
  {
    "shot_id": "abc123_0001",
    "video_id": "abc123def456",
    "start_sec": 0.0,
    "end_sec": 2.5,
    "duration_sec": 2.5,
    "importance_score": 0.87
  },
  {
    "shot_id": "abc123_0002",
    "video_id": "abc123def456",
    "start_sec": 2.5,
    "end_sec": 5.8,
    "duration_sec": 3.3,
    "importance_score": 0.42
  }
]
```

**Example:**
```bash
curl http://localhost:8000/api/shot-scores/abc123def456
```

---

### 7. List All Videos
Get a list of all uploaded videos and their status.

**Endpoint:** `GET /api/videos`

**Response:**
```json
[
  {
    "video_id": "abc123def456",
    "filename": "my_video.mp4",
    "status": "completed",
    "target_duration": 45,
    "selection_method": "greedy",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:35:00Z"
  },
  {
    "video_id": "def789ghi012",
    "filename": "another_video.mp4",
    "status": "processing",
    "target_duration": 60,
    "selection_method": "knapsack",
    "created_at": "2024-01-15T11:00:00Z",
    "updated_at": "2024-01-15T11:05:00Z"
  }
]
```

**Example:**
```bash
curl http://localhost:8000/api/videos
```

---

### 8. Manual Process Trigger
Manually trigger processing for an already uploaded video.

**Endpoint:** `POST /api/process/{video_id}`

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| video_id | string | Unique video identifier |

**Request Body:**
```json
{
  "target_duration": 60,
  "selection_method": "greedy",
  "summary_type": "balanced"
}
```

**Response:**
```json
{
  "message": "Processing started",
  "task_id": "abc123def456"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/process/abc123def456 \
  -H "Content-Type: application/json" \
  -d '{
    "target_duration": 60,
    "selection_method": "greedy",
    "summary_type": "balanced"
  }'
```

---

## WebSocket Endpoints

### Real-time Processing Logs
Connect to WebSocket to receive real-time processing logs.

**Endpoint:** `WS /ws/logs/{video_id}`

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/logs/abc123def456');

ws.onmessage = (event) => {
  const log = JSON.parse(event.data);
  console.log(log);
};
```

**Message Format:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Shot detection complete",
  "stage": "shot_detection",
  "progress": 45
}
```

**Log Levels:**
- `INFO`: General information
- `SUCCESS`: Operation completed successfully
- `WARNING`: Non-critical issue
- `ERROR`: Critical error occurred

**Stages:**
- `preprocessing`
- `shot_detection`
- `feature_extraction`
- `gnn_inference`
- `assembling`
- `completed`
- `failed`

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid file format. Supported formats: mp4, avi, mov, mkv"
}
```

### 404 Not Found
```json
{
  "detail": "Video not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Processing failed: Model checkpoint not found"
}
```

---

## Rate Limiting
Currently no rate limiting is enforced. For production deployment, consider implementing rate limiting using:
- FastAPI middleware
- Redis-based token bucket
- Nginx rate limiting

---

## Authentication
Current version does not require authentication. For production:
- Implement JWT tokens
- Add API key validation
- Use OAuth2 for user management

---

## Summary Type Behaviors

### balanced
- Equal weighting of visual and audio features
- Best for general-purpose summaries
- Includes diverse shot types

### visual
- Prioritizes visual features (scene changes, action)
- Reduces weight of audio importance
- Ideal for visually-driven content (sports, nature)

### audio
- Prioritizes audio features (speech, music)
- Reduces weight of visual importance
- Ideal for lectures, podcasts, interviews

### highlight
- Selects only peak importance shots
- More aggressive filtering
- Creates shorter, more condensed summaries

---

## Best Practices

### File Upload
- Recommended max file size: 500MB
- Supported formats: MP4, AVI, MOV, MKV, WebM
- Pre-process very large videos (>1GB) before upload

### Target Duration
- Minimum: 10 seconds
- Maximum: 300 seconds (5 minutes)
- Recommended: 30-60 seconds for best results
- Should be < 30% of original video length

### Selection Method
- **greedy**: Faster (O(n log n)), good quality
- **knapsack**: Optimal (O(n²)), slightly slower
- Use greedy for videos > 30 minutes
- Use knapsack for short videos requiring optimal selection

### Polling vs WebSocket
- Use WebSocket for real-time updates in UI
- Use polling (GET /api/status) for background jobs
- Poll interval: 5-10 seconds recommended

---

## Testing with Postman

Import this collection:
```json
{
  "info": {
    "name": "VideoSum-GNN API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Upload Video",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/api/upload",
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "file", "type": "file", "src": "/path/to/video.mp4"},
            {"key": "target_duration", "value": "45"},
            {"key": "selection_method", "value": "greedy"},
            {"key": "summary_type", "value": "balanced"}
          ]
        }
      }
    }
  ]
}
```

---

## Interactive API Documentation
Visit http://localhost:8000/docs for interactive Swagger UI with:
- Try-it-out functionality
- Request/response schemas
- Example values
- Authorization testing
