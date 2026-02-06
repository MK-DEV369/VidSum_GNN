# Frontend Integration Guide - VidSum GNN Video Summarization

## 🎯 Overview

Your backend is now fully optimized for **video summarization** with complete support for:
- **Multiple Summary Types**: balanced, visual_priority, audio_priority, highlights
- **Multiple Text Lengths**: short, medium, long
- **Multiple Output Formats**: bullet points, structured, plain text

---

## 📡 API Endpoints Reference

### **1. Get Available Summary Configuration**
```
GET /api/config
```

**Purpose**: Get all available options for the UI

**Response**:
```json
{
  "text_lengths": {
    "short": "Short (50-100 words) - Brief overview",
    "medium": "Medium (100-200 words) - Standard summary",
    "long": "Long (200-400 words) - Detailed summary"
  },
  "summary_formats": {
    "bullet": "Bullet Points (• format) - Quick scanning",
    "structured": "Structured (Sections) - Organized layout",
    "plain": "Plain Text (Paragraphs) - Natural reading"
  },
  "summary_types": {
    "balanced": "Balanced - Mix of visual and audio elements",
    "visual_priority": "Visual Priority - Focus on what you see",
    "audio_priority": "Audio Priority - Focus on dialogue/narration",
    "highlights": "Highlights - Most important moments only"
  },
  "default_options": {
    "text_length": "medium",
    "summary_format": "bullet",
    "summary_type": "balanced"
  }
}
```

---

### **2. Upload Video & Start Processing**
```
POST /api/upload
Content-Type: multipart/form-data
```

**Parameters**:
```json
{
  "file": "<binary video file>",
  "text_length": "medium",           // "short" | "medium" | "long"
  "summary_format": "bullet",         // "bullet" | "structured" | "plain"
  "summary_type": "balanced",         // "balanced" | "visual_priority" | "audio_priority" | "highlights"
  "generate_video": false             // optional, for video summary generation
}
```

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Video uploaded successfully. Processing started.",
  "config": {
    "text_length": "medium",
    "summary_format": "bullet",
    "summary_type": "balanced"
  }
}
```

---

### **3. Monitor Processing Progress via WebSocket**
```
WebSocket ws://localhost:8000/ws/logs/{video_id}
```

**Real-time Log Messages**:
```json
{
  "timestamp": "2026-01-08T12:34:56.789Z",
  "level": "INFO",           // INFO | SUCCESS | WARNING | ERROR
  "message": "Detecting shots...",
  "stage": "shot_detection", // shot_detection | feature_extraction | gnn_inference | summarization | etc
  "progress": 35             // 0-100
}
```

**Example Stages Progression**:
1. UPLOAD (5-15%)
2. preprocessing (15-30%)
3. shot_detection (30-50%)
4. feature_extraction (50-70%)
5. gnn_inference (70-85%)
6. summarization (85-95%)
7. completed (100%)

---

### **4. Get Processing Status**
```
GET /api/status/{video_id}
```

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",  // queued | preprocessing | shot_detection | ... | completed | failed
  "created_at": "2026-01-08T12:34:56.789Z"
}
```

---

### **5. Get Complete Summary Results** ⭐ MAIN ENDPOINT
```
GET /api/results/{video_id}
```

**Response** (after processing complete):
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "text_summaries": {
    "bullet": "📹 Video Summary\n\n• First key point...\n• Second key point...",
    "structured": "## Balanced Summary\n\n### Overview\nThe video shows...\n\n### Key Points\n- Point 1\n- Point 2",
    "plain": "This video demonstrates... The key aspects include..."
  },
  "summary_type": "balanced",
  "fallback_used": false,
  "generated_at": "2026-01-08T12:35:20.123Z"
}
```

---

### **6. Get Specific Summary Format**
```
GET /api/summary/{video_id}/text?format=bullet
```

**Parameters**:
- `format`: "bullet" | "structured" | "plain"

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "summary": "📹 Video Summary\n\n• Point 1...\n• Point 2...",
  "format": "bullet",
  "style": "balanced",
  "generated_at": "2026-01-08T12:35:20.123Z"
}
```

---

### **7. Get Shot Importance Scores**
```
GET /api/shot-scores/{video_id}
```

**Response** (useful for visualization):
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_shots": 156,
  "shots": [
    {
      "shot_id": "550e8400-e29b-41d4-a716-446655440000_0000",
      "start_sec": 0.0,
      "end_sec": 2.5,
      "duration_sec": 2.5,
      "importance_score": 0.92
    },
    {
      "shot_id": "550e8400-e29b-41d4-a716-446655440000_0001",
      "start_sec": 2.5,
      "end_sec": 5.3,
      "duration_sec": 2.8,
      "importance_score": 0.45
    }
  ]
}
```

---

## 🎨 Frontend Implementation Examples

### **React Component for Upload**
```tsx
import React, { useState, useEffect } from 'react';

export function VideoUploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] = useState({
    text_length: 'medium',
    summary_format: 'bullet',
    summary_type: 'balanced',
    generate_video: false
  });
  const [videoId, setVideoId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [config, setConfig] = useState(null);

  // Load available options on component mount
  useEffect(() => {
    fetch('http://localhost:8000/api/config')
      .then(r => r.json())
      .then(setConfig);
  }, []);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('text_length', options.text_length);
    formData.append('summary_format', options.summary_format);
    formData.append('summary_type', options.summary_type);
    formData.append('generate_video', String(options.generate_video));

    const response = await fetch('http://localhost:8000/api/upload', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    setVideoId(data.video_id);
    
    // Start monitoring
    monitorProgress(data.video_id);
  };

  const monitorProgress = (vid: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/logs/${vid}`);
    
    ws.onmessage = (event) => {
      const log = JSON.parse(event.data);
      if (log.progress) {
        setProgress(log.progress);
      }
    };
  };

  if (!config) return <div>Loading...</div>;

  return (
    <form onSubmit={handleUpload} className="upload-form">
      <div className="form-group">
        <label htmlFor="video">Select Video</label>
        <input
          id="video"
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="text_length">Summary Length</label>
        <select
          id="text_length"
          value={options.text_length}
          onChange={(e) => setOptions({...options, text_length: e.target.value})}
        >
          {Object.entries(config.text_lengths).map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="summary_format">Summary Format</label>
        <select
          id="summary_format"
          value={options.summary_format}
          onChange={(e) => setOptions({...options, summary_format: e.target.value})}
        >
          {Object.entries(config.summary_formats).map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="summary_type">Summary Type</label>
        <select
          id="summary_type"
          value={options.summary_type}
          onChange={(e) => setOptions({...options, summary_type: e.target.value})}
        >
          {Object.entries(config.summary_types).map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </div>

      <button type="submit">Upload & Summarize</button>

      {videoId && (
        <div className="progress-container">
          <p>Processing: {progress}%</p>
          <div className="progress-bar">
            <div style={{width: `${progress}%`}}></div>
          </div>
        </div>
      )}
    </form>
  );
}
```

---

### **React Component for Display Results**
```tsx
import React, { useState, useEffect } from 'react';

export function SummaryDisplay({ videoId }: { videoId: string }) {
  const [results, setResults] = useState(null);
  const [displayFormat, setDisplayFormat] = useState('bullet');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkResults = async () => {
      const response = await fetch(`http://localhost:8000/api/results/${videoId}`);
      
      if (response.status === 200) {
        const data = await response.json();
        setResults(data);
        setLoading(false);
      } else {
        // Still processing, check again in 2 seconds
        setTimeout(checkResults, 2000);
      }
    };

    checkResults();
  }, [videoId]);

  if (loading) return <div>Waiting for summary...</div>;
  if (!results) return <div>Summary not found</div>;

  const summaryText = results.text_summaries[displayFormat];

  return (
    <div className="summary-container">
      <div className="summary-info">
        <h2>Summary Results</h2>
        <p>Type: {results.summary_type}</p>
        {results.fallback_used && <p className="warning">⚠️ Gemini fallback used</p>}
      </div>

      <div className="format-selector">
        {['bullet', 'structured', 'plain'].map(fmt => (
          <button
            key={fmt}
            className={displayFormat === fmt ? 'active' : ''}
            onClick={() => setDisplayFormat(fmt)}
          >
            {fmt.charAt(0).toUpperCase() + fmt.slice(1)}
          </button>
        ))}
      </div>

      <div className="summary-text">
        {displayFormat === 'plain' ? (
          <p>{summaryText}</p>
        ) : (
          <pre>{summaryText}</pre>
        )}
      </div>
    </div>
  );
}
```

---

### **Hook for Polling Status**
```tsx
import { useEffect, useState } from 'react';

export function useVideoStatus(videoId: string) {
  const [status, setStatus] = useState<'queued' | 'processing' | 'completed' | 'failed'>('queued');

  useEffect(() => {
    const interval = setInterval(async () => {
      const response = await fetch(`http://localhost:8000/api/status/${videoId}`);
      const data = await response.json();
      setStatus(data.status);
    }, 2000);

    return () => clearInterval(interval);
  }, [videoId]);

  return status;
}
```

---

## 🔄 Complete User Flow

### **Step 1: Load Configuration**
```
Frontend → GET /api/config → Display dropdowns
```

### **Step 2: Upload Video**
```
Frontend → POST /api/upload + params → Get video_id + open WebSocket
```

### **Step 3: Monitor Progress**
```
Frontend ← WebSocket logs ← Backend (real-time updates)
```

### **Step 4: Display Results**
```
Frontend → GET /api/results/{video_id} → Display summaries
Frontend → Allow format switching with local state (no new API calls)
```

---

## 📊 Summary Type Explanations (for UI tooltips)

### **Balanced** 👁️ 👂
Mix of visual and audio elements equally. Best for general understanding.

### **Visual Priority** 🎬
Focuses on visual scenes, camera work, objects, and visual narrative. Good for tutorial, travel, or design videos.

### **Audio Priority** 🎙️
Focuses on dialogue, narration, and spoken content. Best for interviews, podcasts, or documentary videos.

### **Highlights** ⭐
Extracts only the most exciting, important, or memorable moments. Best for sports, action, or entertainment videos.

---

## 🎯 Text Length Explanations (for UI tooltips)

### **Short (50-100 words)**
Quick overview, 2-3 key points. Perfect for busy users or social media sharing.

### **Medium (100-200 words)**
Standard summary with main ideas. Best for most use cases.

### **Long (200-400 words)**
Detailed summary with supporting details. Good for comprehensive analysis.

---

## 🎨 Format Explanations (for UI tooltips)

### **Bullet Points** •
Quick-scan format with one idea per line. Best for readability.

### **Structured**
Organized with headers and sections. Best for detailed analysis.

### **Plain Text**
Natural paragraph format. Best for reading comprehension.

---

## 🛠️ Integration Checklist

- [ ] Fetch and display configuration options from `/api/config`
- [ ] Implement file upload form with all parameter selectors
- [ ] Add WebSocket listener for real-time progress updates
- [ ] Implement status polling with `/api/status/{video_id}`
- [ ] Display all three summary formats from response
- [ ] Allow format switching without re-processing
- [ ] Show shot importance scores visualization (optional)
- [ ] Add error handling and user feedback
- [ ] Test with different video lengths and types

---

## 📝 Example cURL Commands for Testing

### **Get Configuration**
```bash
curl http://localhost:8000/api/config | jq
```

### **Upload Video**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test_video.mp4" \
  -F "text_length=medium" \
  -F "summary_format=bullet" \
  -F "summary_type=balanced"
```

### **Check Status**
```bash
curl http://localhost:8000/api/status/{video_id} | jq
```

### **Get Results**
```bash
curl http://localhost:8000/api/results/{video_id} | jq
```

### **Get Specific Format**
```bash
curl "http://localhost:8000/api/summary/{video_id}/text?format=bullet" | jq
```

---

## 🐛 Error Handling

**Invalid Parameters**:
```json
{
  "detail": "Invalid text_length: invalid_value"
}
```

**Video Not Found**:
```json
{
  "detail": "Video not found"
}
```

**Processing Failed**:
```json
{
  "status": "failed",
  "error": "Shot detection failed: ..."
}
```

---

## 🚀 Performance Optimization Tips

1. **Lazy load configuration**: Cache config in localStorage
2. **Progressive display**: Show results as soon as available, don't wait for all formats
3. **Debounce format switching**: No API calls needed, use local state
4. **Cache video IDs**: Remember recent videos for quick re-access
5. **Optimize file upload**: Show upload progress, validate file size

---

## 📞 Support & Debugging

### Check Backend Health
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
docker logs vidsumgnn-api
```

### Monitor GPU Memory
```bash
nvidia-smi
```

---

## ✨ Summary

Your backend is **production-ready** and supports:
- ✅ Multiple summary types (4 options)
- ✅ Multiple text lengths (3 options)  
- ✅ Multiple output formats (3 options)
- ✅ Real-time progress monitoring
- ✅ Graceful error handling
- ✅ Fallback mechanisms (Gemini API)

**Total combinations**: 4 × 3 × 3 = **36 different summary configurations available to users!**

