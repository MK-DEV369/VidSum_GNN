# Dashboard & Backend Update Summary

## Changes Made

### Frontend (DashboardPage.tsx)

**1. Replaced Duration with Text Length**
- Removed `targetDuration` slider (10-300 seconds)
- Added text length selector: `short`, `medium`, `long`
  - Short: 50-100 words
  - Medium: 100-200 words
  - Long: 200-400 words

**2. Added Summary Format Selection**
- Single format selector instead of tabs
- Options: `bullet`, `structured`, `plain`
- User picks ONE format at a time

**3. Added Video Generation Toggle**
- New checkbox: "Generate Video Summary"
- Default: OFF (text-only mode)
- Only generates video MP4 if enabled

**4. Single-Screen Layout**
- Changed from scrollable to fixed-height layout (`h-screen overflow-hidden`)
- 4-column grid: Settings (1) | Logs (1) | Output (2)
- Reduced padding and spacing for compact view
- Output section shows video only if `generateVideo` was enabled

**5. Simplified UI**
- Removed upload card descriptions
- Smaller content type radio labels
- Combined settings into single card with reduced spacing
- Status card with smaller font sizes

### Backend (routes.py, tasks.py, model_service.py)

**1. Updated Upload Endpoint** (`vidsum_gnn/api/routes.py`)
```python
@router.post("/upload")
async def upload_video(
    file: UploadFile,
    text_length: str = "medium",      # NEW
    summary_format: str = "bullet",   # NEW
    summary_type: str = "balanced",   # kept
    generate_video: bool = False      # NEW
)
```

**2. Updated Text Summary Endpoint**
```python
@router.get("/summary/{video_id}/text")
async def get_text_summary(video_id: str, format: str = "bullet"):
    # Returns single format summary, not all three
    return {"summary": "...", "format": format}
```

**3. Model Service Changes** (`vidsum_gnn/model_service.py`)

**LLMSummarizer.summarize()** now accepts:
- `text_length`: Determines min/max token counts
  - short: 30-70 tokens
  - medium: 70-150 tokens
  - long: 150-300 tokens
- `summary_format`: Returns ONE format
  - bullet: Variable bullet count (3/5/8 based on length)
  - structured: Formatted with sections (varies by length)
  - plain: Raw summary text

**Returns**: Single string (not dict with all 3 formats)

**4. Processing Pipeline** (`vidsum_gnn/api/tasks.py`)

- Extracts `text_length`, `summary_format`, `generate_video` from config
- Passes these to `model_service.process_video_end_to_end()`
- **Video assembly is conditional:**
  ```python
  if generate_video:
      # Select shots and create MP4
  else:
      # Skip video generation
  ```
- Generates all 3 formats for database storage (for future retrieval)
- Only returns the user-requested format initially

## Key Behavioral Changes

### Text Summary Generation
**Before:**
- Always generated 3 formats (bullet/structured/plain) together
- All 3 were identical or very similar
- Length was fixed

**Now:**
- Generates 1 format per user request (on-demand)
- Each format has distinct structure:
  - **Bullet**: `• Item\n• Item\n...` (3-8 bullets based on length)
  - **Structured**: Headers, sections, metadata footer
  - **Plain**: Raw paragraph text
- Length varies (short/medium/long controls word count)

### Video Summary
**Before:**
- Always generated video MP4 (required parameter: target_duration)

**Now:**
- Optional toggle, default OFF
- When OFF: No video processing, faster pipeline
- When ON: Uses default 60s duration with greedy selection

## Usage

### Start Frontend
```powershell
cd frontend
npm install
npm run dev
```

### Start Backend
```powershell
# Activate venv
& "venv\Scripts\Activate.ps1"

# Start API
python -m vidsum_gnn.api.main
```

### Testing Workflow

1. **Upload video**
   - Select file
   - Choose text length (short/medium/long)
   - Choose format (bullet/structured/plain)
   - Choose content type (balanced/visual/audio/highlight)
   - Toggle "Generate Video Summary" ON or OFF
   - Click Upload

2. **Observe logs** in real-time

3. **View output:**
   - If video enabled: Video player + text summary
   - If video disabled: Text summary only

4. **Change format/length:**
   - Frontend will re-fetch with new `format` parameter
   - Backend returns different formatted text from same processed data

## Files Modified

### Frontend
- `frontend/src/pages/DashboardPage.tsx` (major refactor)

### Backend
- `vidsum_gnn/api/routes.py` (upload params, text endpoint)
- `vidsum_gnn/api/tasks.py` (conditional video gen, new params)
- `vidsum_gnn/model_service.py` (LLMSummarizer format logic)

## Breaking Changes

1. **API Contract**: Upload endpoint params changed
   - Old: `target_duration`, `selection_method`
   - New: `text_length`, `summary_format`, `summary_type`, `generate_video`

2. **Text Summary Response**: Changed from dict to single string
   - Old: `{"bullet": "...", "structured": "...", "plain": "..."}`
   - New: `{"summary": "...", "format": "bullet"}`

3. **Video Generation**: Now optional, not guaranteed
   - Check `generate_video` flag in config
   - `video_path` may be None in Summary record

## Notes

- All 3 formats are still stored in DB for flexibility
- Video generation defaults to 60s duration (not user-configurable currently)
- Single-screen layout tested on 1920x1080 resolution
- Logs section scrollable, output section uses flex layout
