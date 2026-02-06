# Quick Start: Testing the New Important Shots Feature

## Prerequisites

✅ Ensure you have:
- FFmpeg installed (`ffmpeg --version`)
- Docker running
- Python venv activated
- Project running locally or in Docker

## Step 1: Start the Application

### Option A: Docker
```bash
docker-compose up --build
```

### Option B: Local Development
```bash
# Terminal 1: Backend
cd /path/to/project
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m vidsum_gnn.api.main

# Terminal 2: Frontend
cd frontend
npm run dev
```

## Step 2: Verify Setup

1. **Check Backend Health**
   ```
   Open: http://localhost:8000/health
   Expected: {"status": "healthy", "database": "connected"}
   ```

2. **Check Frontend**
   ```
   Open: http://localhost:5173
   Expected: Dashboard loads with upload section
   ```

3. **Check Logs**
   ```bash
   docker logs vidsum_gnn_api | tail -20
   # Should show: "✓ VidSum GNN API started on port 8000"
   ```

## Step 3: Upload a Test Video

1. **Prepare Video** (any MP4, WebM, or MOV)
   - Minimum: 30 seconds
   - Recommended: 2-5 minutes for quick testing
   - Maximum: 1 hour (for testing)

2. **On Dashboard:**
   - Click "Click to select video"
   - Select your test video
   - Verify file shows: `filename (XXX MB)`

3. **Configure Summary** (important - test your settings)
   - **Text Length:** Select "medium"
   - **Format:** Select "bullet"
   - **Content Type:** Select "visual_priority"
   - These are your test parameters

4. **Click "Upload & Process"**
   - Progress bar appears
   - Logs begin streaming (in bottom section)
   - Status updates in real-time

## Step 4: Monitor Processing

### Watch the Logs
Scroll through logs to see stages:

```
[00:00] ✓ Upload successful
[00:05] ✓ Preprocessing started
[00:15] ✓ Shot detection: Detected 25 shots
[00:20] ✓ Feature extraction...
[00:45] ✓ GNN inference (scores: 0.234 - 0.892)
[00:50] ✓ Text summary generated (bullet format)
[00:55] ✓ Creating merged video...    ← NEW STAGE
[01:10] ✓ Merged video created
[01:11] ✓ Cleaning up temporary files... ← NEW STAGE
[01:15] ✓ Temporary files cleaned up
[01:16] ✓ Processing complete
```

### Expected Timing
- **Quick Video (30s - 1min):** 2-3 minutes total
- **Medium Video (2-5min):** 4-6 minutes total
- **Long Video (10min+):** 7-10 minutes total

## Step 5: Verify Output

### Check Dashboard Display

After processing completes, verify:

1. **Top Section - Video Player** ✅
   - Video player visible with "📹 Important Shots Compilation" title
   - Play/pause controls present
   - Video timeline showing
   - Typical length: 2-5 minutes (depending on original)

2. **Bottom Section - Text Summary** ✅
   - "📄 Text Summary (bullet • medium)" title
   - Bullet points visible (• format)
   - Download buttons available
   - "Read Aloud" button functional

### Expected Output Example

**Top Section (Video):**
```
📹 Important Shots Compilation
[▶ play button] ━━━━━━●━━━━━━ 00:30 / 02:15
```

**Bottom Section (Summary):**
```
📄 Text Summary (bullet • medium)

• First key point about visual content
• Second important observation
• Third significant element
• And more bullet points...

[Download TXT] [Download JSON] [Read Aloud]
```

## Step 6: Verify Cleanup

### Check Backend Logs
```bash
docker logs vidsum_gnn_api | grep -i "cleanup"
```

Expected output:
```
Cleaning up temporary files...
✓ Cleaned up uploaded file: /path/to/uploads/...
✓ Cleaned up processed directory: /path/to/processed/{video_id}/
✓ Cleaned up canonical video: /path/to/temp/...
✓ Temporary files cleaned up
```

### Check Disk Space
```bash
# Before upload
du -sh /path/to/data/

# After processing
du -sh /path/to/data/
# Should be similar (temp files cleaned)
```

### Check Database
```bash
docker exec -it vidsum_gnn_db psql -U postgres -d vidsum_gnn -c \
  "SELECT video_id, video_path, config_json->>'merged_video_enabled' FROM summary ORDER BY created_at DESC LIMIT 1;"
```

Expected:
```
video_id | video_path | merged_video_enabled
---------+------------------------------------+---------------------
abc123   | /path/to/merged_video.mp4 | true
```

## Step 7: Download and Verify Files

### Download Text Summary

1. In dashboard, click **"Download TXT"**
2. File saves as: `summary_{video_id}_{length}_{format}.txt`
3. Open in text editor, verify bullet points

### Download JSON Summary

1. Click **"Download JSON"**
2. File saves as: `summary_{video_id}_{length}_{format}.json`
3. Verify structure:
```json
{
  "video_id": "abc123",
  "summary_format": "bullet",
  "text_length": "medium",
  "summary_type": "visual_priority",
  "content": "• Bullet 1\n• Bullet 2...",
  "generated_at": "2026-01-15T10:30:00Z"
}
```

### Download Merged Video

1. Browser download should work automatically
2. File saves as: `summary_{video_id}.mp4`
3. Open in VLC/media player to verify
4. Expected length: 2-5 minutes
5. Expected size: 20-50MB

## Step 8: Test Read Aloud

1. Click **"Read Aloud (Text-to-Speech)"** button
2. Browser should read summary aloud
3. Click **"Stop Reading"** to stop
4. Verify button toggles between states

## Common Issues & Solutions

### Issue: Video not showing in top section

**Cause:** Merge failed during processing
**Fix:**
1. Check logs for "Creating merged video" errors
2. Verify FFmpeg installed: `ffmpeg -version`
3. Ensure adequate disk space
4. Re-upload and retry

**Debug:**
```bash
docker logs vidsum_gnn_api | grep -i "merge"
```

---

### Issue: "Processing... will appear here" stays longer than expected

**Cause:** Processing taking longer than expected
**Fix:**
1. Check backend logs
2. Monitor system resources (CPU/Memory)
3. For 30+ minute videos, this is normal
4. Wait for completion (don't refresh page)

**Monitor Resources:**
```bash
docker stats vidsum_gnn_api
```

---

### Issue: Summary text appears but no video

**Cause:** Video merge succeeded but frontend not displaying
**Fix:**
1. Refresh page (F5)
2. Check browser console for errors: Press F12
3. Verify `videoUrl` in browser console:
   ```javascript
   // In browser console
   console.log(document.location)  // Should be http://localhost:5173
   ```

---

### Issue: Download buttons don't work

**Cause:** Text summary not fetched yet
**Fix:**
1. Wait a few seconds for text summary to load
2. Check network tab in browser devtools (F12)
3. Look for failed requests to `/api/summary/`

---

### Issue: Read Aloud doesn't work

**Cause:** Browser speech synthesis not supported
**Fix:**
1. Test in Chrome/Edge/Safari (Firefox limited)
2. Check system audio output
3. Try slower speed: Edit browser console
   ```javascript
   document.querySelector('[class*=speaker]')?.click()
   ```

---

## Performance Testing

### Quick Performance Check

1. **Upload 2-minute video:**
   - Total time should be ~4-5 minutes
   - Merge stage: ~30-40 seconds
   - Cleanup stage: ~5-10 seconds

2. **Monitor CPU:**
   ```bash
   watch -n 1 "docker stats --no-stream"
   ```
   - During merge: High CPU (FFmpeg)
   - During cleanup: Low CPU (I/O)

3. **Monitor Memory:**
   ```bash
   docker stats --no-stream | grep vidsum_gnn_api
   ```
   - Stable around 1-2GB
   - Spikes during feature extraction (normal)

### Benchmark Results (Expected)
| Stage | CPU | Memory | Time |
|-------|-----|--------|------|
| Preprocessing | 30% | 800MB | 30s |
| Shot Detection | 20% | 900MB | 20s |
| Features | 60% | 1.5GB | 30s |
| GNN | 40% | 1.2GB | 20s |
| **Merge** | **80%** | **800MB** | **40s** |
| **Cleanup** | **10%** | **700MB** | **8s** |
| Total | - | - | 148s (2:28) |

## Success Criteria ✅

Process is working correctly if:

- [ ] Dashboard loads without errors
- [ ] Upload succeeds with progress bar
- [ ] Logs show all stages progressing
- [ ] "Creating merged video" appears in logs
- [ ] "Cleaning up temporary files" appears in logs
- [ ] Video player shows in top section
- [ ] Text summary shows in bottom section
- [ ] Download buttons work
- [ ] Read Aloud plays audio
- [ ] Database shows video_path populated
- [ ] Disk space back to baseline after cleanup
- [ ] Processing completes within expected time

## Next Steps

Once testing succeeds:

1. **Test Different Configurations**
   - Try: bullet, structured, plain formats
   - Try: short, medium, long lengths
   - Try: balanced, visual, audio, highlights types

2. **Test Edge Cases**
   - Very short videos (< 1 min)
   - Very long videos (> 30 min)
   - Low quality videos
   - Different frame rates

3. **Load Testing**
   - Upload 5 videos in sequence
   - Check for memory leaks
   - Verify cleanup happens for all

4. **Error Simulation**
   - Stop FFmpeg during merge
   - Fill disk during cleanup
   - Kill process during merge
   - Check graceful failure

## Support

**If you encounter issues:**

1. Check logs: `docker logs vidsum_gnn_api`
2. Review errors in browser console (F12)
3. Check database state
4. Restart containers: `docker-compose restart`
5. Full rebuild if needed: `docker-compose down && docker-compose up --build`

---

**Status:** Ready to Test ✅
**Last Verified:** 2026-01-15
**Expected Duration:** 5-10 minutes for full test
