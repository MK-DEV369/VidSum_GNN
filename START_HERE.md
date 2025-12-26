# 🎬 VIDSUM-GNN - Get Started in 30 Seconds

## Right Now - Everything is Running!

All services are already deployed and operational. Just open these URLs:

### 🌐 Frontend (Main Application)
**http://localhost:5173**

- Home page with project overview
- Dashboard for video upload
- Real-time processing monitoring

### 📚 API Documentation
**http://localhost:8000/docs**

- Interactive Swagger UI
- Try endpoints directly
- See request/response schemas

---

## Quick Demo (2 Minutes)

### 1. Go to Dashboard
Click **"Try Dashboard"** button on the home page

### 2. Upload a Video
- Drag & drop any video file (MP4, WebM, etc.)
- Or click to browse

### 3. Configure
- **Target Duration**: Adjust with slider (default 30s)
- **Method**: Choose Greedy or Knapsack

### 4. Process
- Click **"Upload & Process"**
- Watch real-time logs stream in
- See progress bar advance through stages

### 5. Download
- When complete, click **"Download Summary"**
- Enjoy your concise video summary!

---

## What's Happening Behind the Scenes

When you upload a video:

1. **Upload** (15%) - File saved to server
2. **Shot Detection** (15%) - FFmpeg detects scene changes
3. **Feature Extraction** (40%) - ViT extracts visual features, Wav2Vec2 extracts audio
4. **Graph Construction** (15%) - Creates scene graph with temporal/semantic/audio edges
5. **GNN Inference** (10%) - Graph Attention Network scores importance of each scene
6. **Summary Assembly** (5%) - FFmpeg concatenates selected scenes

---

## Service Addresses

| Service | URL | Login |
|---------|-----|-------|
| Frontend | http://localhost:5173 | No login |
| API | http://localhost:8000 | No login |
| Database | localhost:5432 | postgres/password |
| Redis | localhost:6379 | No password |

---

## Stopping Services

```bash
docker-compose down
```

Restart anytime with:
```bash
docker-compose up -d
```

---

## Common Questions

**Q: How long does summarization take?**  
A: A 10-minute video typically takes 5-10 minutes. Depends on GPU availability and video resolution.

**Q: What video formats are supported?**  
A: MP4, WebM, AVI, MOV, and most formats supported by FFmpeg.

**Q: Can I summarize multiple videos?**  
A: Yes! Upload them one at a time, each gets a unique ID and processes independently.

**Q: Where are my videos stored?**  
A: In `./data/uploads/` and `./data/outputs/` directories on your machine.

**Q: Does it need GPU?**  
A: GPU is optional but highly recommended. Works on CPU but much slower (3-5x).

---

## Troubleshooting

**Frontend not loading?**
- Ensure port 5173 is accessible
- Try: `docker-compose restart frontend`
- Check logs: `docker logs vidsum_gnn_frontend`

**API errors?**
- Try: `docker-compose restart ml_api`
- Check logs: `docker logs vidsum_gnn_ml_api`

**Can't access from other computers?**
- Change `localhost` to your machine's IP address
- E.g., `http://192.168.x.x:5173`

---

## Next Steps

1. **Try it now** - Upload a test video!
2. **Read documentation** - See README.md for full details
3. **Explore API** - Visit http://localhost:8000/docs
4. **Check logs** - View real-time processing in dashboard
5. **Download results** - Save your video summaries

---

**Enjoy AI-powered video summarization! 🎥✨**

Need help? See the comprehensive documentation in the project root.
