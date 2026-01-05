import { useState, useEffect, useRef } from "react";
import { Upload, AlertCircle, CheckCircle, Clock } from "lucide-react";
import axios from "axios";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Progress } from "../components/ui/progress";
import FloatingLines from "../components/FloatingLines";
import GradientText from "../components/GradientText";

interface Log {
  timestamp: string;
  level: string;
  message: string;
  stage?: string;
  progress?: number;
}

interface ProcessingState {
  status: "idle" | "uploading" | "processing" | "completed" | "error";
  progress: number;
  currentStage: string;
  logs: Log[];
  summaryUrl?: string;
  error?: string;
}

type TextLength = "short" | "medium" | "long";
type SummaryFormat = "bullet" | "structured" | "plain";

const API_BASE = "http://localhost:8000";

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [textLength, setTextLength] = useState<TextLength>("medium");
  const [summaryFormat, setSummaryFormat] = useState<SummaryFormat>("bullet");
  const [summaryType, setSummaryType] = useState<"balanced" | "visual" | "audio" | "highlight">("balanced");
  const [textSummary, setTextSummary] = useState<string | null>(null);
  const [state, setState] = useState<ProcessingState>({
    status: "idle",
    progress: 0,
    currentStage: "Waiting for upload...",
    logs: []
  });
  const [videoId, setVideoId] = useState<string>("");
  const [logsExpanded, setLogsExpanded] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.logs]);

  // Connect to WebSocket for logs
  useEffect(() => {
    if (!videoId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/logs/${videoId}`;
    
    console.log(`[Dashboard] Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[WebSocket] Connected");
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "INFO",
          message: "Connected to real-time log stream"
        }]
      }));
    };

    ws.onmessage = (event) => {
      try {
        const logData = JSON.parse(event.data);
        console.log("[WebSocket] Received:", logData);
        const isComplete = logData.stage === "completed" && logData.level === "SUCCESS";
        
        setState(prev => ({
          ...prev,
          logs: [...prev.logs, {
            timestamp: logData.timestamp || new Date().toISOString(),
            level: logData.level || "INFO",
            message: logData.message || "",
            stage: logData.stage,
            progress: logData.progress
          }],
          progress: logData.progress ?? prev.progress,
          currentStage: logData.stage || prev.currentStage,
          status: isComplete ? "completed" : prev.status,
          summaryUrl: isComplete ? `${API_BASE}/api/download/${videoId}` : prev.summaryUrl
        }));
        
        // Fetch text summary when processing completes
        if (isComplete) {
          fetchTextSummary(videoId);
        }
      } catch (err) {
        console.error("[WebSocket] Parse error:", err);
      }
    };

    ws.onerror = (error) => {
      console.error("[WebSocket] Error:", error);
    };

    ws.onclose = () => {
      console.log("[WebSocket] Disconnected");
    };

    wsRef.current = ws;

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [videoId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setState(prev => ({
        ...prev,
        logs: [{
          timestamp: new Date().toISOString(),
          level: "INFO",
          message: `Selected file: ${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)`
        }]
      }));
    }
  };

  const fetchTextSummary = async (videoId: string) => {
    try {
      const response = await axios.get(`${API_BASE}/api/summary/${videoId}/text`, {
        params: { format: summaryFormat }
      });
      setTextSummary(response.data.summary);
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "SUCCESS",
          message: `Text summary (${summaryFormat}) generated successfully`
        }]
      }));
    } catch (error: any) {
      console.error("[Fetch Summary] Error:", error);
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "WARNING",
          message: "Could not fetch text summary"
        }]
      }));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first");
      return;
    }

    setTextSummary(null);
    setState(prev => ({
      ...prev,
      status: "uploading",
      progress: 0,
      currentStage: "Uploading video...",
      summaryUrl: undefined,
      error: undefined
    }));

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("text_length", textLength);
      formData.append("summary_format", summaryFormat);
      formData.append("summary_type", summaryType);

      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const progress = (progressEvent.loaded / (progressEvent.total || 1)) * 100;
          setState(prev => ({
            ...prev,
            status: "uploading",
            progress: Math.min(100, Math.round(progress)),
            currentStage: "Uploading video..."
          }));
        }
      });

      const uploadedVideoId = response.data.video_id;
      setVideoId(uploadedVideoId);

      setState(prev => ({
        ...prev,
        status: "processing",
        progress: Math.max(prev.progress, 20),
        currentStage: "Processing video...",
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "SUCCESS",
          message: response.data.message || "Upload successful. Processing started.",
          stage: "UPLOAD"
        }]
      }));
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || "Upload failed";
      console.error("[Upload] Error:", errorMsg);
      setState(prev => ({
        ...prev,
        status: "error",
        error: errorMsg,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "ERROR",
          message: errorMsg
        }]
      }));
    }
  };

  const getStatusIcon = () => {
    switch (state.status) {
      case "processing":
        return <Clock className="w-5 h-5 animate-spin text-blue-500" />;
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "error":
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Upload className="w-5 h-5 text-gray-500" />;
    }
  };

  const logLevelColor = (level: string) => {
    switch (level) {
      case "ERROR": return "text-red-600";
      case "WARNING": return "text-yellow-600";
      case "SUCCESS": return "text-green-600";
      default: return "text-gray-600";
    }
  };

  return (
    <div className="h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 p-4 flex flex-col overflow-hidden" style={{ position: 'relative' }}>
      {/* Animated Background */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0 }}>
        <FloatingLines
          linesGradient={['#10b981', '#06b6d4', '#3b82f6', '#6366f1', '#8b5cf6']}
          enabledWaves={['middle', 'bottom']}
          lineCount={[6, 5]}
          lineDistance={[5, 4]}
          animationSpeed={0.8}
          interactive={true}
          parallax={true}
          mixBlendMode="screen"
        />
      </div>

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1 }} className="h-full">
      <div className="container mx-auto max-w-full h-full flex flex-col gap-4">
        {/* Main Content Area */}
        <div className={`grid grid-cols-3 gap-4 transition-all ${logsExpanded ? 'h-[60%]' : 'flex-1'} overflow-hidden`}>
          {/* Upload & Controls - 1/3 width */}
          <div className="col-span-1 space-y-3 overflow-y-auto pr-2">
            <Card>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed border-primary/20 rounded-lg p-6 text-center cursor-pointer hover:border-primary/50 transition">
                  <label className="cursor-pointer">
                    <Upload className="w-8 h-8 mx-auto mb-2 text-primary" />
                    <p className="text-sm font-medium text-white">Click to select video</p>
                    <p className="text-xs text-slate-300">or drag and drop</p>
                    <Input 
                      type="file" 
                      accept="video/*" 
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                </div>

                {file && (
                  <div className="bg-white/5 rounded-lg p-3">
                    <p className="text-sm font-medium text-white">{file.name}</p>
                    <p className="text-xs text-slate-300">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                )}

                <Button 
                  onClick={handleUpload}
                  disabled={!file || state.status === "processing"}
                  className="w-full"
                >
                  {state.status === "uploading" ? "Uploading..." : "Upload & Process"}
                </Button>
              </CardContent>
            </Card>

            {/* Status */}
            <Card className="bg-white/10 border-white/20 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  {getStatusIcon()}
                  <span className="capitalize">{state.status}</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-xs text-white">{state.currentStage}</p>
                
                <div>
                  <div className="flex justify-between text-xs mb-1 text-white">
                    <span>Progress</span>
                    <span className="font-semibold">{state.progress}%</span>
                  </div>
                  <Progress value={state.progress} className="h-2" />
                </div>

                {state.error && (
                  <div className="bg-red-900/30 border border-red-500/50 rounded p-2">
                    <p className="text-xs text-red-400">{state.error}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Controls */}
            <Card className="bg-white/10 border-white/20 backdrop-blur-sm\">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">
                  <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >
                    Settings
                  </GradientText>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Text Length & Format - 2 Column Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium block text-white">Text Length</label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["short", "medium", "long"] as TextLength[]).map(len => (
                        <Button
                          key={len}
                          size="sm"
                          variant={textLength === len ? "default" : "outline"}
                          onClick={() => setTextLength(len)}
                          disabled={state.status === "processing"}
                          className="capitalize text-xs px-2"
                        >
                          {len}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium block text-white">Format</label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["bullet", "structured", "plain"] as SummaryFormat[]).map(fmt => (
                        <Button
                          key={fmt}
                          size="sm"
                          variant={summaryFormat === fmt ? "default" : "outline"}
                          onClick={() => setSummaryFormat(fmt)}
                          disabled={state.status === "processing"}
                          className="capitalize text-xs px-2"
                        >
                          {fmt}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Content Type */}
                <div>
                  <label className="text-sm font-medium block mb-2.5 text-white">Content Type</label>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="flex items-center gap-2 cursor-pointer hover:bg-white/10 px-2 py-1.5 rounded-md transition-colors">
                      <input 
                        type="radio"
                        name="summaryType"
                        value="balanced"
                        checked={summaryType === "balanced"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing"}
                        className="text-primary focus:ring-primary"
                      />
                      <span className="text-xs font-medium text-white">Balanced</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer hover:bg-white/10 px-2 py-1.5 rounded-md transition-colors">
                      <input 
                        type="radio"
                        name="summaryType"
                        value="visual"
                        checked={summaryType === "visual"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing"}
                        className="text-primary focus:ring-primary"
                      />
                      <span className="text-xs font-medium text-white">Visual Priority</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer hover:bg-white/10 px-2 py-1.5 rounded-md transition-colors">
                      <input 
                        type="radio"
                        name="summaryType"
                        value="audio"
                        checked={summaryType === "audio"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing"}
                        className="text-primary focus:ring-primary"
                      />
                      <span className="text-xs font-medium text-white">Audio Priority</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer hover:bg-white/10 px-2 py-1.5 rounded-md transition-colors">
                      <input 
                        type="radio"
                        name="summaryType"
                        value="highlight"
                        checked={summaryType === "highlight"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing"}
                        className="text-primary focus:ring-primary"
                      />
                      <span className="text-xs font-medium text-white">Highlight</span>
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right: Output - 2/3 width */}
          <div className="col-span-2 flex flex-col gap-4 overflow-hidden">
            {state.status === "completed" || textSummary ? (
              <>
                {/* Text Summary */}
                {textSummary && (
                  <Card className="flex-1 bg-white/10 border-white/20 backdrop-blur-sm">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">
                        <GradientText
                          colors={['#10b981', '#06b6d4', '#3b82f6', '#06b6d4', '#10b981']}
                          animationSpeed={4}
                        >
                          Text Summary ({summaryFormat} • {textLength})
                        </GradientText>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="h-[calc(100%-60px)] overflow-y-auto">
                      <div className="bg-slate-50 rounded-lg p-4 prose prose-sm max-w-none">
                        <pre className="whitespace-pre-wrap font-sans text-sm">
                          {textSummary}
                        </pre>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            ) : (
              <Card className="flex-1 flex items-center justify-center bg-white/10 border-white/20 backdrop-blur-sm">
                <CardContent>
                  <p className="text-slate-400 text-center">
                    {state.status === "idle" 
                      ? "Upload a video to start" 
                      : "Processing... summary will appear here"}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Bottom: Toggleable Logs */}
        <div className={`transition-all overflow-hidden ${logsExpanded ? 'h-[35%]' : 'h-12'}`}>
          <Card className="h-full flex flex-col bg-white/10 border-white/20 backdrop-blur-sm\">
            <CardHeader 
              className="pb-2 pt-3 cursor-pointer hover:bg-white/20 transition-colors"
              onClick={() => setLogsExpanded(!logsExpanded)}
            >
              <CardTitle className="flex items-center justify-between text-sm">
                <span>
                  <GradientText
                    colors={['#f59e0b', '#f97316', '#ef4444', '#f97316', '#f59e0b']}
                    animationSpeed={4}
                  >
                    Processing Logs ({state.logs.length})
                  </GradientText>
                </span>
                <Button variant="ghost" size="sm" className="h-6">
                  {logsExpanded ? "Minimize ▼" : "Expand ▲"}
                </Button>
              </CardTitle>
            </CardHeader>
            {logsExpanded && (
              <CardContent className="flex-1 overflow-hidden pt-2">
                <div className="bg-slate-900 text-slate-100 rounded-lg p-3 h-full overflow-y-auto font-mono text-xs space-y-1">
                  {state.logs.length === 0 ? (
                    <p className="text-slate-500">Waiting for activity...</p>
                  ) : (
                    state.logs.map((log, idx) => (
                      <div key={idx} className="flex gap-2">
                        <span className="text-slate-500 shrink-0">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                        <span className={`font-semibold shrink-0 w-16 ${logLevelColor(log.level)}`}>
                          {log.level}
                        </span>
                        <span className="flex-1">{log.message}</span>
                        {log.progress !== undefined && (
                          <span className="text-slate-500 shrink-0">{log.progress}%</span>
                        )}
                      </div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>
      </div>
    </div>
  );
}
