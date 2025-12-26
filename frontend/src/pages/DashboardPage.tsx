import { useState, useEffect, useRef } from "react";
import { Upload, Play, Download, AlertCircle, CheckCircle, Clock } from "lucide-react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";

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

const API_BASE = "http://localhost:8000";

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [targetDuration, setTargetDuration] = useState(30);
  const [selectionMethod, setSelectionMethod] = useState<"greedy" | "knapsack">("greedy");
  const [state, setState] = useState<ProcessingState>({
    status: "idle",
    progress: 0,
    currentStage: "Waiting for upload...",
    logs: []
  });
  const [videoId, setVideoId] = useState<string>("");
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
        
        setState(prev => ({
          ...prev,
          logs: [...prev.logs, {
            timestamp: logData.timestamp || new Date().toISOString(),
            level: logData.level || "INFO",
            message: logData.message || "",
            stage: logData.stage,
            progress: logData.progress
          }],
          progress: logData.progress || prev.progress,
          currentStage: logData.stage || prev.currentStage
        }));
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

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first");
      return;
    }

    setState(prev => ({
      ...prev,
      status: "uploading",
      progress: 0,
      currentStage: "Uploading video..."
    }));

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("target_duration", targetDuration.toString());
      formData.append("selection_method", selectionMethod);

      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const progress = (progressEvent.loaded / (progressEvent.total || 1)) * 100;
          setState(prev => ({
            ...prev,
            progress: Math.round(progress * 0.2) // Upload is 20% of total
          }));
        }
      });

      const newVideoId = response.data.video_id;
      setVideoId(newVideoId);

      setState(prev => ({
        ...prev,
        status: "processing",
        progress: 20,
        currentStage: "Processing started...",
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "SUCCESS",
          message: `Upload complete! Video ID: ${newVideoId}`
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
    <div className="min-h-screen bg-gradient-to-b from-background to-muted p-4">
      <div className="container mx-auto max-w-6xl">
        <h1 className="text-4xl font-bold mb-8">Video Summarization Dashboard</h1>

        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* Upload & Controls */}
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Upload Video</CardTitle>
                <CardDescription>Select and configure your video</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed border-primary/20 rounded-lg p-6 text-center cursor-pointer hover:border-primary/50 transition">
                  <label className="cursor-pointer">
                    <Upload className="w-8 h-8 mx-auto mb-2 text-primary" />
                    <p className="text-sm font-medium">Click to select video</p>
                    <p className="text-xs text-muted-foreground">or drag and drop</p>
                    <Input 
                      type="file" 
                      accept="video/*" 
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                </div>

                {file && (
                  <div className="bg-secondary/50 rounded-lg p-3">
                    <p className="text-sm font-medium">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
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

            {/* Controls */}
            <Card>
              <CardHeader>
                <CardTitle>Summarization Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium block mb-2">
                    Target Duration: {targetDuration}s
                  </label>
                  <Slider 
                    value={[targetDuration]}
                    onValueChange={(value) => setTargetDuration(value[0])}
                    min={10}
                    max={300}
                    step={5}
                    disabled={state.status === "processing"}
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    10 - 300 seconds
                  </p>
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">Selection Method</label>
                  <div className="space-y-2">
                    {(["greedy", "knapsack"] as const).map((method) => (
                      <label key={method} className="flex items-center gap-2 cursor-pointer">
                        <input 
                          type="radio"
                          name="method"
                          value={method}
                          checked={selectionMethod === method}
                          onChange={(e) => setSelectionMethod(e.target.value as any)}
                          disabled={state.status === "processing"}
                          className="rounded-full"
                        />
                        <span className="text-sm capitalize">{method}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {getStatusIcon()}
                  Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-sm font-medium capitalize">{state.status}</p>
                  <p className="text-xs text-muted-foreground">{state.currentStage}</p>
                </div>
                
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span>Progress</span>
                    <span>{state.progress}%</span>
                  </div>
                  <Progress value={state.progress} />
                </div>

                {state.status === "completed" && state.summaryUrl && (
                  <Button className="w-full gap-2" variant="default">
                    <Download className="w-4 h-4" />
                    Download Summary
                  </Button>
                )}

                {state.error && (
                  <div className="bg-red-50 border border-red-200 rounded p-2">
                    <p className="text-xs text-red-700">{state.error}</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Logs & Output */}
          <div className="lg:col-span-2 space-y-6">
            {/* Logs */}
            <Card>
              <CardHeader>
                <CardTitle>Processing Logs</CardTitle>
                <CardDescription>Real-time processing pipeline events</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-slate-900 text-slate-100 rounded-lg p-4 h-96 overflow-y-auto font-mono text-xs space-y-1">
                  {state.logs.length === 0 ? (
                    <p className="text-slate-500">Waiting for activity...</p>
                  ) : (
                    state.logs.map((log, idx) => (
                      <div key={idx} className="flex gap-2">
                        <span className="text-slate-500 shrink-0">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                        <span className={`font-semibold shrink-0 w-12 ${logLevelColor(log.level)}`}>
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
            </Card>

            {/* Output Preview */}
            {state.status === "completed" && (
              <Card>
                <CardHeader>
                  <CardTitle>Summary Preview</CardTitle>
                  <CardDescription>Generated video summary</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center">
                    <video 
                      controls 
                      className="w-full h-full rounded-lg"
                      src={state.summaryUrl}
                    >
                      Your browser does not support the video tag.
                    </video>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
