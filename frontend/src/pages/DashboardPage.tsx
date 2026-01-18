import { useState, useEffect, useRef } from "react";
import { Upload, AlertCircle, CheckCircle, Clock, Download, Volume2, StopCircle, History, X, Trash2 } from "lucide-react";
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

interface SummaryHistory {
  id: string;
  title: string;
  summary: string;
  format: SummaryFormat;
  length: TextLength;
  type: "balanced" | "visual" | "audio" | "highlight";
  timestamp: string;
  videoId: string;
}

interface ProcessingState {
  status: "idle" | "uploading" | "processing" | "completed" | "error";
  progress: number;
  currentStage: string;
  logs: Log[];
  summaryUrl?: string;
  videoUrl?: string;
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
  const [processingConfig, setProcessingConfig] = useState<{format: SummaryFormat, length: TextLength, type: string} | null>(null);
  const [logsExpanded, setLogsExpanded] = useState<boolean>(false);
  const [isReading, setIsReading] = useState<boolean>(false);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const speechSynthesisRef = useRef<SpeechSynthesisUtterance | null>(null);
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const [summaryHistory, setSummaryHistory] = useState<SummaryHistory[]>([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<SummaryHistory | null>(null);

  // Scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.logs]);

  // Load history from localStorage
  useEffect(() => {
    const stored = localStorage.getItem("summaryHistory");
    if (stored) {
      try {
        setSummaryHistory(JSON.parse(stored));
      } catch (e) {
        console.error("Failed to load history:", e);
      }
    }
  }, []);

  // Load available voices
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      // Filter for only Google voices in English
      const googleVoices = voices.filter(voice => 
        voice.lang.startsWith('en') && voice.name.includes('Google')
      );
      setAvailableVoices(googleVoices);
      
      // Select first Google voice by default
      const preferredVoice = googleVoices[0];
      setSelectedVoice(preferredVoice);
    };

    loadVoices();
    
    // Chrome loads voices asynchronously
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);

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
          summaryUrl: isComplete ? `${API_BASE}/api/download/${videoId}` : prev.summaryUrl,
          videoUrl: isComplete ? `${API_BASE}/api/download/${videoId}` : prev.videoUrl
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
      // Use the format that was actually sent during processing
      const formatToFetch = processingConfig?.format || summaryFormat;
      const response = await axios.get(`${API_BASE}/api/summary/${videoId}/text`, {
        params: { format: formatToFetch }
      });
      setTextSummary(response.data.summary);
      
      // Update the display to show actual format returned
      const actualFormat = response.data.format || formatToFetch;
      setSummaryFormat(actualFormat);
      
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "SUCCESS",
          message: `Text summary (${actualFormat}) generated successfully`
        }]
      }));
      
      // Save to history
      const title = file?.name?.replace(/\.[^/.]+$/, "") || `Video ${videoId.slice(0, 8)}`;
      saveToHistory(videoId, title, response.data.summary);
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

  const saveToHistory = (videoId: string, title: string, summary: string) => {
    const historyItem: SummaryHistory = {
      id: `${videoId}_${Date.now()}`,
      title: title,
      summary: summary,
      format: summaryFormat,
      length: textLength,
      type: summaryType,
      timestamp: new Date().toISOString(),
      videoId: videoId
    };
    
    const updated = [historyItem, ...summaryHistory].slice(0, 50); // Keep last 50
    setSummaryHistory(updated);
    localStorage.setItem("summaryHistory", JSON.stringify(updated));
  };

  const clearHistory = () => {
    if (confirm("Are you sure you want to clear all summary history?")) {
      setSummaryHistory([]);
      localStorage.removeItem("summaryHistory");
      setSelectedHistoryItem(null);
    }
  };

  const deleteHistoryItem = (id: string) => {
    const updated = summaryHistory.filter(item => item.id !== id);
    setSummaryHistory(updated);
    localStorage.setItem("summaryHistory", JSON.stringify(updated));
    if (selectedHistoryItem?.id === id) {
      setSelectedHistoryItem(null);
    }
  };

  const handleReset = () => {
    setFile(null);
    setTextSummary(null);
    setState({ status: "idle", progress: 0, currentStage: "Waiting for upload...", logs: [] });
    setVideoId("");
    setProcessingConfig(null);
    setIsReading(false);
    window.speechSynthesis.cancel();
    try { if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) wsRef.current.close(); } catch {}
  };

  const handleDownloadText = () => {
    if (!textSummary) return;
    const element = document.createElement("a");
    const file = new Blob([textSummary], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `summary_${videoId}_${textLength}_${summaryFormat}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleDownloadJSON = () => {
    if (!textSummary) return;
    const jsonData = {
      video_id: videoId,
      summary_format: summaryFormat,
      text_length: textLength,
      summary_type: summaryType,
      content: textSummary,
      generated_at: new Date().toISOString()
    };
    const element = document.createElement("a");
    const file = new Blob([JSON.stringify(jsonData, null, 2)], { type: "application/json" });
    element.href = URL.createObjectURL(file);
    element.download = `summary_${videoId}_${textLength}_${summaryFormat}.json`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleReadAloud = () => {
    if (!textSummary) return;

    if (isReading) {
      window.speechSynthesis.cancel();
      setIsReading(false);
      return;
    }

    // Clean text for speech (remove markdown formatting)
    const cleanText = textSummary
      .replace(/[•\-*]/g, "")
      .replace(/[#*_`\[\]]/g, "")
      .replace(/\n+/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    
    // Apply selected voice
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
    
    // Improved voice parameters for more natural speech
    utterance.rate = 0.95;  // Slightly slower for clarity
    utterance.pitch = 1.0;  // Natural pitch
    utterance.volume = 1.0;

    utterance.onend = () => {
      setIsReading(false);
    };

    utterance.onerror = () => {
      setIsReading(false);
      console.error("Speech synthesis error");
    };

    speechSynthesisRef.current = utterance;
    setIsReading(true);
    window.speechSynthesis.speak(utterance);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first");
      return;
    }

    setTextSummary(null);
    
    // Store the config being used for this processing
    const config = {
      format: summaryFormat,
      length: textLength,
      type: summaryType
    };
    setProcessingConfig(config);
    
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
    <div className="h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 p-4 flex flex-col overflow-hidden text-white" style={{ position: 'relative' }}>
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
            <Card className="bg-white/10 border-white/20 backdrop-blur-sm">
              <CardContent className="space-y-4">
                <div className="border-2 border-dashed border-primary/20 rounded-lg p-6 text-center cursor-pointer hover:border-primary/50 transition">
                  <label className="cursor-pointer">
                    <Upload className="w-8 h-8 mx-auto mb-2 text-primary" />
                    <p className="text-sm font-medium text-white">Click to select video</p>
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
                    <p className="text-xs text-white/80">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2">
                  <Button 
                    onClick={handleUpload}
                    disabled={!file || state.status === "processing"}
                    className="w-full bg-gradient-to-r from-violet-600 via-pink-500 to-blue-600 hover:from-violet-700 hover:via-pink-600 hover:to-blue-700 text-white font-semibold transition-all"
                  >
                    {state.status === "uploading" ? "Uploading..." : "Upload & Process"}
                  </Button>
                  <Button
                    onClick={handleReset}
                    disabled={state.status === "processing"}
                    className="w-full bg-gradient-to-r from-pink-600 to-violet-600 hover:from-pink-700 hover:to-violet-700 text-white font-semibold transition-all"
                  >
                    Reset
                  </Button>
                </div>

                <Button
                  onClick={() => setShowHistory(true)}
                  className="w-full flex items-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white font-semibold transition-all"
                >
                  <History className="w-4 h-4" />
                  History ({summaryHistory.length})
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
              <CardContent className="space-y-4">
                {/* Text Length & Format - 2 Column Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >Text Length
                  </GradientText>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["short", "medium", "long"] as TextLength[]).map(len => (
                        <Button
                          key={len}
                          size="sm"
                          onClick={() => setTextLength(len)}
                          disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                          className={`capitalize text-xs px-2 font-semibold transition-all ${
                            textLength === len
                              ? "bg-gradient-to-r from-violet-600 to-blue-600 hover:from-violet-700 hover:to-blue-700 text-white"
                              : "bg-gradient-to-r from-violet-500/30 to-blue-500/30 hover:from-violet-500/50 hover:to-blue-500/50 text-white border border-violet-500/50"
                          }`}
                        >
                          {len}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >Format
                  </GradientText>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["bullet", "structured", "plain"] as SummaryFormat[]).map(fmt => (
                        <Button
                          key={fmt}
                          size="sm"
                          onClick={() => setSummaryFormat(fmt)}
                          disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                          className={`capitalize text-xs px-2 font-semibold transition-all ${
                            summaryFormat === fmt
                              ? "bg-gradient-to-r from-pink-600 to-violet-600 hover:from-pink-700 hover:to-violet-700 text-white"
                              : "bg-gradient-to-r from-pink-500/30 to-violet-500/30 hover:from-pink-500/50 hover:to-violet-500/50 text-white border border-pink-500/50"
                          }`}
                        >
                          {fmt}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Content Type */}
                <div>
                  <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >
                    Content Type
                  </GradientText>
                  <div className="grid grid-cols-2 gap-2">
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      summaryType === "balanced"
                        ? "bg-gradient-to-r from-violet-600 to-blue-600 text-white"
                        : "bg-gradient-to-r from-violet-500/20 to-blue-500/20 text-white hover:from-violet-500/40 hover:to-blue-500/40 border border-violet-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="balanced"
                        checked={summaryType === "balanced"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-violet-400"
                      />
                      <span>Balanced</span>
                    </label>
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      summaryType === "visual"
                        ? "bg-gradient-to-r from-pink-600 to-violet-600 text-white"
                        : "bg-gradient-to-r from-pink-500/20 to-violet-500/20 text-white hover:from-pink-500/40 hover:to-violet-500/40 border border-pink-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="visual"
                        checked={summaryType === "visual"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-pink-400"
                      />
                      <span>Visual Priority</span>
                    </label>
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      summaryType === "audio"
                        ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white"
                        : "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-white hover:from-blue-500/40 hover:to-cyan-500/40 border border-blue-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="audio"
                        checked={summaryType === "audio"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-blue-400"
                      />
                      <span>Audio Priority</span>
                    </label>
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      summaryType === "highlight"
                        ? "bg-gradient-to-r from-violet-600 via-pink-600 to-blue-600 text-white"
                        : "bg-gradient-to-r from-violet-500/20 via-pink-500/20 to-blue-500/20 text-white hover:from-violet-500/40 hover:via-pink-500/40 hover:to-blue-500/40 border border-pink-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="highlight"
                        checked={summaryType === "highlight"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-pink-400"
                      />
                      <span>Highlight</span>
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
                {/* Merged Video Player - Top Section */}
                {state.videoUrl && (
                  <Card className="flex-1 bg-white/10 border-white/20 backdrop-blur-sm flex flex-col min-h-[45%]">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">
                        <GradientText
                          colors={['#ec4899', '#f97316', '#ec4899', '#f97316', '#ec4899']}
                          animationSpeed={4}
                        >
                          Important Shots Compilation
                        </GradientText>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 flex items-center justify-center overflow-hidden relative">
                      <video
                        key={state.videoUrl}
                        controls
                        preload="metadata"
                        playsInline
                        className="w-full h-full object-contain rounded-lg"
                        style={{ maxHeight: "100%" }}
                        onError={(e) => {
                          console.error("[Video] Failed to load:", state.videoUrl);
                          console.error("[Video] Error details:", e);
                        }}
                        onLoadedMetadata={() => {
                          console.log("[Video] Loaded successfully:", state.videoUrl);
                        }}
                      >
                        <source src={`${state.videoUrl}?t=${Date.now()}`} type="video/mp4" />
                        Your browser does not support the video tag.
                      </video>
                      

                    </CardContent>
                  </Card>
                )}

                {/* Text Summary - Bottom Section */}
                {textSummary && (
                  <Card className={`${state.videoUrl ? 'flex-1 min-h-[50%]' : 'flex-1'} bg-white/10 border-white/20 backdrop-blur-sm flex flex-col overflow-hidden`}>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">
                        <GradientText
                          colors={['#10b981', '#06b6d4', '#3b82f6', '#06b6d4', '#10b981']}
                          animationSpeed={4}
                        >
                          Text Summary ({processingConfig?.format || summaryFormat} • {processingConfig?.length || textLength})
                        </GradientText>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 overflow-y-auto mb-4">
                      <div className="bg-white/5 rounded-lg p-4 prose prose-sm max-w-none">
                        <pre className="whitespace-pre-wrap font-sans text-sm text-white">
                          {textSummary}
                        </pre>
                      </div>
                    </CardContent>
                    
                    {/* Action Buttons - Compact Layout */}
                    <div className="px-6 pb-4 border-t border-white/10 pt-4">
                      <div className="grid grid-cols-5 gap-2 items-end">
                        {/* Download TXT Button */}
                        <Button
                          onClick={handleDownloadText}
                          className="flex items-center justify-center gap-1 bg-gradient-to-r from-violet-600 to-blue-600 hover:from-violet-700 hover:to-blue-700 text-white font-semibold transition-all text-xs py-2"
                          title="Download as TXT"
                        >
                          <Download className="w-3 h-3" />
                          TXT
                        </Button>

                        {/* Download JSON Button */}
                        <Button
                          onClick={handleDownloadJSON}
                          className="flex items-center justify-center gap-1 bg-gradient-to-r from-pink-600 to-violet-600 hover:from-pink-700 hover:to-violet-700 text-white font-semibold transition-all text-xs py-2"
                          title="Download as JSON"
                        >
                          <Download className="w-3 h-3" />
                          JSON
                        </Button>

                        {/* Voice Selection Dropdown */}
                        {availableVoices.length > 0 && (
                          <div className="col-span-2">
                            <select
                              value={selectedVoice?.name || ''}
                              onChange={(e) => {
                                const voice = availableVoices.find(v => v.name === e.target.value);
                                setSelectedVoice(voice || null);
                              }}
                              className="w-full bg-white/10 border border-white/20 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-violet-500 transition-colors"
                              title="Select voice for text-to-speech"
                            >
                              {availableVoices.map((voice) => (
                                  <option 
                                    key={voice.name} 
                                    value={voice.name}
                                    className="bg-gray-900 text-white"
                                  >
                                    {voice.name} {voice.localService ? '🌐' : '☁️'}
                                  </option>
                                ))}
                            </select>
                          </div>
                        )}

                        {/* Read Aloud Button */}
                        <Button
                          onClick={handleReadAloud}
                          className={`flex items-center justify-center gap-1 py-2 font-semibold transition-all duration-200 text-xs ${
                            isReading
                              ? "bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700"
                              : "bg-gradient-to-r from-violet-600 via-pink-500 to-blue-600 hover:from-violet-700 hover:via-pink-600 hover:to-blue-700"
                          } text-white`}
                          title={isReading ? "Stop reading" : "Read aloud with text-to-speech"}
                        >
                          {isReading ? (
                            <>
                              <StopCircle className="w-3 h-3" />
                              Stop
                            </>
                          ) : (
                            <>
                              <Volume2 className="w-3 h-3" />
                              Read
                            </>
                          )}
                        </Button>
                      </div>

                      {/* Reading Status Indicator */}
                      {isReading && (
                        <div className="text-center text-xs text-emerald-400 animate-pulse font-semibold mt-2">
                          🔊 Reading aloud...
                        </div>
                      )}
                    </div>
                  </Card>
                )}
              </>
            ) : (
              <Card className="flex-1 flex items-center justify-center bg-white/10 border-white/20 backdrop-blur-sm">
                <CardContent>
                  <p className="text-white text-center">
                    {state.status === "idle" 
                      ? "Upload a video to start" 
                      : "Processing... video compilation and summary will appear here"}
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

      {/* History Modal */}
      {showHistory && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <Card className="w-full max-w-2xl max-h-[80vh] bg-slate-950 border-white/20 backdrop-blur-sm flex flex-col">
            <CardHeader className="flex items-center justify-between border-b border-white/10 pb-3">
              <CardTitle className="flex items-center gap-2 text-lg">
                <History className="w-5 h-5 text-cyan-400" />
                <GradientText
                  colors={['#06b6d4', '#0ea5e9', '#06b6d4', '#0ea5e9', '#06b6d4']}
                  animationSpeed={4}
                >
                  Summary History
                </GradientText>
              </CardTitle>
              <Button
                onClick={() => setShowHistory(false)}
                variant="ghost"
                size="sm"
                className="hover:bg-white/10"
              >
                <X className="w-4 h-4" />
              </Button>
            </CardHeader>

            <CardContent className="flex-1 overflow-hidden flex gap-4 p-4">
              {/* History List */}
              <div className="w-1/3 border-r border-white/10 overflow-y-auto pr-4 space-y-2">
                {summaryHistory.length === 0 ? (
                  <p className="text-xs text-slate-400 text-center py-8">No history yet</p>
                ) : (
                  summaryHistory.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => setSelectedHistoryItem(item)}
                      className={`w-full text-left p-2 rounded-lg transition-all text-xs ${
                        selectedHistoryItem?.id === item.id
                          ? "bg-gradient-to-r from-cyan-600/50 to-blue-600/50 border border-cyan-400/50"
                          : "bg-white/5 hover:bg-white/10 border border-white/10"
                      }`}
                    >
                      <p className="font-medium text-white truncate">{item.title}</p>
                      <p className="text-slate-400 text-xs">
                        {item.format} • {item.length}
                      </p>
                      <p className="text-slate-500 text-xs">
                        {new Date(item.timestamp).toLocaleDateString()} {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </button>
                  ))
                )}
              </div>

              {/* Preview Section */}
              <div className="flex-1 overflow-y-auto flex flex-col gap-3">
                {selectedHistoryItem ? (
                  <>
                    <div className="space-y-2 pb-3 border-b border-white/10">
                      <h3 className="font-semibold text-white">{selectedHistoryItem.title}</h3>
                      <div className="flex flex-wrap gap-2 text-xs">
                        <span className="bg-cyan-600/30 text-cyan-300 px-2 py-1 rounded">
                          {selectedHistoryItem.format}
                        </span>
                        <span className="bg-blue-600/30 text-blue-300 px-2 py-1 rounded">
                          {selectedHistoryItem.length}
                        </span>
                        <span className="bg-purple-600/30 text-purple-300 px-2 py-1 rounded capitalize">
                          {selectedHistoryItem.type}
                        </span>
                      </div>
                      <p className="text-slate-400 text-xs">
                        {new Date(selectedHistoryItem.timestamp).toLocaleString()}
                      </p>
                    </div>

                    {/* Summary Text */}
                    <div className="flex-1 overflow-y-auto">
                      <pre className="whitespace-pre-wrap font-sans text-xs text-slate-200 bg-white/5 p-3 rounded-lg">
                        {selectedHistoryItem.summary}
                      </pre>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-2 pt-3 border-t border-white/10">
                      <Button
                        onClick={() => {
                          const element = document.createElement("a");
                          const file = new Blob([selectedHistoryItem.summary], { type: "text/plain" });
                          element.href = URL.createObjectURL(file);
                          element.download = `summary_${selectedHistoryItem.title}_${selectedHistoryItem.format}.txt`;
                          document.body.appendChild(element);
                          element.click();
                          document.body.removeChild(element);
                        }}
                        className="flex-1 flex items-center gap-2 bg-gradient-to-r from-violet-600 to-blue-600 hover:from-violet-700 hover:to-blue-700 text-white text-xs font-semibold py-2"
                      >
                        <Download className="w-3 h-3" />
                        Download
                      </Button>
                      <Button
                        onClick={() => deleteHistoryItem(selectedHistoryItem.id)}
                        className="flex items-center gap-2 bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white text-xs font-semibold py-2 px-3"
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-400">
                    <p className="text-center text-xs">Select a history item to preview</p>
                  </div>
                )}
              </div>
            </CardContent>

            {/* Footer Actions */}
            <div className="border-t border-white/10 p-4 flex gap-2 justify-end">
              <Button
                onClick={clearHistory}
                disabled={summaryHistory.length === 0}
                className="bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 text-white text-xs font-semibold disabled:opacity-50"
              >
                Clear All
              </Button>
              <Button
                onClick={() => setShowHistory(false)}
                className="bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white text-xs font-semibold"
              >
                Close
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
