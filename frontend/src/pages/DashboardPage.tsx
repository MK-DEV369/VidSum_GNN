import { useState, useEffect, useRef } from "react";
import { Upload, AlertCircle, CheckCircle, Clock, Download, Volume2, StopCircle, History, X, Trash2, ChevronDown, ChevronUp } from "lucide-react";
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
  type: "balanced" | "visual_priority" | "audio_priority" | "highlights";
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
  processingStartedAt?: string;
  processingCompletedAt?: string;
  processingDurationSec?: number;
}

type TtsVoice = {
  short_name: string;
  friendly_name?: string | null;
  locale?: string | null;
  gender?: string | null;
};

type TextLength = "short" | "medium" | "long";
type SummaryFormat = "bullet" | "structured" | "plain";
type SummaryType = "balanced" | "visual_priority" | "audio_priority" | "highlights";

type EvidenceItem = {
  index?: number | null;
  bullet?: string | null;
  shot_index?: number | null;
  shot_id?: string | null;
  orig_start?: number | null;
  orig_end?: number | null;
  merged_start?: number | null;
  merged_end?: number | null;
  score?: number | null;
  transcript_snippet?: string | null;
  thumbnail_url?: string | null;
  signals?: {
    motion?: number | null;
    audio_rms?: number | null;
    scene_change?: number | null;
    transcript_density?: number | null;
    duration_sec?: number | null;
  } | null;
  neighbors?: Array<{
    neighbor_index: number;
    edge_type: "temporal" | "semantic";
    similarity?: number | null;
    distance_sec?: number | null;
  }> | null;
};

type ChapterItem = {
  index: number;
  title: string;
  merged_start: number;
  merged_end: number;
  shot_indices?: number[];
};

const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [textLength, setTextLength] = useState<TextLength>("medium");
  const [summaryFormat, setSummaryFormat] = useState<SummaryFormat>("bullet");
  const [summaryType, setSummaryType] = useState<SummaryType>("balanced");
  const [textSummary, setTextSummary] = useState<string | null>(null);
  const [evidenceItems, setEvidenceItems] = useState<EvidenceItem[]>([]);
  const [chapters, setChapters] = useState<ChapterItem[]>([]);
  const [showChaptersPanel, setShowChaptersPanel] = useState<boolean>(true);
  const [showEvidencePanel, setShowEvidencePanel] = useState<boolean>(true);
  const [expandedEvidenceKey, setExpandedEvidenceKey] = useState<string | null>(null);
  const [state, setState] = useState<ProcessingState>({
    status: "idle",
    progress: 0,
    currentStage: "Waiting for upload...",
    logs: []
  });
  const [videoId, setVideoId] = useState<string>("");
  const [processingConfig, setProcessingConfig] = useState<{format: SummaryFormat, length: TextLength, type: SummaryType} | null>(null);
  const [logsExpanded, setLogsExpanded] = useState<boolean>(false);
  const [isReading, setIsReading] = useState<boolean>(false);
  const [ttsVoices, setTtsVoices] = useState<TtsVoice[]>([]);
  const [selectedTtsVoice, setSelectedTtsVoice] = useState<string>("");
  const [ttsRate, setTtsRate] = useState<number>(1.0);
  const wsRef = useRef<WebSocket | null>(null);
  const didHandleCompleteForVideoRef = useRef<string | null>(null);
  const didFetchSummaryForVideoRef = useRef<string | null>(null);
  const didFetchEvidenceForVideoRef = useRef<string | null>(null);
  const didFetchChaptersForVideoRef = useRef<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
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

  // Load available AI TTS voices from backend (non-browser speech)
  useEffect(() => {
    const loadTtsVoices = async () => {
      try {
        const resp = await axios.get(`${API_BASE}/api/tts/voices`);
        const voices = (resp.data?.voices || []) as TtsVoice[];
        setTtsVoices(voices);
        if (!selectedTtsVoice && voices.length > 0) {
          setSelectedTtsVoice(voices[0].short_name);
        }
      } catch (e) {
        console.warn("[TTS] Failed to load voices", e);
        setTtsVoices([]);
      }
    };
    loadTtsVoices();
  }, []);

  // Connect to WebSocket for logs
  useEffect(() => {
    if (!videoId) return;

    // New video: reset per-video fetch guards
    didHandleCompleteForVideoRef.current = null;
    didFetchSummaryForVideoRef.current = null;
    didFetchEvidenceForVideoRef.current = null;
    didFetchChaptersForVideoRef.current = null;

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
        
        // Fetch result payloads when processing completes (once per videoId)
        if (isComplete) {
          if (didHandleCompleteForVideoRef.current !== videoId) {
            didHandleCompleteForVideoRef.current = videoId;

            didFetchSummaryForVideoRef.current = videoId;
            fetchTextSummary(videoId);

            if (showEvidencePanel) {
              didFetchEvidenceForVideoRef.current = videoId;
              fetchEvidence(videoId);
            }
            if (showChaptersPanel) {
              didFetchChaptersForVideoRef.current = videoId;
              fetchChapters(videoId);
            }
          }
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

  // If user enables panels after completion, lazy-load their data.
  useEffect(() => {
    if (!videoId) return;
    if (state.status !== "completed") return;
    if (!showEvidencePanel) return;
    if (didFetchEvidenceForVideoRef.current === videoId) return;
    if (evidenceItems.length > 0) {
      didFetchEvidenceForVideoRef.current = videoId;
      return;
    }
    didFetchEvidenceForVideoRef.current = videoId;
    fetchEvidence(videoId);
  }, [showEvidencePanel, videoId, state.status]);

  useEffect(() => {
    if (!videoId) return;
    if (state.status !== "completed") return;
    if (!showChaptersPanel) return;
    if (didFetchChaptersForVideoRef.current === videoId) return;
    if (chapters.length > 0) {
      didFetchChaptersForVideoRef.current = videoId;
      return;
    }
    didFetchChaptersForVideoRef.current = videoId;
    fetchChapters(videoId);
  }, [showChaptersPanel, videoId, state.status]);

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
        processingStartedAt: response.data.processing_started_at ?? prev.processingStartedAt,
        processingCompletedAt: response.data.processing_completed_at ?? prev.processingCompletedAt,
        processingDurationSec: typeof response.data.processing_duration_sec === "number" ? response.data.processing_duration_sec : prev.processingDurationSec,
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

  const formatTime = (seconds: number | null | undefined) => {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "--:--";
    const total = Math.max(0, Math.floor(seconds));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  const normalizeStageKey = (stage: string | null | undefined) => {
    const s = String(stage || "").toLowerCase();
    if (!s) return "idle";
    if (s.includes("upload")) return "upload";
    if (s.includes("queue")) return "queued";
    if (s.includes("preprocess")) return "preprocessing";
    if (s.includes("shot")) return "shot_detection";
    if (s.includes("feature")) return "feature_extraction";
    if (s.includes("gnn") || s.includes("inference") || s.includes("score")) return "gnn_inference";
    if (s.includes("merge")) return "video_merge";
    if (s.includes("assem") || s.includes("summ") || s.includes("format")) return "assembling";
    if (s.includes("complete") || s.includes("done")) return "completed";
    if (s.includes("fail") || s.includes("error")) return "failed";
    return s;
  };

  const stageLabel = (stage: string | null | undefined) => {
    const key = normalizeStageKey(stage);
    const map: Record<string, string> = {
      idle: "Waiting",
      upload: "Uploading",
      queued: "Queued",
      preprocessing: "Preparing",
      shot_detection: "Detecting Shots",
      feature_extraction: "Extracting",
      gnn_inference: "Scoring",
      video_merge: "Merging",
      assembling: "Summarizing",
      completed: "Completed",
      failed: "Failed",
    };
    return map[key] || String(stage || "Processing");
  };

  const formatChapterTitle = (rawTitle: string | null | undefined, chapterIndex: number) => {
    const t = String(rawTitle || "").trim();
    if (!t) return `Chapter ${chapterIndex + 1}`;

    // Normalize older/odd formats like: "word . word . word" or "word | word | word"
    let normalized = t
      .replace(/\s+\.\s+/g, " · ")
      .replace(/\s+\|\s+/g, " · ")
      .replace(/\s+•\s+/g, " · ")
      .replace(/\s*·\s*/g, " · ");

    const parts = normalized
      .split(" · ")
      .map(p => p.trim())
      .filter(Boolean);

    if (parts.length >= 3) return parts.slice(0, 3).join(" · ");
    return normalized;
  };

  const stageBarClass = (stage: string | null | undefined) => {
    const key = normalizeStageKey(stage);
    if (key === "failed") return "bg-gradient-to-r from-red-600 via-pink-600 to-red-600";
    if (key === "completed") return "bg-gradient-to-r from-emerald-500 via-cyan-500 to-emerald-500";
    if (key === "upload") return "bg-gradient-to-r from-violet-600 via-pink-500 to-blue-600";
    if (key === "shot_detection") return "bg-gradient-to-r from-amber-500 via-orange-500 to-pink-500";
    if (key === "feature_extraction") return "bg-gradient-to-r from-cyan-500 via-blue-500 to-violet-500";
    if (key === "gnn_inference") return "bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600";
    if (key === "video_merge") return "bg-gradient-to-r from-fuchsia-600 via-pink-600 to-amber-500";
    return "bg-gradient-to-r from-violet-600 via-pink-500 to-blue-600";
  };

  const fetchEvidence = async (videoId: string) => {
    try {
      const response = await axios.get(`${API_BASE}/api/summary/${videoId}/evidence`);
      const items = (response.data?.items || []) as EvidenceItem[];
      setEvidenceItems(items);

      didFetchEvidenceForVideoRef.current = videoId;

      if (items.length > 0) {
        setState(prev => ({
          ...prev,
          logs: [...prev.logs, {
            timestamp: new Date().toISOString(),
            level: "SUCCESS",
            message: `Evidence-linked highlights loaded (${items.length})`
          }]
        }));
      }
    } catch (error: any) {
      console.error("[Fetch Evidence] Error:", error);
      setEvidenceItems([]);
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "WARNING",
          message: "Could not fetch evidence-linked highlights"
        }]
      }));
    }
  };

  const fetchChapters = async (videoId: string) => {
    try {
      const response = await axios.get(`${API_BASE}/api/summary/${videoId}/chapters`);
      const items = (response.data?.items || []) as ChapterItem[];
      setChapters(items);

      didFetchChaptersForVideoRef.current = videoId;

      if (items.length > 0) {
        setState(prev => ({
          ...prev,
          logs: [...prev.logs, {
            timestamp: new Date().toISOString(),
            level: "SUCCESS",
            message: `Chapters generated (${items.length})`
          }]
        }));
      }
    } catch (error: any) {
      console.error("[Fetch Chapters] Error:", error);
      setChapters([]);
      setState(prev => ({
        ...prev,
        logs: [...prev.logs, {
          timestamp: new Date().toISOString(),
          level: "WARNING",
          message: "Could not fetch chapters"
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
    setEvidenceItems([]);
    setChapters([]);
    setExpandedEvidenceKey(null);
    setState({ status: "idle", progress: 0, currentStage: "Waiting for upload...", logs: [] });
    setVideoId("");
    setProcessingConfig(null);
    didHandleCompleteForVideoRef.current = null;
    didFetchSummaryForVideoRef.current = null;
    didFetchEvidenceForVideoRef.current = null;
    didFetchChaptersForVideoRef.current = null;
    setIsReading(false);
    try {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    } catch {}
    try { if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) wsRef.current.close(); } catch {}
  };

  const getSignalStats = () => {
    const keys = ["motion", "audio_rms", "scene_change", "transcript_density"] as const;
    const stats: Record<string, { min: number; max: number }> = {};

    for (const k of keys) {
      const vals = evidenceItems
        .map(it => it.signals?.[k])
        .filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
      const min = vals.length ? Math.min(...vals) : 0;
      const max = vals.length ? Math.max(...vals) : 1;
      stats[k] = { min, max };
    }
    return stats;
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

  const formatEvidenceExport = (items: EvidenceItem[]) => {
    const generatedAt = new Date().toISOString();
    const lines: string[] = [];

    lines.push("Evidence-linked highlights export");
    if (videoId) lines.push(`Video ID: ${videoId}`);
    lines.push(`Generated at: ${generatedAt}`);
    lines.push("");

    const num = (v: number | null | undefined, digits = 3) =>
      typeof v === "number" && !Number.isNaN(v) ? v.toFixed(digits) : "—";

    const signalsKeys: Array<keyof NonNullable<EvidenceItem["signals"]>> = [
      "motion",
      "audio_rms",
      "scene_change",
      "transcript_density",
      "duration_sec",
    ];

    items.forEach((item, idx) => {
      const title = item.bullet || `Shot ${item.shot_index ?? idx}`;
      const mergedStart = typeof item.merged_start === "number" ? item.merged_start : null;
      const mergedEnd = typeof item.merged_end === "number" ? item.merged_end : null;
      const origStart = typeof item.orig_start === "number" ? item.orig_start : null;
      const origEnd = typeof item.orig_end === "number" ? item.orig_end : null;

      lines.push(`### ${idx + 1}. ${title}`);
      lines.push(`shot_index: ${item.shot_index ?? "—"}`);
      if (item.shot_id) lines.push(`shot_id: ${item.shot_id}`);
      lines.push(`score: ${num(item.score, 6)}`);
      lines.push(
        `merged: ${mergedStart != null ? formatTime(mergedStart) : "—"} – ${mergedEnd != null ? formatTime(mergedEnd) : "—"} (sec: ${mergedStart ?? "—"} – ${mergedEnd ?? "—"})`
      );
      if (origStart != null || origEnd != null) {
        lines.push(
          `original: ${origStart != null ? formatTime(origStart) : "—"} – ${origEnd != null ? formatTime(origEnd) : "—"} (sec: ${origStart ?? "—"} – ${origEnd ?? "—"})`
        );
      }

      const sig = item.signals || null;
      if (sig) {
        const sigParts = signalsKeys.map((k) => `${String(k)}=${num(sig[k] as number | null | undefined, 6)}`);
        lines.push(`signals: ${sigParts.join(", ")}`);
      } else {
        lines.push("signals: —");
      }

      if (item.transcript_snippet) {
        lines.push(`transcript_snippet: ${item.transcript_snippet.replace(/\s+/g, " ").trim()}`);
      }

      if (Array.isArray(item.neighbors) && item.neighbors.length > 0) {
        lines.push("neighbors:");
        item.neighbors.slice(0, 12).forEach((n) => {
          lines.push(
            `  - idx=${n.neighbor_index}, edge=${n.edge_type}, similarity=${num(n.similarity, 6)}, distance_sec=${num(n.distance_sec, 3)}`
          );
        });
      }

      lines.push("");
    });

    // TSV section for easy copy/paste graphing.
    lines.push("---");
    lines.push("Signals (TSV)");
    lines.push("index\tstart_sec\tend_sec\tscore\tmotion\taudio_rms\tscene_change\ttranscript_density\tduration_sec");
    items.forEach((item, idx) => {
      const sig = item.signals || {};
      const start = typeof item.merged_start === "number" ? item.merged_start : "";
      const end = typeof item.merged_end === "number" ? item.merged_end : "";
      const row = [
        String(idx + 1),
        String(start),
        String(end),
        typeof item.score === "number" ? String(item.score) : "",
        typeof sig.motion === "number" ? String(sig.motion) : "",
        typeof sig.audio_rms === "number" ? String(sig.audio_rms) : "",
        typeof sig.scene_change === "number" ? String(sig.scene_change) : "",
        typeof sig.transcript_density === "number" ? String(sig.transcript_density) : "",
        typeof sig.duration_sec === "number" ? String(sig.duration_sec) : "",
      ].join("\t");
      lines.push(row);
    });

    return lines.join("\n");
  };

  const handleDownloadEvidenceText = () => {
    if (!videoId) return;
    if (!evidenceItems || evidenceItems.length === 0) return;
    const content = formatEvidenceExport(evidenceItems);
    const element = document.createElement("a");
    const file = new Blob([content], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `evidence_${videoId}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleReadAloud = () => {
    if (!textSummary) return;

    if (isReading) {
      try {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        }
      } catch {}
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

    const voice = selectedTtsVoice || ttsVoices[0]?.short_name;
    if (!voice) {
      alert("No AI voices available (TTS backend may be missing)");
      return;
    }

    setIsReading(true);
    axios
      .post(`${API_BASE}/api/tts`, {
        text: cleanText,
        voice,
        rate: ttsRate,
        video_id: videoId || undefined
      })
      .then((resp) => {
        const audioUrl = resp.data?.audio_url;
        if (!audioUrl) throw new Error("No audio_url returned");

        const fullUrl = `${API_BASE}${audioUrl}`;
        const audio = new Audio(fullUrl);
        audioRef.current = audio;
        audio.onended = () => setIsReading(false);
        audio.onerror = () => {
          console.error("[TTS] Audio playback error");
          setIsReading(false);
        };
        audio.play().catch((e) => {
          console.error("[TTS] Play failed", e);
          setIsReading(false);
        });
      })
      .catch((e) => {
        console.error("[TTS] Generation failed", e);
        setIsReading(false);
        alert("TTS failed (backend missing or network issue)");
      });
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

  const signalStats = getSignalStats();
  const normSignal = (key: keyof NonNullable<EvidenceItem["signals"]>, v: number | null | undefined) => {
    if (typeof v !== "number" || Number.isNaN(v)) return 0;
    const s = signalStats[key as string] || { min: 0, max: 1 };
    const denom = (s.max - s.min) || 1;
    return Math.max(0, Math.min(1, (v - s.min) / denom));
  };

  const showSidePanel = showChaptersPanel || showEvidencePanel;

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
                    className="w-full bg-gradient-to-r from-slate-700 via-slate-600 to-slate-700 hover:from-slate-600 hover:via-slate-500 hover:to-slate-600 text-white font-semibold transition-all border border-white/10"
                  >
                    {state.status === "uploading" ? "Uploading..." : "Upload & Process"}
                  </Button>
                  <Button
                    onClick={handleReset}
                    disabled={state.status === "processing"}
                    className="w-full bg-gradient-to-r from-slate-800 to-slate-700 hover:from-slate-700 hover:to-slate-600 text-white font-semibold transition-all border border-white/10"
                  >
                    Reset
                  </Button>
                </div>

                <Button
                  onClick={() => setShowHistory(true)}
                  className="w-full flex items-center gap-2 bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 hover:from-slate-700 hover:via-slate-600 hover:to-slate-700 text-white font-semibold transition-all border border-white/10"
                >
                  <History className="w-4 h-4" />
                  History ({summaryHistory.length})
                </Button>
              </CardContent>
            </Card>

            {/* Status */}
            <Card className="bg-white/10 border-white/20 backdrop-blur-sm">
              <CardHeader className="py-3">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 text-sm text-white shrink-0 w-28">
                    {getStatusIcon()}
                    <span className="capitalize font-semibold text-white">
                      {state.status === "idle"
                        ? "Idle"
                        : state.status === "uploading"
                          ? "Uploading"
                          : state.status === "processing"
                            ? "Processing"
                            : state.status === "completed"
                              ? "Completed"
                              : "Error"}
                    </span>
                  </div>

                  <div className="flex-1 h-3 rounded-full bg-white/10 overflow-hidden relative">
                    <div
                      className={`h-full rounded-full ${stageBarClass(state.currentStage)} transition-all duration-500 ease-out relative`}
                      style={{ width: `${Math.max(0, Math.min(100, state.progress))}%` }}
                    >
                      <div className="absolute inset-0 opacity-30 animate-pulse bg-white/20" />
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center px-2 pointer-events-none">
                      <span
                        className="text-[10px] font-semibold text-white/90 truncate"
                        title={String(state.currentStage || "") || undefined}
                      >
                        {stageLabel(state.currentStage)} · {Math.max(0, Math.min(100, state.progress))}%
                      </span>
                    </div>
                  </div>

                  <span className="text-xs text-white font-semibold tabular-nums w-10 text-right shrink-0">
                    {state.progress}%
                  </span>
                </div>

                {state.error && (
                  <div className="mt-2 bg-red-900/30 border border-red-500/50 rounded p-2">
                    <p className="text-xs text-red-400">{state.error}</p>
                  </div>
                )}
              </CardHeader>
            </Card>

            {/* Controls */}
            <Card className="bg-white/10 border-white/20 backdrop-blur-sm\">
              <CardContent className="space-y-4">
                {/* Text Length & Format - 2 Column Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="text-center">
                    <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >Text Length
                  </GradientText>
                    </div>
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
                    <div className="text-center">
                    <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >Format
                  </GradientText>
                    </div>
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
                  <div className="text-center">
                  <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >
                    Content Type
                  </GradientText>
                  </div>
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
                      summaryType === "visual_priority"
                        ? "bg-gradient-to-r from-pink-600 to-violet-600 text-white"
                        : "bg-gradient-to-r from-pink-500/20 to-violet-500/20 text-white hover:from-pink-500/40 hover:to-violet-500/40 border border-pink-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="visual_priority"
                        checked={summaryType === "visual_priority"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-pink-400"
                      />
                      <span>Visual Priority</span>
                    </label>
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      summaryType === "audio_priority"
                        ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white"
                        : "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-white hover:from-blue-500/40 hover:to-cyan-500/40 border border-blue-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="audio_priority"
                        checked={summaryType === "audio_priority"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-blue-400"
                      />
                      <span>Audio Priority</span>
                    </label>
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      summaryType === "highlights"
                        ? "bg-gradient-to-r from-violet-600 via-pink-600 to-blue-600 text-white"
                        : "bg-gradient-to-r from-violet-500/20 via-pink-500/20 to-blue-500/20 text-white hover:from-violet-500/40 hover:via-pink-500/40 hover:to-blue-500/40 border border-pink-500/30"
                    }`}>
                      <input 
                        type="radio"
                        name="summaryType"
                        value="highlights"
                        checked={summaryType === "highlights"}
                        onChange={(e) => setSummaryType(e.target.value as any)}
                        disabled={state.status === "processing" || state.status === "uploading" || state.status === "completed"}
                        className="accent-pink-400"
                      />
                      <span>Highlight</span>
                    </label>
                  </div>
                </div>

                {/* Panels */}
                <div>
                  <div className="text-center">
                  <GradientText
                    colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                    animationSpeed={4}
                  >
                    Panels
                  </GradientText>
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      showChaptersPanel
                        ? "bg-gradient-to-r from-emerald-600/60 to-cyan-600/60 text-white border border-emerald-400/40"
                        : "bg-white/5 text-white hover:bg-white/10 border border-white/10"
                    }`}>
                      <input
                        type="checkbox"
                        checked={showChaptersPanel}
                        onChange={(e) => setShowChaptersPanel(e.target.checked)}
                        className="accent-emerald-400"
                      />
                      <span>Chapters</span>
                    </label>
                    <label className={`flex items-center gap-2 cursor-pointer px-3 py-2 rounded-md transition-all font-medium text-xs ${
                      showEvidencePanel
                        ? "bg-gradient-to-r from-amber-600/60 to-red-600/60 text-white border border-amber-400/40"
                        : "bg-white/5 text-white hover:bg-white/10 border border-white/10"
                    }`}>
                      <input
                        type="checkbox"
                        checked={showEvidencePanel}
                        onChange={(e) => setShowEvidencePanel(e.target.checked)}
                        className="accent-amber-400"
                      />
                      <span>Evidence</span>
                    </label>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right: Output - 2/3 width */}
          <div className="col-span-2 overflow-hidden h-full flex flex-col">
            {state.status === "completed" || textSummary ? (
              <div className={`h-full grid gap-4 overflow-hidden ${showSidePanel ? "grid-cols-2" : "grid-cols-1"}`}>
                {/* Left: Intended summary output */}
                <div className="flex flex-col gap-4 overflow-hidden">
                  {/* Merged Video Player */}
                  {state.videoUrl && (
                    <Card className="bg-white/10 border-white/20 backdrop-blur-sm flex flex-col min-h-[45%]">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-base text-center">
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
                          ref={videoRef}
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

                  {/* Text Summary */}
                  {textSummary && (
                    <Card className={`${state.videoUrl ? 'flex-1 min-h-[50%]' : 'flex-1'} bg-white/10 border-white/20 backdrop-blur-sm flex flex-col overflow-hidden`}>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base text-center">
                          <GradientText
                            colors={['#10b981', '#06b6d4', '#3b82f6', '#06b6d4', '#10b981']}
                            animationSpeed={4}
                          >
                            Text Summary ({processingConfig?.format || summaryFormat} • {processingConfig?.length || textLength})
                          </GradientText>
                        </CardTitle>
                        {typeof state.processingDurationSec === "number" && (
                          <div className="mt-1 flex items-center gap-2 text-xs text-slate-300">
                            <Clock className="w-3 h-3" />
                            <span>Processed in {formatTime(state.processingDurationSec)}</span>
                          </div>
                        )}
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
                          className="flex items-center justify-center gap-1 bg-gradient-to-r from-slate-800 to-slate-700 hover:from-slate-700 hover:to-slate-600 text-white font-semibold transition-all text-xs py-2 border border-white/10"
                          title="Download as TXT"
                        >
                          <Download className="w-3 h-3" />
                          TXT
                        </Button>

                        {/* Download JSON Button */}
                        <Button
                          onClick={handleDownloadJSON}
                          className="flex items-center justify-center gap-1 bg-gradient-to-r from-slate-800 to-slate-700 hover:from-slate-700 hover:to-slate-600 text-white font-semibold transition-all text-xs py-2 border border-white/10"
                          title="Download as JSON"
                        >
                          <Download className="w-3 h-3" />
                          JSON
                        </Button>

                        {/* AI Voice + Speed (backend TTS) */}
                        {ttsVoices.length > 0 && (
                          <div className="col-span-2 grid grid-cols-2 gap-2">
                            <select
                              value={selectedTtsVoice || ttsVoices[0]?.short_name || ""}
                              onChange={(e) => setSelectedTtsVoice(e.target.value)}
                              className="w-full bg-white/10 border border-white/20 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-violet-500 transition-colors"
                              title="Select AI voice"
                            >
                              {ttsVoices.map((v) => (
                                <option key={v.short_name} value={v.short_name} className="bg-gray-900 text-white">
                                  {(v.friendly_name || v.short_name) + (v.locale ? ` (${v.locale})` : "")}
                                </option>
                              ))}
                            </select>

                            <select
                              value={String(ttsRate)}
                              onChange={(e) => setTtsRate(Number(e.target.value))}
                              className="w-full bg-white/10 border border-white/20 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-violet-500 transition-colors"
                              title="Speaking speed"
                            >
                              <option value="0.75" className="bg-gray-900 text-white">0.75×</option>
                              <option value="0.9" className="bg-gray-900 text-white">0.9×</option>
                              <option value="1" className="bg-gray-900 text-white">1.0×</option>
                              <option value="1.1" className="bg-gray-900 text-white">1.1×</option>
                              <option value="1.25" className="bg-gray-900 text-white">1.25×</option>
                            </select>
                          </div>
                        )}

                        {/* Read Aloud Button */}
                        <Button
                          onClick={handleReadAloud}
                          className={`flex items-center justify-center gap-1 py-2 font-semibold transition-all duration-200 text-xs ${
                            isReading
                              ? "bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700"
                              : "bg-gradient-to-r from-slate-700 via-slate-600 to-slate-700 hover:from-slate-600 hover:via-slate-500 hover:to-slate-600"
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
                </div>

                {/* Right: Chapters + evidence side panel */}
                {showSidePanel && (
                  <div className="flex flex-col gap-4 overflow-hidden">
                    {/* Chapters (YouTube-style TOC) */}
                    {showChaptersPanel && state.videoUrl && chapters.length > 0 && (
                      <Card className="bg-white/10 border-white/20 backdrop-blur-sm">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-base text-center">
                            <GradientText
                              colors={['#22c55e', '#06b6d4', '#22c55e', '#06b6d4', '#22c55e']}
                              animationSpeed={4}
                            >
                              Chapters
                            </GradientText>
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2 overflow-y-auto max-h-[35vh]">
                          {chapters.map((ch) => (
                            <button
                              key={`ch_${ch.index}`}
                              type="button"
                              onClick={() => {
                                const el = videoRef.current;
                                if (!el) return;
                                el.currentTime = ch.merged_start;
                                el.play().catch(() => undefined);
                                el.scrollIntoView({ behavior: "smooth", block: "center" });
                              }}
                              className="w-full text-left flex items-center justify-between gap-3 p-3 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-all"
                              title={`Jump to ${formatTime(ch.merged_start)}`}
                            >
                              <div className="min-w-0">
                                <p className="text-sm text-white font-semibold truncate" title={String(ch.title || "") || undefined}>
                                  {formatChapterTitle(ch.title, ch.index)}
                                </p>
                                <p className="text-xs text-slate-300">{formatTime(ch.merged_start)} – {formatTime(ch.merged_end)}</p>
                              </div>
                              <span className="text-xs text-slate-300 shrink-0">Chapter {ch.index + 1}</span>
                            </button>
                          ))}
                        </CardContent>
                      </Card>
                    )}

                    {/* Evidence-linked Highlights */}
                    {showEvidencePanel && state.videoUrl && evidenceItems.length > 0 && (
                      <Card className="bg-white/10 border-white/20 backdrop-blur-sm flex flex-col flex-1 min-h-0 overflow-hidden">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-base flex items-center justify-center gap-2">
                            <GradientText
                              colors={['#f59e0b', '#ef4444', '#f59e0b', '#ef4444', '#f59e0b']}
                              animationSpeed={4}
                            >
                              Evidence-linked Highlights
                            </GradientText>
                            <Button
                              onClick={handleDownloadEvidenceText}
                              className="flex items-center justify-center gap-1 bg-gradient-to-r from-slate-800 to-slate-700 hover:from-slate-700 hover:to-slate-600 text-white font-semibold transition-all text-xs py-1.5 px-2 border border-white/10"
                              title="Download evidence as text"
                            >
                              <Download className="w-3 h-3" />
                              TXT
                            </Button>
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 min-h-0 min-w-0 space-y-2 overflow-y-auto overflow-x-hidden overscroll-contain pr-1">
                          {evidenceItems.map((item: EvidenceItem, idx: number) => {
                            const canSeek = typeof item.merged_start === "number" && item.merged_start >= 0;
                            const thumbSrc = item.thumbnail_url ? `${API_BASE}${item.thumbnail_url}` : null;
                            const evKey = String(item.shot_id || item.shot_index || idx);
                            const expanded = expandedEvidenceKey === evKey;
                            return (
                              <div
                                key={`${item.shot_id || item.shot_index || idx}`}
                                className={`w-full text-left p-3 rounded-lg border transition-all overflow-hidden ${
                                  canSeek
                                    ? "bg-white/5 border-white/10"
                                    : "bg-white/5 border-white/5 opacity-60"
                                }`}
                              >
                                <div className="flex gap-3 min-w-0">
                                  <button
                                    type="button"
                                    disabled={!canSeek}
                                    onClick={() => {
                                      if (!canSeek) return;
                                      const el = videoRef.current;
                                      if (!el) return;
                                      el.currentTime = item.merged_start as number;
                                      el.play().catch(() => undefined);
                                      el.scrollIntoView({ behavior: "smooth", block: "center" });
                                    }}
                                    className={`flex-1 text-left flex gap-3 rounded-lg transition-all ${
                                      canSeek ? "hover:bg-white/5" : "cursor-not-allowed"
                                    }`}
                                    title={canSeek ? `Jump to ${formatTime(item.merged_start)}` : "No merged timestamp available"}
                                  >
                                    <div className="w-24 h-14 rounded-md overflow-hidden bg-black/40 shrink-0 flex items-center justify-center">
                                      {thumbSrc ? (
                                        <img
                                          src={thumbSrc}
                                          alt={`Shot ${item.shot_index ?? ""}`}
                                          className="w-full h-full object-cover"
                                          loading="lazy"
                                          onError={(e) => {
                                            (e.currentTarget as HTMLImageElement).style.display = "none";
                                          }}
                                        />
                                      ) : (
                                        <span className="text-xs text-slate-300">No thumbnail</span>
                                      )}
                                    </div>

                                    <div className="min-w-0 flex-1 overflow-hidden">
                                      <div className="flex items-center justify-between gap-2">
                                        <p className="text-sm text-white font-medium truncate">
                                          {item.bullet || `Shot ${item.shot_index ?? idx}`}
                                        </p>
                                        <span className="text-xs text-slate-300 shrink-0">
                                          {formatTime(item.merged_start)}
                                        </span>
                                      </div>
                                      <div className="flex items-center justify-between mt-1">
                                        <p className="text-xs text-slate-300">
                                          Importance: {typeof item.score === "number" ? item.score.toFixed(3) : "—"}
                                        </p>
                                      </div>
                                      {item.transcript_snippet && (
                                        <p className="text-xs text-slate-300 mt-1 line-clamp-2 break-all whitespace-normal">
                                          {item.transcript_snippet}
                                        </p>
                                      )}
                                    </div>
                                  </button>

                                  <button
                                    type="button"
                                    onClick={() => setExpandedEvidenceKey(prev => (prev === evKey ? null : evKey))}
                                    className="text-xs text-emerald-300 hover:text-emerald-200 font-semibold flex items-center gap-1 shrink-0 px-2"
                                    title="Why this shot was selected"
                                  >
                                    Why
                                    {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                  </button>
                                </div>

                                {expanded && (
                                  <div className="mt-3 pt-3 border-t border-white/10 grid grid-cols-2 gap-3 overflow-hidden min-w-0">
                                    <div className="space-y-2 min-w-0">
                                      <p className="text-xs text-white font-semibold">Top contributing signals</p>

                                      {([
                                        ["motion", item.signals?.motion],
                                        ["audio_rms", item.signals?.audio_rms],
                                        ["scene_change", item.signals?.scene_change],
                                        ["transcript_density", item.signals?.transcript_density],
                                      ] as Array<[keyof NonNullable<EvidenceItem["signals"]>, number | null | undefined]>).map(([k, v]) => (
                                        <div key={String(k)} className="space-y-1">
                                          <div className="flex justify-between text-xs text-slate-300">
                                            <span className="capitalize">{String(k).replace("_", " ")}</span>
                                            <span>{typeof v === "number" ? v.toFixed(3) : "—"}</span>
                                          </div>
                                          <Progress value={Math.round(normSignal(k, v) * 100)} className="h-2" />
                                        </div>
                                      ))}

                                      {typeof item.signals?.duration_sec === "number" && (
                                        <p className="text-xs text-slate-400">Shot duration: {item.signals.duration_sec.toFixed(2)}s</p>
                                      )}
                                    </div>

                                    <div className="space-y-2 min-w-0">
                                      <p className="text-xs text-white font-semibold">Graph neighbors</p>
                                      {item.neighbors && item.neighbors.length > 0 ? (
                                        <div className="space-y-1">
                                          {item.neighbors.slice(0, 8).map((n: NonNullable<EvidenceItem["neighbors"]>[number]) => (
                                            <div key={`${evKey}_${n.neighbor_index}_${n.edge_type}`} className="text-xs text-slate-300 flex items-center justify-between gap-2 min-w-0">
                                              <span className="min-w-0 flex-1 truncate">Shot {n.neighbor_index} ({n.edge_type})</span>
                                              <span className="shrink-0">
                                                {n.edge_type === "semantic"
                                                  ? `sim ${(n.similarity ?? 0).toFixed(2)}`
                                                  : `${Math.round(n.distance_sec ?? 0)}s`}
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      ) : (
                                        <p className="text-xs text-slate-400">No neighbor info</p>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </CardContent>
                      </Card>
                    )}
                  </div>
                )}
              </div>
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
                <span className="flex-1 text-center">
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
              <CardTitle className="flex items-center gap-2 text-lg w-full justify-center text-center">
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
