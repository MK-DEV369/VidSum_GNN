import { Link } from "react-router-dom";
import Mahantesh from "../lib/Mahantesh.jpeg";
import Morya from "../lib/Morya.jpg";
import { 
  Upload, 
  Video, 
  Scissors, 
  Brain, 
  Zap,
  GitGraph,
  Play,
  ArrowRight 
} from "lucide-react";
import { Button } from "../components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/card";
import FloatingLines from "../components/FloatingLines";
import SpotlightCard from "../components/SpotlightCard";
import GradientText from "../components/GradientText";
import ChromaGrid from "../components/ChromaGrid";
import "./chroma-grid-override.css";

export default function HomePage() {
  const features = [
    {
      icon: <GitGraph className="w-10 h-10 text-white" />,
      title: "Graph Attention Networks",
      description: "Advanced GNN architecture analyzes complex relationships between video scenes using temporal, semantic, and audio edges for intelligent importance scoring."
    },
    {
      icon: <Brain className="w-10 h-10 text-white" />,
      title: "Multimodal Deep Learning",
      description: "Integrates Vision Transformers (ViT), HuBert audio processing (Wav2Vec2), and temporal analysis for comprehensive multimodal scene understanding."
    },
    {
      icon: <Zap className="w-10 h-10 text-white" />,
      title: "Enterprise-Scale Processing",
      description: "Handles long-form videos with intelligent chunking, automatic memory management, GPU acceleration, and real-time progress tracking."
    },
    {
      icon: <Scissors className="w-10 h-10 text-white" />,
      title: "Smart Summary Generation",
      description: "Flexible Content Types (Balanced/Visual/Audio/Highlights) for generating summaries with customizable length, format (bullet/structured/plain), and content priority."
    }
  ];

  const workflow = [
    { step: 1, icon: <Upload className="w-6 h-6 text-white" />, label: "Upload & Transcode", desc: "Process video file" },
    { step: 2, icon: <Video className="w-6 h-6 text-white" />, label: "Scene Detection", desc: "Identify shot boundaries" },
    { step: 3, icon: <Brain className="w-6 h-6 text-white" />, label: "Feature Extraction", desc: "ViT + Wav2Vec2 analysis" },
    { step: 4, icon: <GitGraph className="w-6 h-6 text-white" />, label: "Graph Construction", desc: "Build temporal-semantic graph" },
    { step: 5, icon: <Zap className="w-6 h-6 text-white" />, label: "GNN Inference", desc: "GAT importance scoring" },
    { step: 6, icon: <Play className="w-6 h-6 text-white" />, label: "Summary Generation", desc: "Assemble & export" }
  ];

  const techStack = [
    { category: "Frontend", items: ["React 18", "TypeScript", "Vite 5", "TailwindCSS", "shadcn/ui", "WebSocket"] },
    { category: "Backend", items: ["FastAPI", "PyTorch 2.1", "PyTorch Geometric", "SQLAlchemy", "Asyncio"] },
    { category: "AI Models", items: ["ViT-B/16", "HuBert-Base", "GAT", "Scene Detection", "FLAN-T5"] },
    { category: "Infrastructure", items: ["Docker", "PostgreSQL", "FFmpeg", "SceneDetect", "CUDA"] }
  ];

  return (
  <>
      <style>{`
        .chroma-grid-team img {
          width: 300px !important;
          height: 300px !important;
          object-fit: cover !important;
          object-position: center !important;
          border-radius: 16px !important;
          background: #222;
          overflow: hidden;
        }
      `}</style>
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950" style={{ position: 'relative' }}>
      {/* Animated Background */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0 }}>
        <FloatingLines
          linesGradient={['#4338ca', '#6366f1', '#8b5cf6', '#a855f7', '#c084fc']}
          enabledWaves={['middle', 'bottom']}
          lineCount={[8, 6]}
          lineDistance={[6, 5]}
          animationSpeed={1}
          interactive={true}
          parallax={true}
          mixBlendMode="screen"
        />
      </div>

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1 }}>
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto mb-20">
          <h1 className="text-6xl font-bold mb-6 drop-shadow-lg">
            <GradientText
              colors={['#4338ca', '#6366f1', '#8b5cf6', '#a855f7', '#c084fc', '#4338ca']}
              animationSpeed={5}
              showBorder={false}
            >
              VIDSUM-GNN
            </GradientText>
          </h1>
          <p className="text-2xl mb-4 drop-shadow-md">
            <GradientText
              colors={['#3b82f6', '#06b6d4', '#10b981', '#06b6d4', '#3b82f6']}
              animationSpeed={6}
              showBorder={false}
            >
              Intelligent Video Summarization with Graph Neural Networks
            </GradientText>
          </p>
          <p className="text-lg text-slate-300 mb-8">
            An advanced AI-powered platform that automatically generates concise, context-aware video summaries 
            by analyzing visual, audio, and temporal relationships using Graph Attention Networks and multimodal deep learning.
          </p>
          <div className="flex gap-4 justify-center">
            <Link to="/dashboard">
              <Button size="lg" className="gap-2 bg-gradient-to-r from-violet-600 via-pink-500 to-blue-600 hover:from-violet-700 hover:via-pink-600 hover:to-blue-700 text-white font-semibold transition-all">
                Try Dashboard <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
          </div>
        </div>

        {/* Workflow Section */}
        <Card className="mb-16 bg-white/10 border-white/20 backdrop-blur-sm">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl">
              <GradientText
                colors={['#6366f1', '#8b5cf6', '#a855f7', '#8b5cf6', '#6366f1']}
                animationSpeed={5}
              >
                Processing Pipeline
              </GradientText>
            </CardTitle>
            <CardDescription className="text-white">
              Six-stage multimodal processing with real-time progress tracking
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {workflow.map((item, idx) => (
                <SpotlightCard key={idx} spotlightColor="rgba(99, 102, 241, 0.2)">
                  <div className="flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                      {item.icon}
                    </div>
                    <div className="text-xs font-semibold text-primary mb-1">Step {item.step}</div>
                    <div className="font-medium mb-1 text-white">{item.label}</div>
                    <div className="text-xs text-white">{item.desc}</div>
                  </div>
                </SpotlightCard>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Key Features Section */}
        <Card className="mb-16 bg-white/10 border-white/20 backdrop-blur-sm">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl">
              <GradientText
                colors={['#8b5cf6', '#a855f7', '#c084fc', '#a855f7', '#8b5cf6']}
                animationSpeed={5}
              >
                Key Features
              </GradientText>
            </CardTitle>
            <CardDescription className="text-white">
              Advanced AI capabilities for intelligent video summarization
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              {features.map((feature, idx) => (
                <SpotlightCard key={idx} spotlightColor="rgba(139, 92, 246, 0.2)">
                  <div className="mb-4">{feature.icon}</div>
                  <h3 className="text-xl font-semibold mb-3 text-white">{feature.title}</h3>
                  <p className="text-white">{feature.description}</p>
                </SpotlightCard>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Technology Stack Section */}
        <Card className="mb-16 bg-white/10 border-white/20 backdrop-blur-sm">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl">
              <GradientText
                colors={['#3b82f6', '#06b6d4', '#0ea5e9', '#06b6d4', '#3b82f6']}
                animationSpeed={5}
              >
                Technology Stack
              </GradientText>
            </CardTitle>
            <CardDescription className="text-white">
              State-of-the-art frameworks and libraries powering VIDSUM-GNN
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {techStack.map((stack, idx) => (
                <SpotlightCard key={idx} spotlightColor="rgba(59, 130, 246, 0.2)">
                  <h3 className="font-semibold mb-3 text-white">{stack.category}</h3>
                  <div className="flex flex-wrap gap-2">
                    {stack.items.map((item, i) => (
                      <span 
                        key={i}
                        className="px-3 py-1 bg-secondary rounded-full text-xs font-medium"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </SpotlightCard>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Team Section */}
        <Card className="bg-white/10 border-white/20 backdrop-blur-sm">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl">
              <GradientText
                colors={['#10b981', '#06b6d4', '#3b82f6', '#06b6d4', '#10b981']}
                animationSpeed={5}
              >
                Project Team
              </GradientText>
            </CardTitle>
            <CardDescription className="text-white">
              AI253IA - Artificial Neural Networks and Deep Learning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div style={{ minHeight: '400px', position: 'relative' }}>
              <div className="chroma-grid-team">
              <ChromaGrid
                items={[
                  {
                    image: Morya,
                    title: 'L Moryakantha',
                    subtitle: 'Backend & GNN Architecture',
                    handle: '1RV24AI406',
                    borderColor: '#4F46E5',
                    gradient: 'linear-gradient(145deg, #4F46E5, #000)'
                  },
                  {
                    image: Mahantesh,
                    title: 'Mahantesh PB',
                    subtitle: 'Frontend & UI Development',
                    handle: '1RV24AI407',
                    borderColor: '#10B981',
                    gradient: 'linear-gradient(210deg, #10B981, #000)'
                  }
                ]}
                columns={2}
                rows={1}
                radius={250}
                damping={0.5}
                fadeOut={0.7}
              />
              </div>
            </div>
            
          </CardContent>
        </Card>
      </div>
      </div>
    </div>
    </>
  );  
}
