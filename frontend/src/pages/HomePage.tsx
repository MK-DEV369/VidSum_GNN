import { Link } from "react-router-dom";
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

export default function HomePage() {
  const features = [
    {
      icon: <GitGraph className="w-10 h-10 text-primary" />,
      title: "GNN-Based Importance Scoring",
      description: "Graph Attention Networks analyze temporal, semantic, and audio relationships between video scenes for intelligent scoring."
    },
    {
      icon: <Brain className="w-10 h-10 text-primary" />,
      title: "Multimodal Feature Extraction",
      description: "Combines ViT (visual), Wav2Vec2 (audio), and temporal features for comprehensive scene understanding."
    },
    {
      icon: <Zap className="w-10 h-10 text-primary" />,
      title: "Batch-wise Processing",
      description: "Handles long videos efficiently with 300s chunks, automatic memory management, and GPU optimization."
    },
    {
      icon: <Scissors className="w-10 h-10 text-primary" />,
      title: "Flexible Selection",
      description: "Choose greedy or knapsack algorithms to generate summaries matching your target duration constraints."
    }
  ];

  const workflow = [
    { step: 1, icon: <Upload />, label: "Upload Video", desc: "Submit your video file" },
    { step: 2, icon: <Video />, label: "Scene Detection", desc: "Identify key shots" },
    { step: 3, icon: <Brain />, label: "Feature Extraction", desc: "Extract multimodal features" },
    { step: 4, icon: <GitGraph />, label: "Graph Construction", desc: "Build scene graph" },
    { step: 5, icon: <Zap />, label: "GNN Inference", desc: "Calculate importance scores" },
    { step: 6, icon: <Play />, label: "Summary Assembly", desc: "Generate final video" }
  ];

  const techStack = [
    { category: "Frontend", items: ["React 18", "TypeScript", "Vite", "TailwindCSS", "shadcn/ui"] },
    { category: "Backend", items: ["FastAPI", "PyTorch 2.1", "PyTorch Geometric", "SQLAlchemy"] },
    { category: "AI Models", items: ["ViT-B/16", "Wav2Vec2", "GAT (Graph Attention)"] },
    { category: "Infrastructure", items: ["Docker", "TimescaleDB", "Redis", "FFmpeg"] }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto mb-16">
          <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
            VIDSUM-GNN
          </h1>
          <p className="text-2xl text-muted-foreground mb-4">
            Graph Neural Network-Based Video Summarization
          </p>
          <p className="text-lg text-muted-foreground mb-8">
            Leveraging Graph Attention Networks to automatically generate intelligent, 
            context-aware video summaries through multimodal scene analysis.
          </p>
          <div className="flex gap-4 justify-center">
            <Link to="/dashboard">
              <Button size="lg" className="gap-2">
                Try Dashboard <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
            <Button size="lg" variant="outline">
              View Documentation
            </Button>
          </div>
        </div>

        {/* Workflow Section */}
        <Card className="mb-16">
          <CardHeader>
            <CardTitle className="text-3xl">Processing Pipeline</CardTitle>
            <CardDescription>
              Six-stage multimodal processing with real-time progress tracking
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {workflow.map((item, idx) => (
                <div key={idx} className="flex flex-col items-center text-center">
                  <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                    {item.icon}
                  </div>
                  <div className="text-xs font-semibold text-primary mb-1">Step {item.step}</div>
                  <div className="font-medium mb-1">{item.label}</div>
                  <div className="text-xs text-muted-foreground">{item.desc}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Features Section */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="mb-4">{feature.icon}</div>
                  <CardTitle>{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Tech Stack Section */}
        <Card className="mb-16">
          <CardHeader>
            <CardTitle className="text-3xl">Technology Stack</CardTitle>
            <CardDescription>
              Built with modern AI/ML and web technologies
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {techStack.map((stack, idx) => (
                <div key={idx}>
                  <h3 className="font-semibold mb-3 text-primary">{stack.category}</h3>
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
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Team Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-3xl">Project Team</CardTitle>
            <CardDescription>
              AI253IA - Artificial Neural Networks and Deep Learning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { name: "Team Member 1", role: "Backend & GNN", usn: "USN001" },
                { name: "Team Member 2", role: "Frontend & UI", usn: "USN002" },
                { name: "Team Member 3", role: "ML Pipeline", usn: "USN003" },
                { name: "Team Member 4", role: "Infrastructure", usn: "USN004" }
              ].map((member, idx) => (
                <Card key={idx}>
                  <CardHeader>
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-3 mx-auto">
                      <span className="text-2xl font-bold text-primary">
                        {member.name.split(' ')[0][0]}{member.name.split(' ')[1]?.[0] || ''}
                      </span>
                    </div>
                    <CardTitle className="text-center text-lg">{member.name}</CardTitle>
                    <CardDescription className="text-center">{member.usn}</CardDescription>
                  </CardHeader>
                  <CardContent className="text-center">
                    <p className="text-sm text-muted-foreground">{member.role}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
