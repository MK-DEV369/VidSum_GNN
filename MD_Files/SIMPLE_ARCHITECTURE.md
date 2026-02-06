graph LR
    A[👤 User] --> B[📱 Frontend<br/>React + TypeScript]
    B --> C[🚀 Backend<br/>FastAPI]

    C --> D[📤 Upload<br/>Video]
    D --> E[⚙️ Process<br/>Video]
    E --> F[🧠 Extract<br/>Features]
    F --> G[📊 Build<br/>Graph]
    G --> H[🕸️ GNN<br/>Score]
    H --> I[🎯 Select<br/>Shots]
    I --> J[🎬 Generate<br/>Summary]

    J --> K[💾 Store<br/>Results]
    K --> B

    B -.-> L[WebSocket<br/>Updates]
    F -.-> M[ViT + Wav2Vec2]
    J -.-> N[FFmpeg + FLAN-T5]
