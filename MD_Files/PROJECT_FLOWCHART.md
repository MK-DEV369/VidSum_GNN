```mermaid
flowchart TD
    %% User Interface Flow
    A[User Visits HomePage] --> B[View Project Info & Workflow]
    B --> C[Navigate to Dashboard]

    %% Upload and Configuration
    C --> D[Select Video File]
    D --> E[Configure Summary Options]
    E --> F[Text Length: Short/Medium/Long]
    E --> G[Summary Format: Bullet/Structured/Plain]
    E --> H[Summary Type: Balanced/Visual/Audio/Highlight]
    F --> I[Upload Video to Backend]
    G --> I
    H --> I

    %% Backend Processing Pipeline
    I --> J[Save Video to uploads/]
    J --> K[Video Transcoding & Validation]
    K --> L[Shot Detection & Segmentation]

    %% Feature Extraction (Parallel)
    L --> M[Visual Feature Extraction]
    L --> N[Audio Feature Extraction]
    L --> O[Handcrafted Feature Extraction]

    M --> P[ViT-B/16 Model<br/>768-dim embeddings]
    N --> Q[Wav2Vec2 Model<br/>768-dim embeddings]
    O --> R[Handcrafted Features<br/>Motion, Color, etc.]

    %% Graph Construction
    P --> S[Scene Graph Construction]
    Q --> S
    R --> S

    S --> T[Nodes: Shot Features<br/>1536-dim combined]
    S --> U[Edges: Temporal<br/>Adjacent shots]
    S --> V[Edges: Semantic<br/>Content similarity]
    S --> W[Edges: Audio<br/>Speech patterns]

    %% GNN Processing
    T --> X[GNN Inference]
    U --> X
    V --> X
    W --> X

    X --> Y[GAT-based VidSumGNN<br/>Importance Scoring]

    %% Summary Generation
    Y --> Z[Shot Selection Strategy]
    Z --> AA[Greedy Algorithm]
    Z --> BB[Knapsack DP]
    Z --> CC[User Preferences Applied]

    AA --> DD[Select Top-K Shots]
    BB --> DD
    CC --> DD

    DD --> EE[Summary Assembly]
    EE --> FF[FFmpeg Video Concatenation]
    EE --> GG[Text Summary Generation<br/>FLAN-T5 Model]

    %% Results and Output
    FF --> HH[Save Summary Video<br/>to outputs/]
    GG --> II[Save Text Summary<br/>to database]

    HH --> JJ[Return Results to Frontend]
    II --> JJ

    %% Frontend Results Display
    JJ --> KK[Display Processing Complete]
    KK --> LL[Show Text Summary]
    KK --> MM[Download Summary Video]
    KK --> NN[View Shot Scores & Visualization]

    %% Real-time Feedback
    I --> OO[WebSocket Connection]
    OO --> PP[Stream Progress Logs]
    PP --> QQ[Update UI Progress Bar]
    PP --> RR[Show Current Stage]
    PP --> SS[Display Real-time Logs]

    %% Database Storage
    J --> TT[Store Video Metadata<br/>TimescaleDB]
    L --> UU[Store Shot Information]
    Y --> VV[Store Shot Scores]
    II --> WW[Store Summary Text]

    %% Error Handling
    K --> XX{Validation Failed?}
    XX -->|Yes| YY[Return Error to User]
    XX -->|No| L

    X --> ZZ{Processing Error?}
    ZZ -->|Yes| AAA[Log Error & Notify User]
    ZZ -->|No| Y

    %% Memory Management
    X --> BBB[GPU Memory Monitoring]
    BBB --> CCC{ Memory High? }
    CCC -->|Yes| DDD[Clear Cache<br/>torch.cuda.empty_cache()]
    CCC -->|No| Y
    DDD --> Y

    %% Styling
    classDef userInterface fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef backend fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef processing fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef database fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef error fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    class A,B,C,D,E,F,G,H userInterface
    class I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,BB,CC,DD,EE,FF,GG,HH,II,JJ,KK,LL,MM,NN processing
    class OO,PP,QQ,RR,SS backend
    class TT,UU,VV,WW database
    class XX,YY,ZZ,AAA error
    class BBB,CCC,DDD processing
```