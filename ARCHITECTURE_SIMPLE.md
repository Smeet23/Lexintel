# LexIntel - Simplified Architecture

## 1. System Overview

```mermaid
graph TB
    Client["<b style='font-size:18px'>🖥️ CLIENT</b>"]
    API["<b style='font-size:18px'>⚡ FASTAPI</b><br/>API Server"]
    DB["<b style='font-size:18px'>🗄️ POSTGRESQL</b><br/>Database"]
    Queue["<b style='font-size:18px'>📮 REDIS</b><br/>Queue"]
    Worker["<b style='font-size:18px'>⚙️ CELERY</b><br/>Worker"]
    Qdrant["<b style='font-size:18px'>🔍 QDRANT</b><br/>Vectors"]
    Blob["<b style='font-size:18px'>☁️ AZURE BLOB</b><br/>Storage"]
    GoogleAI["<b style='font-size:18px'>🤖 GOOGLE AI</b><br/>API"]

    Client -->|REST API| API
    API --> DB
    API --> Queue
    API --> Qdrant
    API --> GoogleAI

    Queue --> Worker
    Worker --> DB
    Worker --> Blob
    Worker --> Qdrant
    Worker --> GoogleAI

    style Client fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,padding:20px
    style API fill:#bbdefb,stroke:#1976d2,stroke-width:3px,padding:20px
    style DB fill:#fff3e0,stroke:#f57c00,stroke-width:3px,padding:20px
    style Queue fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,padding:20px
    style Worker fill:#e8f5e9,stroke:#388e3c,stroke-width:3px,padding:20px
    style Qdrant fill:#c8e6c9,stroke:#388e3c,stroke-width:3px,padding:20px
    style Blob fill:#f1f8e9,stroke:#689f38,stroke-width:3px,padding:20px
    style GoogleAI fill:#fce4ec,stroke:#c2185b,stroke-width:3px,padding:20px
```

---

## 2. Document Upload Flow

```mermaid
graph LR
    A["<b style='font-size:16px'>📤 UPLOAD<br/>PDF</b>"] --> B["<b style='font-size:16px'>✅ VALIDATE</b>"]
    B --> C["<b style='font-size:16px'>💾 SAVE<br/>TO BLOB</b>"]
    C --> D["<b style='font-size:16px'>📮 QUEUE<br/>JOB</b>"]
    D --> E["<b style='font-size:16px'>⚙️ PROCESS</b>"]
    E --> F["<b style='font-size:16px'>✅ READY</b>"]

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,padding:15px
    style B fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,padding:15px
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px,padding:15px
    style D fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,padding:15px
    style E fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,padding:15px
    style F fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,padding:15px
```

---

## 3. Query Flow

```mermaid
graph LR
    A["<b style='font-size:16px'>❓ ASK<br/>QUESTION</b>"] --> B["<b style='font-size:16px'>🔍 SEARCH<br/>VECTORS</b>"]
    B --> C["<b style='font-size:16px'>📄 GET<br/>CONTEXT</b>"]
    C --> D["<b style='font-size:16px'>🤖 GENERATE<br/>ANSWER</b>"]
    D --> E["<b style='font-size:16px'>✅ RETURN<br/>ANSWER</b>"]

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,padding:15px
    style B fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,padding:15px
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px,padding:15px
    style D fill:#fce4ec,stroke:#c2185b,stroke-width:2px,padding:15px
    style E fill:#a5d6a7,stroke:#2e7d32,stroke-width:2px,padding:15px
```

---

## 4. Data Models

```mermaid
erDiagram
    USER ||--o{ CASE : owns
    CASE ||--o{ CHUNK : contains
    CASE ||--o{ QUERY : has

    USER {
        uuid id
        string email
        string password
    }

    CASE {
        uuid id
        string name
        string status
    }

    CHUNK {
        uuid id
        text content
        int page
    }

    QUERY {
        uuid id
        string question
        string answer
    }
```

---

## 5. Authentication

```mermaid
graph TB
    A["<b style='font-size:16px'>📝 REGISTER</b>"] --> B["<b style='font-size:16px'>🔐 HASH<br/>PASSWORD</b>"]
    B --> C["<b style='font-size:16px'>💾 STORE<br/>USER</b>"]

    D["<b style='font-size:16px'>🔑 LOGIN</b>"] --> E["<b style='font-size:16px'>✓ VERIFY<br/>PASSWORD</b>"]
    E --> F["<b style='font-size:16px'>🎟️ GENERATE<br/>JWT</b>"]

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,padding:12px
    style B fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,padding:12px
    style C fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,padding:12px
    style D fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,padding:12px
    style E fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,padding:12px
    style F fill:#a5d6a7,stroke:#2e7d32,stroke-width:2px,padding:12px
```

---

## 6. Background Job Retry

```mermaid
graph TD
    A["<b style='font-size:16px'>▶️ START<br/>JOB</b>"] --> B{"<b style='font-size:15px'>SUCCESS?</b>"}
    B -->|Yes| C["<b style='font-size:16px'>✅ COMPLETE</b>"]
    B -->|No| D{"<b style='font-size:15px'>ATTEMPTS<br/>< 3?</b>"}
    D -->|Yes| E["<b style='font-size:16px'>⏱️ RETRY<br/>0s→5s→10s</b>"]
    D -->|No| F["<b style='font-size:16px'>❌ FAILED</b>"]
    E --> A

    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,padding:15px
    style B fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,padding:15px
    style C fill:#a5d6a7,stroke:#2e7d32,stroke-width:2px,padding:15px
    style D fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,padding:15px
    style E fill:#fff3e0,stroke:#f57c00,stroke-width:2px,padding:15px
    style F fill:#ffcdd2,stroke:#c62828,stroke-width:2px,padding:15px
```

---

## 7. Scaling Strategy

```mermaid
graph TB
    LB["<b style='font-size:16px'>⚖️ LOAD<br/>BALANCER</b>"]

    subgraph "API Layer (Scale 3-10)"
        A1["<b style='font-size:14px'>🖥️ POD 1</b>"]
        A2["<b style='font-size:14px'>🖥️ POD 2</b>"]
        A3["<b style='font-size:14px'>🖥️ POD 3</b>"]
    end

    subgraph "Workers (Scale 5-20)"
        W1["<b style='font-size:14px'>⚙️ WORKER 1</b>"]
        W2["<b style='font-size:14px'>⚙️ WORKER 2</b>"]
        W3["<b style='font-size:14px'>⚙️ WORKER 3</b>"]
    end

    DB["<b style='font-size:16px'>🗄️ POSTGRESQL<br/>Primary + Replicas</b>"]
    Cache["<b style='font-size:16px'>📮 REDIS<br/>Cluster</b>"]
    Vector["<b style='font-size:16px'>🔍 QDRANT<br/>Cluster</b>"]

    LB --> A1
    LB --> A2
    LB --> A3

    A1 --> DB
    A2 --> DB
    A3 --> DB

    Cache --> W1
    Cache --> W2
    Cache --> W3

    W1 --> Vector
    W2 --> Vector
    W3 --> Vector

    style LB fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,padding:15px
    style A1 fill:#bbdefb,stroke:#1976d2,stroke-width:2px,padding:12px
    style A2 fill:#bbdefb,stroke:#1976d2,stroke-width:2px,padding:12px
    style A3 fill:#bbdefb,stroke:#1976d2,stroke-width:2px,padding:12px
    style W1 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,padding:12px
    style W2 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,padding:12px
    style W3 fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,padding:12px
    style DB fill:#fff3e0,stroke:#f57c00,stroke-width:2px,padding:15px
    style Cache fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,padding:15px
    style Vector fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,padding:15px
```

---

## Key Concepts

### API Endpoints
- `POST /auth/register` - Create account
- `POST /auth/login` - Get JWT token
- `POST /cases` - Upload PDF
- `POST /cases/{id}/ask` - Ask question
- `GET /cases/{id}/status` - Check progress

### Processing Pipeline
1. **Upload** → Validate PDF → Save to Azure Blob
2. **Queue** → Background worker picks up
3. **Process** → Chunk → Embed → Store vectors
4. **Ready** → User can query

### Query Pipeline
1. **Question** → Embed to vector
2. **Search** → Find similar chunks in Qdrant
3. **Context** → Format with page numbers
4. **Generate** → Ask Gemini 2.5 Flash Lite
5. **Return** → Answer with citations

### Security
- Passwords hashed with bcrypt
- JWT tokens (24-hour expiry)
- File validation (magic bytes)
- Parameterized SQL queries
- User ownership verification

### Tech Stack
| Component | Technology |
|-----------|-----------|
| API | FastAPI |
| Database | PostgreSQL |
| Vectors | Qdrant |
| Storage | Azure Blob |
| LLM | Google Gemini |
| Embeddings | Google gemini-embedding-001 |
| Queue | Redis + Celery |
| Auth | JWT + Bcrypt |

### Performance
- Document processing: 2-3 seconds
- Query latency: 3-5 seconds
- Throughput: 30 docs/min, 100+ queries/min

### Error Handling
- Automatic retries with backoff (0s → 5s → 10s)
- Max 3 attempts per job
- Case marked as "error" on failure
- User notified of issues
