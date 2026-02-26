# LexIntel - Architecture Diagrams (Mermaid)

## 1. System Architecture Overview

```mermaid
graph TB
    Client["🖥️ Client<br/>Frontend/Mobile"]

    subgraph "API Layer (FastAPI)"
        Auth["🔐 Authentication<br/>- Register<br/>- Login<br/>- JWT"]
        Cases["📁 Case Management<br/>- Upload<br/>- List<br/>- Status"]
        RAG["🧠 RAG Engine<br/>- Query<br/>- Search<br/>- Generate"]
        Health["❤️ Health Check"]
    end

    subgraph "Data Layer"
        PG["🗄️ PostgreSQL<br/>Users, Cases<br/>Chunks, Queries"]
        Redis["📮 Redis<br/>Task Queue"]
    end

    subgraph "Vector & Storage"
        Qdrant["🔍 Qdrant<br/>Vector Database<br/>768-d embeddings"]
        Blob["☁️ Azure Blob<br/>PDF Storage"]
    end

    subgraph "External Services"
        GoogleAI["🤖 Google AI API<br/>- Embeddings<br/>- Gemini 2.5 Flash Lite"]
    end

    subgraph "Background Processing"
        Worker["👷 Celery Worker<br/>PDF Processing<br/>Chunking, Embedding"]
    end

    Client -->|REST API| Auth
    Client -->|REST API| Cases
    Client -->|REST API| RAG
    Client -->|Health Check| Health

    Auth -->|User Data| PG
    Cases -->|Case Data| PG
    RAG -->|Query/Results| PG

    Cases -->|Queue Task| Redis
    Redis -->|Pick Job| Worker

    Worker -->|Store Chunks| PG
    Worker -->|Store Vectors| Qdrant
    Worker -->|Download/Upload| Blob
    Worker -->|Embed Chunks| GoogleAI

    RAG -->|Semantic Search| Qdrant
    RAG -->|Generate Answer| GoogleAI

    style Client fill:#e1f5ff
    style Auth fill:#fff3e0
    style Cases fill:#f3e5f5
    style RAG fill:#e8f5e9
    style Health fill:#fce4ec
    style PG fill:#ffe0b2
    style Redis fill:#ffccbc
    style Qdrant fill:#c8e6c9
    style Blob fill:#b3e5fc
    style GoogleAI fill:#f8bbd0
    style Worker fill:#d1c4e9
```

---

## 2. Document Upload & Processing Pipeline

```mermaid
graph LR
    Start([User Uploads PDF])

    Validate["✓ Validate<br/>- MIME type<br/>- Magic bytes<br/>- Filename<br/>- Case name"]

    CreateCase["Create Case<br/>status: processing"]

    UploadBlob["Upload PDF to<br/>Azure Blob<br/>cases/{case_id}/{file}"]

    QueueTask["Queue Celery Task<br/>process_document_task"]

    subgraph Worker["🔄 Background Worker"]
        Download["Download PDF<br/>from Blob"]
        Chunk["Chunk PDF<br/>1500 chars<br/>300 overlap"]
        Embed["Generate Embeddings<br/>Google AI gemini-embedding-001"]
        StorePG["Store Chunks<br/>in PostgreSQL"]
        CreateQdrant["Create Qdrant<br/>Collection"]
        Upsert["Upsert Vectors<br/>with Metadata"]
    end

    Success["✅ Success<br/>Case: processing → ready<br/>Job: completed"]

    Failure["❌ Failure<br/>Attempt++<br/>Schedule Retry<br/>0s → 5s → 10s"]

    MaxRetry{"Max Attempts<br/>Exceeded?"}

    FinalFail["Case: error<br/>Job: failed<br/>User Notified"]

    UserReady["👤 User Notified<br/>Case Ready"]

    Start --> Validate
    Validate -->|Valid| CreateCase
    Validate -->|Invalid| Start

    CreateCase --> UploadBlob
    UploadBlob --> QueueTask
    QueueTask -->|Wait in Queue| Worker

    Download --> Chunk
    Chunk --> Embed
    Embed --> StorePG
    StorePG --> CreateQdrant
    CreateQdrant --> Upsert

    Upsert -->|Success| Success
    Upsert -->|Error| Failure

    Failure --> MaxRetry
    MaxRetry -->|No| QueueTask
    MaxRetry -->|Yes| FinalFail

    Success --> UserReady
    FinalFail --> UserReady

    style Start fill:#c8e6c9
    style Validate fill:#fff9c4
    style CreateCase fill:#ffe0b2
    style UploadBlob fill:#ffccbc
    style QueueTask fill:#f8bbd0
    style Download fill:#e1bee7
    style Chunk fill:#d1c4e9
    style Embed fill:#c5cae9
    style StorePG fill:#bbdefb
    style CreateQdrant fill:#b3e5fc
    style Upsert fill:#b2dfdb
    style Success fill:#c8e6c9
    style Failure fill:#ffccbc
    style FinalFail fill:#ffcdd2
    style UserReady fill:#c8e6c9
```

---

## 3. RAG Query Pipeline (Detailed)

```mermaid
graph TD
    Start([User Asks Question])

    Validate["1️⃣ Validate Query<br/>Length: 1-5000 chars<br/>Case ownership"]

    QueryEmbed["2️⃣ Embed Query<br/>Google AI gemini-embedding-001<br/>→ 768-d vector"]

    VectorSearch["3️⃣ Vector Search<br/>Qdrant semantic similarity<br/>→ Top 10 results"]

    Filter["4️⃣ Filter by Confidence<br/>Keep score ≥ 0.15<br/>Deduplicate"]

    Format["5️⃣ Format Context<br/>Sort by relevance<br/>Add page numbers<br/>Calculate tokens"]

    TokenCheck{"6️⃣ Token Budget<br/>Check?"}

    TrimChunks["Trim to 2 chunks<br/>Recount tokens"]

    FinalCheck{"Still within<br/>12,800 tokens?"}

    ErrorToken["⚠️ Error: Context<br/>Too Large"]

    GenerateAnswer["7️⃣ Generate Answer<br/>Google AI gemini-2.5-flash-lite<br/>Temperature: 0.2<br/>Timeout: 30s"]

    ExtractCite["8️⃣ Extract Citations<br/>Parse [Page X]<br/>Match to sources<br/>Flag hallucinations"]

    CalcConfidence["Calculate Confidence<br/>high: ≥0.9<br/>medium: ≥0.8<br/>low: <0.8<br/>none: error"]

    Response["📨 Return Response<br/>answer, sources,<br/>confidence, tokens"]

    Start --> Validate
    Validate -->|Valid| QueryEmbed
    Validate -->|Invalid| ErrorToken

    QueryEmbed --> VectorSearch
    VectorSearch --> Filter
    Filter --> Format
    Format --> TokenCheck

    TokenCheck -->|≤12,800| GenerateAnswer
    TokenCheck -->|>12,800| TrimChunks
    TrimChunks --> FinalCheck
    FinalCheck -->|≤12,800| GenerateAnswer
    FinalCheck -->|>12,800| ErrorToken

    GenerateAnswer --> ExtractCite
    ExtractCite --> CalcConfidence
    CalcConfidence --> Response

    ErrorToken --> Response

    style Start fill:#c8e6c9
    style Validate fill:#fff9c4
    style QueryEmbed fill:#b2dfdb
    style VectorSearch fill:#b3e5fc
    style Filter fill:#bbdefb
    style Format fill:#c5cae9
    style TokenCheck fill:#fff9c4
    style TrimChunks fill:#ffe0b2
    style FinalCheck fill:#fff9c4
    style ErrorToken fill:#ffcdd2
    style GenerateAnswer fill:#f8bbd0
    style ExtractCite fill:#e1bee7
    style CalcConfidence fill:#d1c4e9
    style Response fill:#c8e6c9
```

---

## 4. Data Model Relationships (ER Diagram)

```mermaid
erDiagram
    USER ||--o{ CASE : owns
    USER ||--o{ QUERY : asks
    CASE ||--o{ CHUNK : contains
    CASE ||--o{ QUERY : has
    CASE ||--o{ PROCESSINGJOB : tracked_by

    USER {
        uuid id PK
        string email UK
        string password_hash
        boolean is_deleted
        datetime created_at
        datetime updated_at
    }

    CASE {
        uuid id PK
        uuid user_id FK
        string name
        string blob_storage_path
        enum status
        datetime created_at
        datetime updated_at
    }

    CHUNK {
        uuid id PK
        uuid case_id FK
        integer page_num
        string section_name
        text content
        string embedding_hash
        integer chunk_sequence
        datetime created_at
    }

    QUERY {
        uuid id PK
        uuid case_id FK
        uuid user_id FK
        string question
        text answer
        json citations
        datetime created_at
    }

    PROCESSINGJOB {
        uuid id PK
        uuid case_id FK
        enum status
        string error_message
        integer attempts
        integer max_attempts
        datetime created_at
        datetime started_at
        datetime completed_at
        datetime next_retry_at
    }
```

---

## 5. Authentication & Authorization Flow

```mermaid
graph TD
    Register["📝 User Registration"]
    Email["Input: email, password"]
    Validate["Validate Email<br/>Validate Password<br/>8+ chars, uppercase, digit"]
    Hash["Hash Password<br/>bcrypt(password, salt)"]
    StorePG["Store in PostgreSQL"]
    Created["✅ User Created"]

    Login["🔑 User Login"]
    LoginEmail["Input: email, password"]
    CheckEmail["Check Email Exists"]
    VerifyPass["Verify Password<br/>bcrypt.verify"]
    GenerateJWT["Generate JWT Token<br/>Payload: sub=user_id<br/>Expiry: 24 hours<br/>Algorithm: HS256"]
    ReturnToken["Return access_token"]

    Protected["🛡️ Protected Requests"]
    ExtractToken["Extract Authorization<br/>Bearer header"]
    DecodeJWT["Decode JWT<br/>Verify signature<br/>Verify expiry"]
    GetUser["Retrieve User<br/>from Database"]
    VerifyNotDeleted["Verify not deleted"]
    GrantAccess["✅ Access Granted"]

    Denied["❌ Access Denied<br/>401 Unauthorized"]

    subgraph "Registration"
        Register --> Email
        Email --> Validate
        Validate -->|Valid| Hash
        Validate -->|Invalid| Register
        Hash --> StorePG
        StorePG --> Created
    end

    subgraph "Login"
        Login --> LoginEmail
        LoginEmail --> CheckEmail
        CheckEmail -->|Exists| VerifyPass
        CheckEmail -->|Not Found| Denied
        VerifyPass -->|Match| GenerateJWT
        VerifyPass -->|No Match| Denied
        GenerateJWT --> ReturnToken
    end

    subgraph "Protected Access"
        Protected --> ExtractToken
        ExtractToken -->|No Token| Denied
        ExtractToken -->|Token Found| DecodeJWT
        DecodeJWT -->|Invalid| Denied
        DecodeJWT -->|Valid| GetUser
        GetUser --> VerifyNotDeleted
        VerifyNotDeleted -->|Deleted| Denied
        VerifyNotDeleted -->|Active| GrantAccess
    end

    style Register fill:#c8e6c9
    style Login fill:#c8e6c9
    style Protected fill:#c8e6c9
    style Created fill:#a5d6a7
    style ReturnToken fill:#a5d6a7
    style GrantAccess fill:#a5d6a7
    style Denied fill:#ffcdd2
```

---

## 6. Celery Background Job Processor

```mermaid
graph TD
    Start["🚀 Start Job Processor"]
    Sleep["💤 Sleep 10 seconds"]
    GetJobs["Get pending jobs<br/>LIMIT 5<br/>FIFO order"]

    NoJobs{"Any Jobs?"}

    PickJob["Pick next job<br/>status: pending<br/>→ processing"]

    subgraph Processing["📦 Process Job"]
        Download["Download PDF<br/>from Azure Blob"]
        Chunk["Chunk PDF<br/>1500 chars, 300 overlap"]
        Embed["Embed chunks<br/>Google AI API"]
        CreateColl["Create Qdrant<br/>Collection"]
        UpsertVec["Upsert Vectors<br/>with metadata"]
        StoreMeta["Store Chunks<br/>PostgreSQL"]
    end

    Success{"Success?"}

    SuccessBranch["✅ Mark Complete<br/>Job: completed<br/>Case: ready"]

    ErrorBranch["❌ Catch Error<br/>Increment attempts"]

    MaxAttempts{"Attempts<br/>< Max?"}

    Retry["Schedule Retry<br/>Attempt 1: 0s<br/>Attempt 2: 5s<br/>Attempt 3: 10s<br/>status: pending"]

    FailJob["Mark Failed<br/>Job: failed<br/>Case: error<br/>error_message: set"]

    BatchComplete{"More jobs<br/>in batch?"}

    NextJob["Pick next<br/>job"]

    AllDone["All jobs<br/>complete"]

    Back["Loop back"]

    Start --> Sleep
    Sleep --> GetJobs
    GetJobs --> NoJobs

    NoJobs -->|No| Back
    NoJobs -->|Yes| PickJob

    PickJob --> Download
    Download --> Chunk
    Chunk --> Embed
    Embed --> CreateColl
    CreateColl --> UpsertVec
    UpsertVec --> StoreMeta

    StoreMeta --> Success

    Success -->|Yes| SuccessBranch
    Success -->|No| ErrorBranch

    ErrorBranch --> MaxAttempts
    MaxAttempts -->|Yes| Retry
    MaxAttempts -->|No| FailJob

    Retry --> BatchComplete
    FailJob --> BatchComplete
    SuccessBranch --> BatchComplete

    BatchComplete -->|Yes| NextJob
    NextJob --> Download

    BatchComplete -->|No| AllDone
    AllDone --> Sleep

    style Start fill:#c8e6c9
    style Sleep fill:#fff9c4
    style GetJobs fill:#bbdefb
    style NoJobs fill:#fff9c4
    style PickJob fill:#b3e5fc
    style Download fill:#e1bee7
    style Chunk fill:#d1c4e9
    style Embed fill:#c5cae9
    style CreateColl fill:#bbdefb
    style UpsertVec fill:#b2dfdb
    style StoreMeta fill:#a5d6a7
    style Success fill:#fff9c4
    style SuccessBranch fill:#c8e6c9
    style ErrorBranch fill:#ffccbc
    style Retry fill:#ffe0b2
    style FailJob fill:#ffcdd2
    style BatchComplete fill:#fff9c4
    style AllDone fill:#c8e6c9
```

---

## 7. Component Interaction Sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI API
    participant Auth as Auth Service
    participant DB as PostgreSQL
    participant Queue as Redis Queue
    participant Worker as Celery Worker
    participant Blob as Azure Blob
    participant GoogleAI as Google AI API
    participant Qdrant as Qdrant VectorDB

    Client->>API: POST /auth/register
    API->>Auth: hash_password()
    Auth->>DB: Store user
    DB-->>API: User created
    API-->>Client: ✅ Success

    Client->>API: POST /auth/login
    API->>Auth: verify_password()
    Auth->>DB: Get user
    DB-->>Auth: User data
    Auth->>Auth: create_jwt_token()
    Auth-->>API: Token
    API-->>Client: access_token

    Client->>API: POST /cases (with PDF)
    API->>API: Validate file
    API->>DB: Create Case record
    DB-->>API: Case created
    API->>Blob: upload_pdf_to_blob()
    Blob-->>API: ✅ Stored
    API->>Queue: Queue task: process_document_task
    Queue-->>API: Task queued
    API-->>Client: Case ID + status:processing

    Worker->>Queue: Poll for jobs
    Queue-->>Worker: process_document_task
    Worker->>Blob: download_pdf_from_blob()
    Blob-->>Worker: PDF bytes
    Worker->>Worker: chunk_pdf()
    Worker->>GoogleAI: embed_text() [batch]
    GoogleAI-->>Worker: 768-d vectors
    Worker->>DB: Create Chunk records
    DB-->>Worker: ✅ Stored
    Worker->>Qdrant: create_collection()
    Qdrant-->>Worker: ✅ Collection created
    Worker->>Qdrant: upsert_vectors()
    Qdrant-->>Worker: ✅ Vectors stored
    Worker->>DB: Update Case status=ready
    DB-->>Worker: ✅ Updated
    Worker->>Queue: Mark job completed
    Queue-->>Worker: ✅ Complete

    Client->>API: GET /cases/{case_id}/status
    API->>DB: Get Case
    DB-->>API: status=ready
    API-->>Client: Processing complete

    Client->>API: POST /cases/{case_id}/ask (with question)
    API->>GoogleAI: embed_query()
    GoogleAI-->>API: Query vector
    API->>Qdrant: search_vectors()
    Qdrant-->>API: Top 10 chunks
    API->>API: Filter + Format context
    API->>GoogleAI: generate_answer(context, question)
    GoogleAI-->>API: Answer + tokens
    API->>API: extract_citations()
    API->>DB: Create Query record
    DB-->>API: ✅ Stored
    API-->>Client: answer + sources + confidence
```

---

## 8. Token Budget Management Flow

```mermaid
graph TD
    Start["Start Query Processing"]

    GetChunks["Get top 4 chunks<br/>from Qdrant"]

    CountQuery["Count tokens<br/>Query: 50 tokens"]
    CountSystem["Count tokens<br/>System prompt: 500 tokens"]
    CountContext["Count tokens<br/>Context (4 chunks): ~8000 tokens"]
    CountBuffer["Add buffer: 500 tokens"]

    Total["Total = 50+500+8000+500<br/>= 9,050 tokens"]

    BudgetCheck{"Total ≤<br/>12,800?"}

    Within["✅ Within Budget<br/>Proceed with<br/>4 chunks"]

    Trim2["⚠️ Reduce to<br/>2 chunks<br/>Recalculate"]

    Recount["New total<br/>= 50+500+4000+500<br/>= 5,050 tokens"]

    Recheck{"Total ≤<br/>12,800?"}

    StillWithin["✅ Now Within Budget<br/>Proceed with<br/>2 chunks"]

    ErrorBudget["❌ Error: Context<br/>Too Large<br/>Return empty response"]

    LLMCall["Call Google AI Gemini 2.5 Flash Lite<br/>with context<br/>Max output: 2000 tokens"]

    FinalTotal["Final tokens used =<br/>input + output"]

    Complete["✅ Complete"]

    Start --> GetChunks
    GetChunks --> CountQuery
    CountQuery --> CountSystem
    CountSystem --> CountContext
    CountContext --> CountBuffer
    CountBuffer --> Total

    Total --> BudgetCheck
    BudgetCheck -->|Yes| Within
    BudgetCheck -->|No| Trim2

    Within --> LLMCall

    Trim2 --> Recount
    Recount --> Recheck
    Recheck -->|Yes| StillWithin
    Recheck -->|No| ErrorBudget

    StillWithin --> LLMCall
    ErrorBudget --> Complete

    LLMCall --> FinalTotal
    FinalTotal --> Complete

    style Start fill:#c8e6c9
    style GetChunks fill:#bbdefb
    style CountQuery fill:#b3e5fc
    style CountSystem fill:#b3e5fc
    style CountContext fill:#b3e5fc
    style CountBuffer fill:#b3e5fc
    style Total fill:#b2dfdb
    style BudgetCheck fill:#fff9c4
    style Within fill:#a5d6a7
    style Trim2 fill:#ffe0b2
    style Recount fill:#ffccbc
    style Recheck fill:#fff9c4
    style StillWithin fill:#a5d6a7
    style ErrorBudget fill:#ffcdd2
    style LLMCall fill:#f8bbd0
    style FinalTotal fill:#e1bee7
    style Complete fill:#c8e6c9
```

---

## 9. Horizontal Scaling Architecture

```mermaid
graph TB
    Users["👥 Multiple Users"]
    LB["⚖️ Load Balancer<br/>nginx / HAProxy"]

    subgraph "API Layer - Scaled"
        API1["FastAPI Pod 1<br/>Port 8000"]
        API2["FastAPI Pod 2<br/>Port 8000"]
        API3["FastAPI Pod 3<br/>Port 8000"]
        APIMore["...API Pods<br/>3-10 instances"]
    end

    subgraph "Database Tier"
        PGPrimary["🗄️ PostgreSQL<br/>Primary"]
        PGReplica["🗄️ PostgreSQL<br/>Read Replica"]
    end

    subgraph "Cache & Queue"
        RedisCluster["📮 Redis Cluster<br/>High Availability"]
    end

    subgraph "Background Workers - Scaled"
        Worker1["Celery Worker 1"]
        Worker2["Celery Worker 2"]
        Worker3["Celery Worker 3"]
        WorkerMore["...Workers<br/>5-20 instances"]
    end

    subgraph "Vector & Search"
        QdrantNode1["🔍 Qdrant Node 1"]
        QdrantNode2["🔍 Qdrant Node 2"]
        QdrantNode3["🔍 Qdrant Node 3"]
        QdrantCluster["Qdrant Cluster<br/>High Availability"]
    end

    subgraph "Storage"
        BlobStorage["☁️ Azure Blob Storage<br/>Distributed, Redundant"]
    end

    Users --> LB

    LB --> API1
    LB --> API2
    LB --> API3
    LB --> APIMore

    API1 --> PGPrimary
    API2 --> PGPrimary
    API3 --> PGPrimary
    APIMore --> PGPrimary

    API1 --> PGReplica
    API2 --> PGReplica
    API3 --> PGReplica

    API1 --> RedisCluster
    API2 --> RedisCluster
    API3 --> RedisCluster
    APIMore --> RedisCluster

    RedisCluster --> Worker1
    RedisCluster --> Worker2
    RedisCluster --> Worker3
    RedisCluster --> WorkerMore

    Worker1 --> BlobStorage
    Worker2 --> BlobStorage
    Worker3 --> BlobStorage
    WorkerMore --> BlobStorage

    Worker1 --> QdrantNode1
    Worker2 --> QdrantNode2
    Worker3 --> QdrantNode3
    WorkerMore --> QdrantCluster

    PGPrimary -.->|Replicates| PGReplica
    QdrantNode1 -.->|Syncs| QdrantCluster
    QdrantNode2 -.->|Syncs| QdrantCluster
    QdrantNode3 -.->|Syncs| QdrantCluster

    style LB fill:#fff9c4
    style API1 fill:#bbdefb
    style API2 fill:#bbdefb
    style API3 fill:#bbdefb
    style APIMore fill:#bbdefb
    style PGPrimary fill:#ffe0b2
    style PGReplica fill:#ffccbc
    style RedisCluster fill:#f8bbd0
    style Worker1 fill:#d1c4e9
    style Worker2 fill:#d1c4e9
    style Worker3 fill:#d1c4e9
    style WorkerMore fill:#d1c4e9
    style QdrantNode1 fill:#b2dfdb
    style QdrantNode2 fill:#b2dfdb
    style QdrantNode3 fill:#b2dfdb
    style QdrantCluster fill:#a5d6a7
    style BlobStorage fill:#b3e5fc
```

---

## 10. Error Handling & Retry Strategy

```mermaid
graph TD
    Start["Job Processing Starts"]

    Attempt1["🔄 Attempt 1<br/>Execute immediately<br/>retry_delay=0"]

    Error1{"Error?"}

    Success1["✅ Success"]

    Error1Branch["Store error<br/>attempts=1<br/>Set status=pending"]

    Retry1["⏱️ Schedule Retry<br/>next_retry_at = now<br/>Sleep 0 seconds"]

    Attempt2["🔄 Attempt 2<br/>Execute<br/>retry_delay=5s"]

    Error2{"Error?"}

    Success2["✅ Success"]

    Error2Branch["Store error<br/>attempts=2<br/>Set status=pending"]

    Retry2["⏱️ Schedule Retry<br/>next_retry_at = now + 5s<br/>Sleep 5 seconds"]

    Attempt3["🔄 Attempt 3<br/>Execute<br/>retry_delay=10s"]

    Error3{"Error?"}

    Success3["✅ Success"]

    Error3Branch["Store error<br/>attempts=3<br/>Set status=pending"]

    Retry3["⏱️ Schedule Retry<br/>next_retry_at = now + 10s<br/>Sleep 10 seconds"]

    MaxExceeded["❌ Max Attempts<br/>Exceeded<br/>attempts > max_attempts"]

    FinalFail["Set status=failed<br/>Case status=error<br/>Log error message"]

    Notify["📧 Notify User<br/>Document processing failed"]

    Start --> Attempt1
    Attempt1 --> Error1

    Error1 -->|No| Success1
    Error1 -->|Yes| Error1Branch

    Error1Branch --> Retry1
    Retry1 --> Attempt2

    Attempt2 --> Error2
    Error2 -->|No| Success2
    Error2 -->|Yes| Error2Branch

    Error2Branch --> Retry2
    Retry2 --> Attempt3

    Attempt3 --> Error3
    Error3 -->|No| Success3
    Error3 -->|Yes| Error3Branch

    Error3Branch --> Retry3
    Retry3 --> MaxExceeded

    MaxExceeded --> FinalFail
    FinalFail --> Notify

    Success1 --> Notify
    Success2 --> Notify
    Success3 --> Notify

    style Start fill:#c8e6c9
    style Attempt1 fill:#b3e5fc
    style Attempt2 fill:#b3e5fc
    style Attempt3 fill:#b3e5fc
    style Error1 fill:#fff9c4
    style Error2 fill:#fff9c4
    style Error3 fill:#fff9c4
    style Success1 fill:#a5d6a7
    style Success2 fill:#a5d6a7
    style Success3 fill:#a5d6a7
    style Error1Branch fill:#ffccbc
    style Error2Branch fill:#ffccbc
    style Error3Branch fill:#ffccbc
    style Retry1 fill:#ffe0b2
    style Retry2 fill:#ffe0b2
    style Retry3 fill:#ffe0b2
    style MaxExceeded fill:#ffcdd2
    style FinalFail fill:#ef5350
    style Notify fill:#c8e6c9
```

---

## 11. Security Layers

```mermaid
graph TB
    Client["👤 Client Request"]

    Layer1["Layer 1: Network<br/>├─ HTTPS/TLS<br/>├─ CORS Policy<br/>└─ WAF Rules"]

    Layer2["Layer 2: Request Validation<br/>├─ Content-Type check<br/>├─ MIME type validation<br/>├─ File magic bytes<br/>└─ Filename validation"]

    Layer3["Layer 3: Authentication<br/>├─ JWT token validation<br/>├─ Signature verification<br/>├─ Expiry check<br/>└─ User existence check"]

    Layer4["Layer 4: Authorization<br/>├─ Ownership verification<br/>├─ Role-based access<br/>└─ Resource ACL"]

    Layer5["Layer 5: Input Sanitization<br/>├─ Pydantic validation<br/>├─ Type checking<br/>└─ Range validation"]

    Layer6["Layer 6: SQL Injection<br/>├─ Parameterized queries<br/>├─ ORM protection<br/>└─ No raw SQL"]

    Layer7["Layer 7: Processing<br/>├─ Error handling<br/>├─ Transaction rollback<br/>└─ Logging without secrets"]

    Response["✅ Approved Request<br/>or<br/>❌ Rejected with<br/>Generic Error Message"]

    Client --> Layer1
    Layer1 -->|Pass| Layer2
    Layer1 -->|Fail| Response

    Layer2 -->|Pass| Layer3
    Layer2 -->|Fail| Response

    Layer3 -->|Pass| Layer4
    Layer3 -->|Fail| Response

    Layer4 -->|Pass| Layer5
    Layer4 -->|Fail| Response

    Layer5 -->|Pass| Layer6
    Layer5 -->|Fail| Response

    Layer6 -->|Pass| Layer7
    Layer6 -->|Fail| Response

    Layer7 --> Response

    style Client fill:#e3f2fd
    style Layer1 fill:#bbdefb
    style Layer2 fill:#90caf9
    style Layer3 fill:#64b5f6
    style Layer4 fill:#42a5f5
    style Layer5 fill:#2196f3
    style Layer6 fill:#1e88e5
    style Layer7 fill:#1976d2
    style Response fill:#c8e6c9
```

---

## 12. Deployment Pipeline (CI/CD)

```mermaid
graph LR
    Developer["👨‍💻 Developer<br/>Commits Code"]

    Git["📦 Git Repository<br/>Push to main/develop"]

    CI["🔄 CI Pipeline<br/>Tests + Lint"]

    Tests{"Tests Pass?"}

    Build["🏗️ Build Stage<br/>Docker image<br/>Push to registry"]

    Dev["🚀 Deploy Dev<br/>Docker Compose<br/>PostgreSQL, Qdrant, Redis"]

    Staging["🚀 Deploy Staging<br/>Kubernetes<br/>3x API pods<br/>3x Workers<br/>Read replicas"]

    Tests2{"Smoke Tests<br/>Pass?"}

    Prod["🚀 Deploy Production<br/>Kubernetes<br/>5-10x API pods<br/>10-20x Workers<br/>Qdrant cluster<br/>HA database"}

    Monitor["📊 Monitoring<br/>Prometheus<br/>Grafana<br/>ELK Stack"]

    Developer --> Git
    Git --> CI

    CI --> Tests
    Tests -->|Fail| Developer
    Tests -->|Pass| Build

    Build --> Dev
    Dev --> Staging
    Staging --> Tests2

    Tests2 -->|Fail| Developer
    Tests2 -->|Pass| Prod

    Prod --> Monitor

    style Developer fill:#c8e6c9
    style Git fill:#bbdefb
    style CI fill:#90caf9
    style Tests fill:#fff9c4
    style Build fill:#ffe0b2
    style Dev fill:#ffccbc
    style Staging fill:#f8bbd0
    style Tests2 fill:#fff9c4
    style Prod fill:#e1bee7
    style Monitor fill:#d1c4e9
```

---

## 13. RAG Pipeline with Confidence Scoring

```mermaid
graph TD
    Query["User Query"]

    RetrievalTop10["Retrieve Top 10"]
    Scores["Scores: [0.95, 0.89, 0.82, 0.78,<br/>0.72, 0.68, 0.45, 0.38,<br/>0.22, 0.11]"]

    FilterThreshold["Filter ≥ 0.15"]
    FilledScores["Filtered: [0.95, 0.89, 0.82,<br/>0.78, 0.72, 0.68, 0.45, 0.38, 0.22]"]

    Top4["Select Top 4"]
    Top4Scores["[0.95, 0.89, 0.82, 0.78]"]

    CalcAvg["Calculate Average<br/>avg = (0.95+0.89+0.82+0.78)/4<br/>= 3.44/4 = 0.86"]

    ConfidenceLevel{"Confidence<br/>Level?"}

    High["avg ≥ 0.9?<br/>→ High"]
    Medium["avg ≥ 0.8?<br/>→ Medium"]
    Low["avg < 0.8?<br/>→ Low"]

    HighResult["🟢 HIGH CONFIDENCE<br/>Use top 4 chunks<br/>Full context"]

    MediumResult["🟡 MEDIUM CONFIDENCE<br/>Use top 4 chunks<br/>Add disclaimer"]

    LowResult["🔴 LOW CONFIDENCE<br/>Use top 2 chunks<br/>Explicit warning"]

    GenerateAnswer["Generate Answer<br/>with confidence label"]

    Response["Return Response<br/>answer + confidence +<br/>sources + score"]

    Query --> RetrievalTop10
    RetrievalTop10 --> Scores
    Scores --> FilterThreshold
    FilterThreshold --> FilledScores
    FilledScores --> Top4
    Top4 --> Top4Scores
    Top4Scores --> CalcAvg
    CalcAvg --> ConfidenceLevel

    ConfidenceLevel -->|≥0.9| High
    ConfidenceLevel -->|0.8-0.9| Medium
    ConfidenceLevel -->|<0.8| Low

    High --> HighResult
    Medium --> MediumResult
    Low --> LowResult

    HighResult --> GenerateAnswer
    MediumResult --> GenerateAnswer
    LowResult --> GenerateAnswer

    GenerateAnswer --> Response

    style Query fill:#c8e6c9
    style RetrievalTop10 fill:#bbdefb
    style Scores fill:#90caf9
    style FilterThreshold fill:#64b5f6
    style FilledScores fill:#42a5f5
    style Top4 fill:#2196f3
    style Top4Scores fill:#1e88e5
    style CalcAvg fill:#1565c0
    style ConfidenceLevel fill:#fff9c4
    style High fill:#a5d6a7
    style Medium fill:#fff9c4
    style Low fill:#ffccbc
    style HighResult fill:#81c784
    style MediumResult fill:#ffe082
    style LowResult fill:#ffb74d
    style GenerateAnswer fill:#f8bbd0
    style Response fill:#c8e6c9
```

---

## Summary

These Mermaid diagrams provide:

✅ **System-level architecture** - All components and integrations
✅ **Data flow diagrams** - Upload and query pipelines
✅ **Database relationships** - ER diagram of all models
✅ **Authentication flow** - Registration to protected requests
✅ **Job processing** - Background worker loop with retries
✅ **Sequence diagrams** - Component interactions over time
✅ **Token management** - Budget checking and trimming logic
✅ **Scaling architecture** - Horizontal scaling strategy
✅ **Error handling** - Retry strategy with exponential backoff
✅ **Security layers** - Multi-layer validation approach
✅ **CI/CD pipeline** - From dev to production
✅ **Confidence scoring** - RAG confidence calculation

All diagrams are in **Mermaid format** and can be:
- Rendered in GitHub markdown
- Exported as PNG/SVG
- Used in presentations
- Embedded in documentation
