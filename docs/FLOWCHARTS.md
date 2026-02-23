# LexIntel Process Flowcharts

This document contains comprehensive Mermaid process flowcharts for all critical flows in the LexIntel system. Each diagram illustrates decision points, error handling, and system interactions for key operations.

---

## 1. Document Upload & Processing Flow

This flow covers the complete lifecycle of a document from user upload through background processing, including Azure Blob storage, Redis queue, Celery workers, text extraction, chunking, embeddings, and vector store storage.

```mermaid
graph TD
    Start([User Upload Request]) --> Auth{Auth Token<br/>Valid?}
    Auth -->|No| AuthError["401 Unauthorized<br/>Return Error"]
    Auth -->|Yes| Validate{File Format<br/>Valid?}

    Validate -->|Invalid| FormatError["400 Bad Request<br/>Return Format Error"]
    Validate -->|Valid| CheckDB{"Case Name<br/>Already<br/>Exists?"}

    CheckDB -->|Yes| DuplicateError["400 Conflict<br/>Return Duplicate Error"]
    CheckDB -->|No| CreateCase["Create Case Record<br/>status: pending<br/>user_id: authenticated_user"]

    CreateCase --> Upload["Upload File to<br/>Azure Blob Storage<br/>blob_storage_path: set"]
    Upload --> UploadSuccess{Upload<br/>Success?}

    UploadSuccess -->|Error| UploadError["Update Case Status<br/>failed<br/>Return Error"]
    UploadSuccess -->|Success| QueueJob["Queue Job to Redis<br/>process_document_task<br/>case_id: UUID"]

    QueueJob --> UpdatePending["Update Case Status<br/>queued"]
    UpdatePending --> RespondAsync["Return 202 Accepted<br/>case_id, status, timestamp"]

    RespondAsync --> CeleryWorker["⟹ Celery Worker<br/>Dequeues Job"]
    CeleryWorker --> UpdateProcessing["Update Case Status<br/>processing"]

    UpdateProcessing --> Download["Download Document<br/>from Azure Blob"]
    Download --> DownloadOK{Download<br/>Success?}

    DownloadOK -->|Fail| Retry{Retry Count<br/>< 3?}
    Retry -->|Yes| BackoffWait["Wait 5s<br/>Attempt: N+1"]
    BackoffWait --> Download
    Retry -->|No| ProcessingFailed["Update Case Status<br/>failed<br/>Log Error"]

    DownloadOK -->|Success| DetectType{"File Type<br/>Detected"}
    DetectType -->|PDF| ExtractPDF["Text Extraction<br/>Using Docling<br/>PDF Handler"]
    DetectType -->|DOCX| ExtractDOCX["Text Extraction<br/>Using Docling<br/>DOCX Handler"]
    DetectType -->|TXT| ExtractTXT["Text Extraction<br/>Read UTF-8<br/>Encoding"]
    DetectType -->|Unknown| ExtractionError["Log Error<br/>Mark Case: failed"]

    ExtractPDF --> ChunkContent["Chunk Content<br/>Overlapping Windows<br/>chunk_size: 512"]
    ExtractDOCX --> ChunkContent
    ExtractTXT --> ChunkContent

    ChunkContent --> ChunkOK{Chunks<br/>Generated?}
    ChunkOK -->|None| NoChunksError["Log Error<br/>Mark Case: failed"]
    ChunkOK -->|Yes| GenerateEmbed["Generate Embeddings<br/>Call Google AI Embedding API<br/>model: gemini-embedding-001"]

    GenerateEmbed --> EmbedOK{Embeddings<br/>Success?}
    EmbedOK -->|Fail| EmbedRetry{Retry Count<br/>< 3?}
    EmbedRetry -->|Yes| BackoffEmbed["Wait 10s<br/>Attempt: N+1"]
    BackoffEmbed --> GenerateEmbed
    EmbedRetry -->|No| EmbedFailed["Log Error<br/>Mark Case: failed"]

    EmbedOK -->|Success| CreateQdrant["Create Qdrant<br/>Collection<br/>collection: case_id"]
    CreateQdrant --> UpsertVectors["Upsert Vectors to Qdrant<br/>chunk_id, embedding, metadata<br/>vector_size: 768"]

    UpsertVectors --> UpsertOK{Upsert<br/>Success?}
    UpsertOK -->|Fail| UpsertRetry{Retry Count<br/>< 3?}
    UpsertRetry -->|Yes| BackoffUpsert["Wait 5s<br/>Attempt: N+1"]
    BackoffUpsert --> UpsertVectors
    UpsertRetry -->|No| UpsertFailed["Log Error<br/>Mark Case: failed"]

    UpsertOK -->|Success| StoreChunks["Store Chunk Metadata<br/>in PostgreSQL<br/>chunk_id, case_id, content<br/>page_number, position"]

    StoreChunks --> StoreOK{Store<br/>Success?}
    StoreOK -->|Fail| StoreFailed["Log Error<br/>Mark Case: failed"]
    StoreOK -->|Success| UpdateComplete["Update Case Status<br/>completed<br/>processed_at: timestamp"]

    UpdateComplete --> LogSuccess["Log: Case Processing Complete<br/>case_id, chunk_count, timestamp"]
    LogSuccess --> End([Process Complete])

    AuthError --> EndError([Return to User])
    FormatError --> EndError
    DuplicateError --> EndError
    UploadError --> EndError
    ProcessingFailed --> EndError
    ExtractionError --> EndError
    NoChunksError --> EndError
    EmbedFailed --> EndError
    UpsertFailed --> EndError
    StoreFailed --> EndError

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style EndError fill:#FF6B6B
    style AuthError fill:#FF8C8C
    style FormatError fill:#FF8C8C
    style DuplicateError fill:#FF8C8C
    style UploadError fill:#FF8C8C
    style ProcessingFailed fill:#FF8C8C
    style ExtractionError fill:#FF8C8C
    style NoChunksError fill:#FF8C8C
    style EmbedFailed fill:#FF8C8C
    style UpsertFailed fill:#FF8C8C
    style StoreFailed fill:#FF8C8C
    style CeleryWorker fill:#87CEEB
```

### Key Decision Points:
- **Auth Token Validation**: Ensures user is authenticated before accepting upload
- **File Format Validation**: Checks magic bytes (PDF, DOCX, TXT) to prevent malicious files
- **Duplicate Detection**: Prevents duplicate case names per user
- **Extraction Type Selection**: Routes to appropriate text extraction handler based on file type
- **Chunk Generation**: Validates that text extraction produced chunks
- **Retry Logic**: 3 retry attempts with exponential backoff (5s, 10s) for API calls

### Error Paths:
- Authentication failures return 401
- Format/validation errors return 400
- Upload errors are caught and case marked as failed
- Download, extraction, embedding, and storage failures trigger retry logic
- Final failure after max retries marks case as failed and logs error

---

## 2. Query & Answer Generation Flow

This flow demonstrates the complete Q&A pipeline: query validation, embedding generation, vector search, context formatting, LLM inference, and citation extraction.

```mermaid
graph TD
    Start([User Query Request]) --> Auth{Auth Token<br/>Valid?}
    Auth -->|No| AuthError["401 Unauthorized<br/>Return Error"]
    Auth -->|Yes| ValidateQuery{Query Length<br/>>= 3 chars?}

    ValidateQuery -->|No| QueryError["400 Bad Request<br/>Min 3 characters"]
    ValidateQuery -->|Yes| CheckCase{Case Found<br/>& Owned?}

    CheckCase -->|No| CaseError["404 Not Found<br/>or 403 Forbidden"]
    CheckCase -->|Yes| CheckStatus{"Case Status<br/>== completed?"}

    CheckStatus -->|No| StatusError["400 Bad Request<br/>Case Not Ready"]
    CheckStatus -->|Yes| StoreQuery["Store Query in DB<br/>case_id, user_id<br/>status: pending"]

    StoreQuery --> EmbedQuery["Generate Query Embedding<br/>Call Google AI Embedding API<br/>model: gemini-embedding-001"]

    EmbedQuery --> EmbedOK{Embedding<br/>Success?}
    EmbedOK -->|Fail| EmbedRetry{Retry Count<br/>< 3?}
    EmbedRetry -->|Yes| BackoffEmbed["Wait 10s<br/>Attempt: N+1"]
    BackoffEmbed --> EmbedQuery
    EmbedRetry -->|No| EmbedFailed["Update Query Status<br/>failed<br/>Return Error"]

    EmbedOK -->|Success| VectorSearch["Vector Search in Qdrant<br/>collection: case_id<br/>top_k: 10<br/>min_score: 0.6"]

    VectorSearch --> SearchOK{Results<br/>Found?}
    SearchOK -->|No| NoResults["Update Query Status<br/>completed<br/>answer: empty, citations: []"]
    SearchOK -->|Yes| RetrieveChunks["Retrieve Chunk Details<br/>from PostgreSQL<br/>chunk_id, content, page"]

    RetrieveChunks --> RerankChunks["Rerank Retrieved Chunks<br/>Using Similarity Scoring<br/>Select Top 4 Most Relevant"]

    RerankChunks --> FormatContext["Format Context String<br/>Concatenate Chunks<br/>Max Tokens: 12,800<br/>Include Citations: Page X"]

    FormatContext --> CountTokens["Count Tokens<br/>Using token estimation<br/>for Gemini"]

    CountTokens --> CheckTokens{Tokens<br/><= 12,800?}
    CheckTokens -->|No| TruncateContext["Truncate Context<br/>Keep Most Recent Chunks"]
    CheckTokens -->|Yes| ReadyContext["Context Ready"]
    TruncateContext --> ReadyContext

    ReadyContext --> CallLLM["Call Gemini API<br/>system: legal_prompt<br/>user: query+context<br/>max_tokens: 2000"]

    CallLLM --> LLMOk{LLM<br/>Success?}
    LLMOk -->|Fail| LLMRetry{Retry Count<br/>< 3?}
    LLMRetry -->|Yes| BackoffLLM["Wait 10s<br/>Attempt: N+1"]
    BackoffLLM --> CallLLM
    LLMRetry -->|No| LLMFailed["Update Query Status<br/>failed<br/>Return Error"]

    LLMOk -->|Success| ExtractAnswer["Extract Answer Text<br/>from LLM Response"]

    ExtractAnswer --> ExtractCitations["Extract Citations<br/>Parse [Page X] references<br/>Match with Context"]

    ExtractCitations --> ValidateCitations{"Citations<br/>Valid &<br/>Grounded?"}

    ValidateCitations -->|No| LogWarning["Log Citation Warning<br/>Store Partial Results"]
    ValidateCitations -->|Yes| ValidatedCitations["Citations Validated"]

    LogWarning --> ValidatedCitations
    ValidatedCitations --> StoreAnswer["Store Query Result<br/>answer: text<br/>citations: list<br/>context_used: chunks"]

    StoreAnswer --> UpdateQuery["Update Query Status<br/>completed<br/>completed_at: timestamp"]

    UpdateQuery --> FormatResponse["Format Response<br/>answer, citations<br/>case_id, query_id"]

    FormatResponse --> ReturnSuccess["Return 200 OK<br/>Answer + Citations"]
    ReturnSuccess --> End([Process Complete])

    AuthError --> EndError([Return Error])
    QueryError --> EndError
    CaseError --> EndError
    StatusError --> EndError
    EmbedFailed --> EndError
    NoResults --> EndFinal([Return Empty Result])
    LLMFailed --> EndError

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style EndError fill:#FF6B6B
    style EndFinal fill:#FFD700
    style AuthError fill:#FF8C8C
    style QueryError fill:#FF8C8C
    style CaseError fill:#FF8C8C
    style StatusError fill:#FF8C8C
    style EmbedFailed fill:#FF8C8C
    style LLMFailed fill:#FF8C8C
    style VectorSearch fill:#87CEEB
    style CallLLM fill:#87CEEB
    style RerankChunks fill:#DDA0DD
```

### Key Decision Points:
- **Query Validation**: Enforces minimum length (3 characters)
- **Case Ownership**: Ensures user owns the case being queried
- **Case Status Check**: Only allows queries on fully processed cases
- **Embedding Success**: Retries with backoff if embedding API fails
- **Vector Search**: Minimum confidence threshold (0.6) filters weak matches
- **Token Budget**: Ensures context fits within 12,800 token budget for Gemini
- **Citation Extraction**: Parses and validates citations are grounded in context
- **Result Handling**: Returns empty results if no relevant chunks found

### Error Paths:
- Authentication and validation errors return 4xx HTTP responses
- Embedding failures trigger retry logic (3 attempts, 10s backoff)
- LLM failures trigger retry logic (3 attempts, 10s backoff)
- Citation validation warnings are logged but don't block response
- Final success includes answer and validated citations

---

## 3. User Authentication Flow

This flow covers user registration, login, and protected request authentication using JWT tokens with bcrypt password hashing and 24-hour expiry.

```mermaid
graph TD
    Start([Client Request]) --> CheckEndpoint{Endpoint Type}

    CheckEndpoint -->|Register| RegStart["POST /register<br/>Receive UserCreate"]
    CheckEndpoint -->|Login| LoginStart["POST /login<br/>Receive UserCreate"]
    CheckEndpoint -->|Protected| ProtectedStart["GET /profile, /cases<br/>or POST /ask<br/>Receive Authorization"]

    RegStart --> ValidateEmail{Email Format<br/>Valid?}
    ValidateEmail -->|No| EmailError["400 Bad Request<br/>Invalid Email Format"]
    ValidateEmail -->|Yes| CheckEmailExists{Email Already<br/>Exists?}

    CheckEmailExists -->|Yes| DuplicateEmail["400 Conflict<br/>Email Already Registered"]
    CheckEmailExists -->|No| ValidatePassword{Password<br/>Valid?}

    ValidatePassword -->|No| PasswordError["400 Bad Request<br/>Password Requirements Not Met"]
    ValidatePassword -->|Yes| HashPassword["Hash Password<br/>Using bcrypt<br/>algorithm: bcrypt"]

    HashPassword --> CreateUser["Create User Record<br/>email, hashed_password<br/>created_at: timestamp"]

    CreateUser --> CreateOK{User Created<br/>Success?}
    CreateOK -->|Fail| CreateError["500 Internal Error<br/>Database Error"]
    CreateOK -->|Yes| RegSuccess["Return 201 Created<br/>user_id, email"]
    RegSuccess --> EndReg([Registration Complete])

    LoginStart --> ValidateInput{Email & Password<br/>Provided?}
    ValidateInput -->|No| InputError["400 Bad Request<br/>Missing Credentials"]
    ValidateInput -->|Yes| FindUser["Query User by Email<br/>from PostgreSQL"]

    FindUser --> UserExists{User<br/>Found?}
    UserExists -->|No| LoginFail["401 Unauthorized<br/>Invalid Credentials"]
    UserExists -->|Yes| VerifyPassword["Verify Password<br/>Compare plain vs<br/>hashed using bcrypt"]

    VerifyPassword --> PassMatch{Password<br/>Correct?}
    PassMatch -->|No| LoginFail
    PassMatch -->|Yes| CreateToken["Create JWT Token<br/>Payload: sub=user_id<br/>Expires: 24 hours"]

    CreateToken --> SignToken["Sign Token<br/>Algorithm: HS256<br/>Secret: SECRET_KEY"]
    SignToken --> ReturnToken["Return 200 OK<br/>access_token, token_type"]
    ReturnToken --> EndLogin([Login Complete])

    ProtectedStart --> GetAuthHeader["Extract Authorization<br/>Header<br/>Format: Bearer TOKEN"]

    GetAuthHeader --> HeaderOK{Header<br/>Present &<br/>Valid Format?}
    HeaderOK -->|No| HeaderError["401 Unauthorized<br/>Missing or Invalid Header"]
    HeaderOK -->|Yes| ExtractToken["Extract Token<br/>Remove 'Bearer ' prefix"]

    ExtractToken --> DecodeToken["Decode JWT Token<br/>Verify Signature<br/>Algorithm: HS256"]

    DecodeToken --> DecodeOK{Token Valid<br/>& Not Expired?}
    DecodeOK -->|Invalid| InvalidToken["401 Unauthorized<br/>Invalid/Expired Token"]
    DecodeOK -->|Expired| ExpiredToken["401 Unauthorized<br/>Token Expired"]
    DecodeOK -->|Valid| ExtractUserID["Extract user_id<br/>from 'sub' claim"]

    ExtractUserID --> GetUser["Query User by ID<br/>from PostgreSQL"]

    GetUser --> UserOK{User<br/>Found?}
    UserOK -->|No| UserNotFound["401 Unauthorized<br/>User Not Found"]
    UserOK -->|Yes| GrantAccess["Grant Access<br/>Set current_user<br/>Inject to Endpoint"]

    GrantAccess --> CallEndpoint["Execute Protected<br/>Endpoint Logic<br/>with user_id"]

    CallEndpoint --> EndSuccess([Request Complete])

    EmailError --> EndReg
    DuplicateEmail --> EndReg
    PasswordError --> EndReg
    CreateError --> EndReg

    InputError --> EndLogin
    LoginFail --> EndLogin

    HeaderError --> EndProtected([Return 401])
    InvalidToken --> EndProtected
    ExpiredToken --> EndProtected
    UserNotFound --> EndProtected

    style Start fill:#90EE90
    style EndReg fill:#FFB6C1
    style EndLogin fill:#FFB6C1
    style EndSuccess fill:#FFB6C1
    style EndProtected fill:#FF6B6B
    style EmailError fill:#FF8C8C
    style DuplicateEmail fill:#FF8C8C
    style PasswordError fill:#FF8C8C
    style CreateError fill:#FF8C8C
    style InputError fill:#FF8C8C
    style LoginFail fill:#FF8C8C
    style HeaderError fill:#FF8C8C
    style InvalidToken fill:#FF8C8C
    style ExpiredToken fill:#FF8C8C
    style UserNotFound fill:#FF8C8C
    style HashPassword fill:#87CEEB
    style CreateToken fill:#87CEEB
    style DecodeToken fill:#87CEEB
```

### Key Decision Points:
- **Registration Path**:
  - Email format validation (RFC 5322)
  - Duplicate email detection
  - Password strength validation
  - Bcrypt hashing with salt rounds

- **Login Path**:
  - Email existence check
  - Password verification using bcrypt.verify()
  - JWT creation with 24-hour expiry
  - Token signing with HS256 algorithm

- **Protected Request Path**:
  - Authorization header presence and format
  - Token decoding and signature verification
  - Token expiration check
  - User record lookup
  - Access grant to endpoint

### Error Paths:
- Validation errors return 400 Bad Request
- Duplicate email returns 409 Conflict
- Invalid/expired token returns 401 Unauthorized
- User not found returns 401 Unauthorized
- Database errors return 500 Internal Server Error

---

## 4. Case Management Flow

This flow covers case list retrieval, single case viewing, and document upload operations with ownership validation and pagination.

```mermaid
graph TD
    Start([Case Management Request]) --> Auth{Auth Token<br/>Valid?}
    Auth -->|No| AuthError["401 Unauthorized"]
    Auth -->|Yes| GetUserID["Extract user_id<br/>from JWT token"]

    GetUserID --> CheckOp{Operation Type}

    CheckOp -->|List Cases| ListStart["GET /cases<br/>Optional: ?skip=0&limit=10"]
    CheckOp -->|View Case| ViewStart["GET /cases/{case_id}"]
    CheckOp -->|Upload Case| UploadStart["POST /cases/upload"]

    ListStart --> QueryCases["Query Cases from<br/>PostgreSQL<br/>WHERE user_id = current"]

    QueryCases --> ApplyFilter{Filter/Search<br/>Provided?}
    ApplyFilter -->|Yes| FilterCases["Apply WHERE Conditions<br/>status, name, date"]
    ApplyFilter -->|No| NoFilter["Use All Cases"]

    FilterCases --> NoFilter
    NoFilter --> ApplyPagination["Apply Pagination<br/>skip, limit params<br/>Default: limit=10"]

    ApplyPagination --> FetchCount["Fetch Total Count<br/>Query COUNT(*)"]

    FetchCount --> FetchRecords["Fetch Case Records<br/>With Pagination<br/>ORDER BY created_at DESC"]

    FetchRecords --> BuildResponse["Build Response<br/>items: []<br/>total: count<br/>skip: offset<br/>limit: limit"]

    BuildResponse --> ReturnList["Return 200 OK<br/>Case List + Metadata"]
    ReturnList --> EndList([List Operation Complete])

    ViewStart --> ValidateID{Case ID<br/>Valid UUID?}
    ValidateID -->|No| IDError["400 Bad Request<br/>Invalid UUID Format"]
    ValidateID -->|Yes| QueryCase["Query Case by ID<br/>FROM PostgreSQL"]

    QueryCase --> CaseFound{Case<br/>Found?}
    CaseFound -->|No| NotFound["404 Not Found"]
    CaseFound -->|Yes| CheckOwner{Case Owner ==<br/>current_user?}

    CheckOwner -->|No| Forbidden["403 Forbidden<br/>Not Case Owner"]
    CheckOwner -->|Yes| QueryChunks["Query Chunk Count<br/>Count chunks for case"]

    QueryChunks --> BuildCaseDetail["Build Case Detail<br/>case, status<br/>chunk_count, created_at<br/>processed_at"]

    BuildCaseDetail --> ReturnDetail["Return 200 OK<br/>Complete Case Object"]
    ReturnDetail --> EndView([View Operation Complete])

    UploadStart --> ValidateName{Case Name<br/>Provided?}
    ValidateName -->|No| NameError["400 Bad Request<br/>Name Required"]
    ValidateName -->|Yes| ValidateFile{File<br/>Provided?}

    ValidateFile -->|No| FileError["400 Bad Request<br/>File Required"]
    ValidateFile -->|Yes| CheckNameExists{Case Name<br/>Exists for User?}

    CheckNameExists -->|Yes| NameDuplicate["400 Conflict<br/>Name Already Used"]
    CheckNameExists -->|No| ValidateFileType{File Type<br/>Supported?}

    ValidateFileType -->|No| TypeError["400 Bad Request<br/>File Type Not Supported<br/>Allowed: PDF, DOCX, TXT"]
    ValidateFileType -->|Yes| ReadFile["Read Uploaded File<br/>Read file bytes<br/>Check size limit"]

    ReadFile --> SizeCheck{File Size<br/><= Limit?}
    SizeCheck -->|No| SizeError["413 Payload Too Large<br/>Max 50MB"]
    SizeCheck -->|Yes| CheckMagic["Validate Magic Bytes<br/>Ensure File Type Match"]

    CheckMagic --> MagicOK{File Content<br/>Matches Type?}
    MagicOK -->|No| MagicError["400 Bad Request<br/>File Content Doesn't Match Type"]
    MagicOK -->|Yes| CreateCaseRec["Create Case Record<br/>status: pending<br/>user_id, name<br/>file_type, blob_path"]

    CreateCaseRec --> CreateOK{Case Created?}
    CreateOK -->|Fail| CreateError["500 Internal Error"]
    CreateOK -->|Success| UploadBlob["Upload File to<br/>Azure Blob Storage"]

    UploadBlob --> BlobOK{Upload<br/>Success?}
    BlobOK -->|Fail| BlobError["500 Internal Error<br/>Blob Upload Failed"]
    BlobOK -->|Success| UpdatePath["Update Case<br/>blob_storage_path"]

    UpdatePath --> QueueJob["Queue Celery Job<br/>process_document_task"]

    QueueJob --> UpdateStatus["Update Case Status<br/>queued"]

    UpdateStatus --> ReturnAsync["Return 202 Accepted<br/>case_id, status"]
    ReturnAsync --> EndUpload([Upload Operation Complete])

    AuthError --> EndError([Return 401])
    IDError --> EndError
    NotFound --> EndError
    Forbidden --> EndError
    NameError --> EndError
    FileError --> EndError
    NameDuplicate --> EndError
    TypeError --> EndError
    SizeError --> EndError
    MagicError --> EndError
    CreateError --> EndError
    BlobError --> EndError

    style Start fill:#90EE90
    style EndList fill:#FFB6C1
    style EndView fill:#FFB6C1
    style EndUpload fill:#FFB6C1
    style EndError fill:#FF6B6B
    style AuthError fill:#FF8C8C
    style IDError fill:#FF8C8C
    style NotFound fill:#FF8C8C
    style Forbidden fill:#FF8C8C
    style NameError fill:#FF8C8C
    style FileError fill:#FF8C8C
    style NameDuplicate fill:#FF8C8C
    style TypeError fill:#FF8C8C
    style SizeError fill:#FF8C8C
    style MagicError fill:#FF8C8C
    style CreateError fill:#FF8C8C
    style BlobError fill:#FF8C8C
```

### Key Decision Points:
- **List Cases**:
  - Query all cases for authenticated user
  - Apply optional filters and pagination
  - Return with total count metadata

- **View Case**:
  - Validate UUID format of case_id
  - Verify case exists
  - Ensure user owns the case
  - Include chunk count in response

- **Upload Case**:
  - Validate name and file provided
  - Check name uniqueness per user
  - Validate file type (PDF, DOCX, TXT)
  - Check file size (max 50MB)
  - Validate magic bytes match file type
  - Create case record and upload to blob
  - Queue async processing job

### Error Paths:
- Authentication errors return 401
- Invalid input (name, file) returns 400
- Duplicate name returns 409 Conflict
- Not found or forbidden returns 404/403
- File validation errors return 400
- Upload failures return 500
- Async job queuing returns 202 Accepted

---

## 5. Embedding Cache Flow

This flow demonstrates the in-memory LRU cache for embeddings, including cache hits, misses, API calls, and storage with automatic eviction.

```mermaid
graph TD
    Start([Request Embedding]) --> NeedEmbed["Need Embedding<br/>for chunk_id"]

    NeedEmbed --> CheckCache{"Embedding<br/>in Cache?"}

    CheckCache -->|Yes - Cache Hit| GetCache["Retrieve from<br/>In-Memory Cache<br/>OrderedDict"]

    GetCache --> MoveRecent["Move Item to End<br/>Mark as Recently Used"]

    MoveRecent --> UpdateHit["Update Cache Stats<br/>hits += 1"]

    UpdateHit --> ReturnCached["Return Cached<br/>Embedding"]
    ReturnCached --> EndHit([Hit - Use Cached])

    CheckCache -->|No - Cache Miss| UpdateMiss["Update Cache Stats<br/>misses += 1"]

    UpdateMiss --> CheckDB{"Embedding<br/>in Database?"}

    CheckDB -->|Yes| RetrieveDB["Retrieve from<br/>PostgreSQL<br/>embedding column"]
    RetrieveDB --> CacheMiss["Cache Retrieved<br/>Embedding"]
    CacheMiss --> ReturnDB["Return Embedding"]
    ReturnDB --> EndDBHit([Database Hit])

    CheckDB -->|No| CallAPI["Call Google AI API<br/>gemini-embedding-001<br/>input: chunk_content"]

    CallAPI --> APISuccess{API Call<br/>Success?}

    APISuccess -->|Fail| APIError{Retry Count<br/>< 3?}
    APIError -->|Yes| BackoffAPI["Wait Backoff<br/>5s, 10s, 10s"]
    BackoffAPI --> CallAPI
    APIError -->|No| APIFailed["Return Error<br/>Raise Exception"]
    APIFailed --> EndAPIFail([API Error])

    APISuccess -->|Success| ExtractEmbed["Extract Embedding<br/>Vector 768-dim<br/>numpy array"]

    ExtractEmbed --> StoreDB["Store in PostgreSQL<br/>UPDATE chunk SET<br/>embedding = vector"]

    StoreDB --> StoreOK{Store<br/>Success?}
    StoreOK -->|Fail| StoreFail["Log Error<br/>Continue (Cache Only)"]
    StoreOK -->|Success| DBStored["Embedding Persisted"]

    StoreFail --> DBStored

    DBStored --> CheckSize{"Cache Size<br/>< Max?"}

    CheckSize -->|Yes| AddCache["Add to Cache<br/>cache[chunk_id] = embedding"]
    CheckSize -->|No| Evict["Evict LRU Item<br/>pop(last=False)<br/>Remove oldest"]

    Evict --> EvictLog["Log Eviction<br/>evicted_key"]
    EvictLog --> AddCache

    AddCache --> ReturnAPI["Return Embedding"]
    ReturnAPI --> EndAPIDone([API Success])

    style Start fill:#90EE90
    style EndHit fill:#FFB6C1
    style EndDBHit fill:#FFD700
    style EndAPIDone fill:#FFB6C1
    style EndAPIFail fill:#FF6B6B
    style GetCache fill:#87CEEB
    style CallAPI fill:#87CEEB
    style Evict fill:#DDA0DD
```

### Key Decision Points:
- **Cache Check**: Fast lookup in OrderedDict for recently used embeddings
- **Hit Path**: Move accessed item to end (most recent), increment hit counter, return
- **Miss Path**: Check PostgreSQL for previously computed embeddings
- **API Call Path**: If not cached or stored, call Google AI with retry logic
- **Size Management**: Evict LRU (oldest) item when cache reaches max_size
- **Persistence**: Always store computed embeddings in PostgreSQL for future hits

### Cache Stats:
- Track hits and misses for cache performance monitoring
- Calculate hit rate: hits / (hits + misses)
- Monitor eviction frequency to tune cache size

### Error Paths:
- API failures trigger 3 retry attempts with backoff
- Database storage failures are logged but don't block cache operations
- API final failure returns exception to caller
- Cache operations are designed to be fast with no blocking I/O

---

## 6. Retry & Error Recovery Flow

This flow demonstrates the comprehensive retry strategy with exponential backoff, max attempt limits, logging, and final error handling for transient failures in document processing and API calls.

```mermaid
graph TD
    Start([Operation Initiated]) --> CallOp["Execute Operation<br/>download, embed,<br/>upsert, or LLM call"]

    CallOp --> OpResult{Operation<br/>Success?}

    OpResult -->|Success| LogSuccess["Log: Operation Succeeded<br/>Operation, duration"]
    LogSuccess --> ReturnSuccess["Return Result<br/>to Caller"]
    ReturnSuccess --> End([Operation Complete])

    OpResult -->|Failure| CatchError["Catch Exception<br/>GoogleAIError, AzureError<br/>Qdrant, Database Error"]

    CatchError --> LogError["Log Error<br/>Attempt N, operation<br/>error_type, message"]

    LogError --> CheckRetry{Attempt Count<br/>< Max Retries<br/>3?}

    CheckRetry -->|No| FinalFail["Log: Max Retries Exceeded<br/>operation_id, error"]

    FinalFail --> RaiseError["Raise Exception<br/>Return Error to Caller"]
    RaiseError --> EndFail([Final Failure])

    CheckRetry -->|Yes| IncrAttempt["Increment Attempt Counter<br/>attempt = N + 1"]

    IncrAttempt --> CalcBackoff["Calculate Backoff Delay<br/>Attempt 1: 0s<br/>Attempt 2: 5s<br/>Attempt 3: 10s"]

    CalcBackoff --> LogBackoff["Log: Retrying<br/>attempt N, waiting Xs<br/>operation_id"]

    LogBackoff --> WaitDelay["Wait Backoff Duration<br/>asyncio.sleep(delay_seconds)"]

    WaitDelay --> LogRetry["Log: Retry Attempt Starting<br/>attempt N, operation"]

    LogRetry --> RetryOp["Execute Operation Again<br/>Same parameters<br/>state unchanged"]

    RetryOp --> RetryResult{Retry<br/>Success?}

    RetryResult -->|Success| LogRetrySuccess["Log: Retry Succeeded<br/>Attempt N, operation"]
    LogRetrySuccess --> ReturnSuccess

    RetryResult -->|Failure| CatchRetryError["Catch Exception<br/>from Retry Attempt"]
    CatchRetryError --> LogError

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style EndFail fill:#FF6B6B
    style CallOp fill:#87CEEB
    style RetryOp fill:#87CEEB
    style WaitDelay fill:#DDA0DD
    style IncrAttempt fill:#FFD700
    style CalcBackoff fill:#FFD700
    style FinalFail fill:#FF8C8C
    style RaiseError fill:#FF8C8C
```

### Retry Strategy:
- **Max Attempts**: 3 total attempts (1 initial + 2 retries)
- **Backoff Schedule**:
  - Attempt 1: Immediate execution
  - Attempt 2: Wait 5 seconds
  - Attempt 3: Wait 10 seconds
  - After 3rd failure: Raise exception

- **Operations with Retry Logic**:
  - Document download from Azure Blob
  - Embedding generation via Google AI API
  - Vector upsert to Qdrant
  - LLM call to Gemini
  - Query embedding generation

### Error Handling:
- **Transient Errors**: Network timeouts, rate limits, temporary service issues
- **Logging**: Each attempt and backoff duration logged for debugging
- **State Management**: No state changes during retries; clean idempotency
- **Final Failure**: After max retries, exception raised and case/query marked failed

### Conditions for Retry:
- **Retry**: Rate limit (429), timeout, temporary connection failure
- **No Retry**: Invalid input (400), authentication failure (401), not found (404), forbidden (403), malformed content

---

## Integration Notes

All flowcharts integrate as follows:

1. **Upload Flow** → **Document Processing Flow**: Queues job on successful upload
2. **Query Flow** → **Embedding Cache Flow**: Uses cached embeddings when available
3. **Query Flow** → **Retry Flow**: Retries failed embedding and LLM calls
4. **Case Management** → **Upload Flow**: Validates ownership before allowing upload
5. **Auth Flow** → **All Protected Flows**: Validates JWT token on all requests
6. **Processing Flow** → **Retry Flow**: Retries all external API calls with backoff

---

## Mermaid Rendering Notes

These diagrams use the following Mermaid syntax features:

- **Graph Type**: Flowchart (graph TD = top-down)
- **Node Types**:
  - Round rectangle `([text])`: Start/end nodes
  - Square `[text]`: Process nodes
  - Diamond `{text}`: Decision nodes
  - Rectangular `["text"]`: Multi-line process

- **Styling**:
  - Green fill: Start nodes
  - Pink fill: Success end nodes
  - Red fill: Error end nodes
  - Light blue: Operation/API nodes
  - Purple: Cache/backoff nodes
  - Yellow: State transition nodes

All diagrams are valid Mermaid syntax and should render correctly in GitHub Markdown, GitLab, Notion, and other Mermaid-supporting platforms.
