# LexIntel

Legal document analysis platform.

## Status

The backend and AI pipeline are being rebuilt from scratch. This repository currently contains:

- `frontend/` — Next.js app (UI)
- `test_cases/` — sample matters with labelled questions, used as the seed evaluation set

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI expects the API at `NEXT_PUBLIC_API_URL` (see `frontend/.env.example`). Until the new backend lands, API-backed screens will not load data.
