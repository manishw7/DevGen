# DevGen Frontend

Vite + React frontend for the DevGen Devanagari OCR project.

## Run

```powershell
npm install
npm run dev
```

The app expects the backend at `http://localhost:8000` by default.

To override it, create `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

For full project setup, see the root [README](../README.md).
