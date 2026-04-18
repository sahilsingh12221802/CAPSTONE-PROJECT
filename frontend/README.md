# Frontend (React Dashboard)

User-facing dashboard for classification, project overview, and results presentation.

## Tech Stack

- React 18
- React Router
- Tailwind CSS
- Axios
- Nginx (for containerized static serving)

## Key Pages

- Home page
- Breeds overview page
- Breed details page
- Classify page (upload + inference result)
- Dashboard analytics page

## API Integration

Frontend calls backend classify and records endpoints.

Configuration source:

- Environment variable: `REACT_APP_API_BASE_URL`

Default fallback in app code:

- `http://localhost:8000`

## Local Development

```bash
npm install
npm start
```

Open:

- http://localhost:3000

## Production Build (Local)

```bash
npm run build
```

## Docker Deployment

- Build file: [frontend/Dockerfile](Dockerfile)
- Web server config: [frontend/nginx.conf](nginx.conf)
- Container serves static app at port `80`

Compose exposes frontend at:

- http://localhost

## Troubleshooting

- If classification shows network error, verify backend is running on `http://localhost:8000`.
- If routes 404 in container mode, confirm Nginx config includes SPA fallback to `index.html`.
- If environment URL is wrong, rebuild frontend image because env is baked at build time.
