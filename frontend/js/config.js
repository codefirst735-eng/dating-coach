// Development: Always use local backend
// Production: Switch back to production URL before deploying to main
// const API_BASE_URL = 'http://127.0.0.1:8001';

// Production deployment - using Render backend:
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8001'
    : 'https://dating-coach-ytos.onrender.com';
