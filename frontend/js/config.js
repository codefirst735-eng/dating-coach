// Development: Always use local backend
// const API_BASE_URL = 'http://127.0.0.1:8000';

// Production: Switch back to production URL before deploying to main
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://python-backend-z8w6.onrender.com';
