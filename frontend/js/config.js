// Development: Always use local backend
// const API_BASE_URL = 'http://127.0.0.1:8000';

// Production: Point to Render backend
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://rfh-backend.onrender.com';
