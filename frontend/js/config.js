// Development: Use local backend (comment out before deploying)
// const API_BASE_URL = 'http://127.0.0.1:8000';

// Production: Point to actual Render backend URL
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : 'https://dating-coach-ytos.onrender.com';
