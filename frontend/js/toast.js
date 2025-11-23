// Toast Notification System for RFH
function showToast(message, type = 'info') {
    // Remove any existing toasts
    const existingToast = document.getElementById('rfh-toast');
    if (existingToast) {
        existingToast.remove();
    }

    // Create toast container
    const toast = document.createElement('div');
    toast.id = 'rfh-toast';
    toast.className = `rfh-toast rfh-toast-${type}`;

    // Icon based on type
    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };

    toast.innerHTML = `
        <div class="rfh-toast-icon">${icons[type] || icons.info}</div>
        <div class="rfh-toast-message">${message}</div>
    `;

    document.body.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('rfh-toast-show'), 10);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.classList.remove('rfh-toast-show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Add CSS styles
const toastStyles = document.createElement('style');
toastStyles.textContent = `
    .rfh-toast {
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 2px solid #d00000;
        border-radius: 0;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        min-width: 300px;
        max-width: 500px;
        box-shadow: 0 10px 40px rgba(208, 0, 0, 0.3);
        z-index: 10000;
        transform: translateX(120%);
        transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        font-family: 'Roboto', sans-serif;
    }
    
    .rfh-toast-show {
        transform: translateX(0);
    }
    
    .rfh-toast-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    .rfh-toast-success .rfh-toast-icon {
        background: #d00000;
        color: white;
    }
    
    .rfh-toast-error .rfh-toast-icon {
        background: #ff4444;
        color: white;
    }
    
    .rfh-toast-warning .rfh-toast-icon {
        background: #ffaa00;
        color: #000;
    }
    
    .rfh-toast-info .rfh-toast-icon {
        background: #333;
        color: #d00000;
        border: 2px solid #d00000;
    }
    
    .rfh-toast-message {
        color: #fff;
        font-size: 0.95rem;
        line-height: 1.5;
        flex-grow: 1;
    }
    
    .rfh-toast-success {
        border-color: #d00000;
    }
    
    .rfh-toast-error {
        border-color: #ff4444;
    }
    
    .rfh-toast-warning {
        border-color: #ffaa00;
    }
    
    .rfh-toast-info {
        border-color: #666;
    }
    
    @media (max-width: 768px) {
        .rfh-toast {
            right: 10px;
            left: 10px;
            min-width: auto;
        }
    }
`;
document.head.appendChild(toastStyles);
