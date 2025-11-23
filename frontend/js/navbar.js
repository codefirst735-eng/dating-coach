// Navbar dynamic link handler
window.initNavbar = function () {
    console.log('Navbar script loaded');
    const token = localStorage.getItem('access_token');
    const navbarNav = document.querySelector('.navbar-nav');

    if (!navbarNav) {
        console.error('Navbar element not found!');
        return;
    }

    console.log('Token:', token ? 'exists' : 'not found');

    // Remove any existing auth links to avoid duplicates
    const existingAuthLinks = navbarNav.querySelectorAll('.auth-link');
    existingAuthLinks.forEach(link => link.remove());

    if (token) {
        console.log('User is logged in');
        // User is logged in - show Profile and Logout
        const profileLi = document.createElement('li');
        profileLi.className = 'nav-item auth-link';
        profileLi.innerHTML = '<a class="nav-link" href="profile.html">Profile</a>';

        const logoutLi = document.createElement('li');
        logoutLi.className = 'nav-item auth-link';
        const logoutLink = document.createElement('a');
        logoutLink.className = 'nav-link';
        logoutLink.href = '#';
        logoutLink.textContent = 'Logout';
        logoutLink.addEventListener('click', (e) => {
            e.preventDefault();
            localStorage.removeItem('access_token');
            window.location.href = 'index.html';
        });
        logoutLi.appendChild(logoutLink);

        navbarNav.appendChild(profileLi);
        navbarNav.appendChild(logoutLi);
    } else {
        console.log('User is guest');
        // User is not logged in - show Login and Join Now
        const loginLi = document.createElement('li');
        loginLi.className = 'nav-item auth-link';
        loginLi.innerHTML = '<a class="nav-link" href="login.html">Login</a>';

        const signupLi = document.createElement('li');
        signupLi.className = 'nav-item auth-link';
        signupLi.innerHTML = '<a class="nav-link btn btn-danger rounded-0 px-3 ms-lg-2" href="signup.html">Join Now</a>';

        navbarNav.appendChild(loginLi);
        navbarNav.appendChild(signupLi);
    }
};

// Auto-initialize if navbar already exists (for backwards compatibility)
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.navbar-nav')) {
        window.initNavbar();
    }
});
