document.addEventListener('DOMContentLoaded', () => {
    console.log('Navbar script loaded');
    const token = localStorage.getItem('access_token');
    const navbarNav = document.querySelector('.navbar-nav');

    if (!navbarNav) {
        console.error('Navbar element not found!');
        return;
    }

    if (token) {
        // User is logged in
        console.log('User is logged in');

        // Remove existing login/signup links if they exist (e.g. from static HTML)
        const existingLogin = document.querySelector('a[href="login.html"]');
        if (existingLogin) existingLogin.closest('li').remove();

        const existingSignup = document.querySelector('a[href="signup.html"]');
        if (existingSignup) existingSignup.closest('li').remove();

        // Check if "Profile" already exists to avoid duplicates
        if (!document.querySelector('a[href="profile.html"]')) {
            const profileLi = document.createElement('li');
            profileLi.className = 'nav-item';
            profileLi.innerHTML = '<a class="nav-link" href="profile.html">Profile</a>';
            navbarNav.appendChild(profileLi);

            const logoutLi = document.createElement('li');
            logoutLi.className = 'nav-item';
            logoutLi.innerHTML = '<a class="nav-link" href="#" id="navLogout">Logout</a>';
            navbarNav.appendChild(logoutLi);

            document.getElementById('navLogout').addEventListener('click', (e) => {
                e.preventDefault();
                localStorage.removeItem('access_token');
                window.location.href = 'index.html';
            });
        }
    } else {
        // User is guest
        console.log('User is guest');

        // Remove Profile/Logout if they exist
        const existingProfile = document.querySelector('a[href="profile.html"]');
        if (existingProfile) existingProfile.closest('li').remove();

        if (!document.querySelector('a[href="login.html"]')) {
            const loginLi = document.createElement('li');
            loginLi.className = 'nav-item';
            loginLi.innerHTML = '<a class="nav-link" href="login.html">Login</a>';
            navbarNav.appendChild(loginLi);

            const signupLi = document.createElement('li');
            signupLi.className = 'nav-item';
            signupLi.innerHTML = '<a class="nav-link btn btn-danger px-3 ms-2 text-white rounded-0" href="signup.html">Join Now</a>';
            navbarNav.appendChild(signupLi);
        }
    }
});
