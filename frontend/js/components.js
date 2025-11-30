// Component Loader - Loads header and footer components dynamically
(function () {
    // Load a component into a target element
    async function loadComponent(componentPath, targetSelector) {
        try {
            const response = await fetch(componentPath);
            if (!response.ok) {
                console.error(`Failed to load component: ${componentPath}`);
                return;
            }
            const html = await response.text();
            const targetElement = document.querySelector(targetSelector);
            if (targetElement) {
                targetElement.innerHTML = html;
            }
        } catch (error) {
            console.error(`Error loading component ${componentPath}:`, error);
        }
    }

    // Load all components when DOM is ready
    async function loadAllComponents() {
        // Load header
        await loadComponent('components/header.html', '#header-placeholder');

        // Load footer
        await loadComponent('components/footer.html', '#footer-placeholder');

        // After components are loaded, initialize navbar dynamic links
        // This ensures navbar.js runs after the navbar HTML is in the DOM
        if (window.initNavbar) {
            window.initNavbar();
        }

        // Manually initialize Bootstrap Collapse for the mobile navbar
        // This is necessary because the navbar is added dynamically after Bootstrap has loaded
        const navbarCollapse = document.querySelector('.navbar-collapse');
        const navbarToggler = document.querySelector('.navbar-toggler');

        if (navbarCollapse && typeof bootstrap !== 'undefined') {
            const bsCollapse = new bootstrap.Collapse(navbarCollapse, {
                toggle: false
            });

            // Add click listener to toggler if it exists
            if (navbarToggler) {
                navbarToggler.addEventListener('click', function (e) {
                    e.preventDefault();
                    bsCollapse.toggle();
                });
            }
        }
    }

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadAllComponents);
    } else {
        loadAllComponents();
    }
})();
