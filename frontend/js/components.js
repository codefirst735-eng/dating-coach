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
    }

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadAllComponents);
    } else {
        loadAllComponents();
    }
})();
