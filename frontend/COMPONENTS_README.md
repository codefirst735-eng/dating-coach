# Component System Usage Guide

## Overview
The RFH website now uses a reusable component system for headers and footers. This eliminates code duplication and makes maintenance easier.

## How It Works

### 1. Component Files
Located in `frontend/components/`:
- **header.html** - Contains the navigation bar
- **footer.html** - Contains the footer section

### 2. Component Loader
The `js/components.js` script automatically loads these components into your pages.

### 3. Dynamic Navbar
The `js/navbar.js` script adds Login/Profile links dynamically based on authentication status.

## How to Use Components in a Page

### Step 1: Add Placeholders
Replace your hardcoded header and footer with placeholders:

```html
<body>
  <!-- Header Component -->
  <div id="header-placeholder"></div>
  
  <!-- Your page content here -->
  
  <!-- Footer Component -->
  <div id="footer-placeholder"></div>
</body>
```

### Step 2: Load Scripts in Correct Order
Add scripts at the end of your `<body>` tag:

```html
  <!-- Load components first, then other scripts -->
  <script src="js/components.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script src="js/navbar.js"></script>
  <script src="js/toast.js"></script>
  
  <!-- Your page-specific scripts here -->
</body>
```

**Important:** `components.js` must load BEFORE `navbar.js` to ensure the navbar HTML exists before dynamic links are added.

## Example: Converting a Page

### Before (Old Way):
```html
<body>
  <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
    <!-- 30+ lines of navbar HTML -->
  </nav>
  
  <!-- Page content -->
  
  <footer class="bg-black text-white-50 py-5">
    <!-- 20+ lines of footer HTML -->
  </footer>
  
  <script src="js/navbar.js"></script>
</body>
```

### After (New Way):
```html
<body>
  <div id="header-placeholder"></div>
  
  <!-- Page content -->
  
  <div id="footer-placeholder"></div>
  
  <script src="js/components.js"></script>
  <script src="js/navbar.js"></script>
</body>
```

## Benefits

1. **DRY Principle**: Header/footer code exists in only ONE place
2. **Easy Updates**: Change header/footer once, updates everywhere
3. **Dynamic Auth**: Login/Logout links automatically appear based on user state
4. **Cleaner Code**: Pages are much shorter and easier to read
5. **Consistent UI**: Guaranteed consistency across all pages

## Files Updated

The following pages have been converted to use components:
- ✅ `index.html` (example implementation)

## To Convert Remaining Pages

For each HTML file (`chat.html`, `screenshot-analysis.html`, `results.html`, etc.):

1. Replace `<nav>...</nav>` with `<div id="header-placeholder"></div>`
2. Replace `<footer>...</footer>` with `<div id="footer-placeholder"></div>`
3. Add `<script src="js/components.js"></script>` before other scripts
4. Ensure `navbar.js` loads after `components.js`

## Troubleshooting

**Problem:** Navbar doesn't show Login/Profile links
- **Solution:** Make sure `components.js` loads before `navbar.js`

**Problem:** Components don't load
- **Solution:** Check browser console for errors. Ensure paths are correct.

**Problem:** 404 error for component files
- **Solution:** Verify `components/header.html` and `components/footer.html` exist

## Technical Details

### Load Sequence:
1. `components.js` loads and fetches `header.html` and `footer.html`
2. Components are injected into placeholders
3. `components.js` calls `window.initNavbar()`
4. `navbar.js` adds dynamic auth links to the navbar

### Backwards Compatibility:
`navbar.js` still works on pages with hardcoded navbars (auto-detects and initializes).
