const role = localStorage.getItem('role');
const token = localStorage.getItem('token');

if (!token) {
    window.location.href = '/static/login.html';
}

const navHTML = `
<nav class="sidebar">
    <div class="logo">
        <ion-icon name="finger-print-outline"></ion-icon>
        <span>WitSketch</span>
    </div>
    
    <div class="nav-links">
        <a href="/static/dashboard.html" class="nav-item" id="nav-dash">
            <ion-icon name="grid-outline"></ion-icon> Dashboard
        </a>
        <a href="/static/generate.html" class="nav-item" id="nav-gen">
            <ion-icon name="create-outline"></ion-icon> Text Generator
        </a>
        <a href="/static/builder.html?v=2" class="nav-item" id="nav-builder">
            <ion-icon name="construct-outline"></ion-icon> Visual Builder
        </a>
        <a href="/static/match.html" class="nav-item" id="nav-match">
            <ion-icon name="search-outline"></ion-icon> Match Database
        </a>
        <a href="/static/cctv.html" class="nav-item" id="nav-cctv">
            <ion-icon name="videocam-outline"></ion-icon> CCTV Video Face Match
        </a>
        ${role === 'admin' ? `
        <a href="/static/admin.html" class="nav-item" id="nav-admin">
            <ion-icon name="shield-checkmark-outline"></ion-icon> Admin Panel
        </a>
        ` : ''}
    </div>

    <div class="user-info">
        <div class="user-role">${role.toUpperCase()}</div>
        <button id="logoutBtn" class="logout-btn">
            <ion-icon name="log-out-outline"></ion-icon> Logout
        </button>
    </div>
</nav>
`;

document.addEventListener('DOMContentLoaded', () => {
    // Inject Nav
    const placeholder = document.getElementById('nav-placeholder');
    if (placeholder) {
        placeholder.innerHTML = navHTML;

        // Highlight active
        const path = window.location.pathname;
        if (path.includes('dashboard')) document.getElementById('nav-dash').classList.add('active');
        if (path.includes('generate')) document.getElementById('nav-gen').classList.add('active');
        if (path.includes('builder')) document.getElementById('nav-builder').classList.add('active');
        if (path.includes('match') && !path.includes('cctv')) document.getElementById('nav-match').classList.add('active');
        if (path.includes('cctv')) document.getElementById('nav-cctv')?.classList.add('active');
        if (path.includes('admin')) document.getElementById('nav-admin')?.classList.add('active');

        // Logout
        document.getElementById('logoutBtn').addEventListener('click', () => {
            localStorage.clear();
            window.location.href = '/static/login.html';
        });
    }
});
