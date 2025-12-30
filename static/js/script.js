// Additional JavaScript functionality

// Auto-refresh data every 5 minutes
setInterval(() => {
    console.log('Auto-refreshing dashboard data...');
    // Could implement auto-refresh here
}, 300000);

// Export data functionality
function exportData(format) {
    alert(`Exporting data as ${format}...`);
    // Implement export functionality
}

// Print dashboard
function printDashboard() {
    window.print();
}

// Toggle dark mode
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    document.body.classList.toggle('bg-dark');
    document.body.classList.toggle('text-light');
}

// Tooltip initialization
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});