// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const themeToggle = document.getElementById('theme-toggle');
const predictionForm = document.getElementById('prediction-form');
const resultContainer = document.getElementById('result');
const resetFormButton = document.getElementById('reset-form');

// Sidebar toggling
if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
        sidebarOverlay.classList.toggle('active');
    });
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', () => {
        sidebar.classList.remove('active');
        sidebarOverlay.classList.remove('active');
    });
}

// Theme toggling
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    }
}

// Reset form button
if (resetFormButton && predictionForm) {
    resetFormButton.addEventListener('click', () => {
        predictionForm.reset();
        resultContainer.style.display = 'none';
        resultContainer.innerHTML = '';
    });
}

// Handle form submission
if (predictionForm) {
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitButton = predictionForm.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

        const formData = {};
        Array.from(predictionForm.elements).forEach(element => {
            if (element.name) {
                formData[element.name] = element.value;
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw {
                    type: errorData.type || 'SERVER_ERROR',
                    message: errorData.message || 'Server error',
                    fields: errorData.fields || [],
                    details: errorData.details || `Status: ${response.status}`
                };
            }

            const result = await response.json();
            renderResult(result, formData);

        } catch (error) {
            showError(error);
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonText;
        }
    });
}

function renderResult(result, formData) {
    const isApproved = result.approved;
    const probability = (result.probability * 100).toFixed(1);
    const confidence = result.confidence.toFixed(1);
    const risk = result.risk_level;
    const rate = result.interest_rate;
    const keyFactors = result.key_factors || [];

    let html = `
        <div class="result-header">
            <div class="result-icon ${isApproved ? '' : 'rejected'}">
                <i class="fas ${isApproved ? 'fa-check' : 'fa-times'}"></i>
            </div>
            <div>
                <div class="result-title">Loan ${isApproved ? 'Approved' : 'Rejected'}</div>
                <div class="result-subtitle">Based on your provided information</div>
            </div>
        </div>
        <div class="progress-container">
            <div class="progress-label">
                <span>Approval Probability</span>
                <span>${probability}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width: ${probability}%; background-color: ${isApproved ? 'var(--success)' : 'var(--danger)'};"></div>
            </div>
        </div>
        <div class="result-details">
            <div class="result-item"><strong>Risk Level:</strong> ${risk}</div>
            <div class="result-item"><strong>Confidence Score:</strong> ${confidence}%</div>
            <div class="result-item"><strong>Interest Rate:</strong> ${rate}%</div>
        </div>
        <div class="feature-importance">
            <h4>Key Factors</h4>
            ${keyFactors.map(f => `
                <div class="feature-item">
                    <span>${f.name}</span>
                    <div class="feature-bar">
                        <div class="feature-bar-fill" style="width: ${f.value}%; background-color: ${getFactorColor(f.value)};"></div>
                    </div>
                    <span>${f.value}%</span>
                </div>
            `).join('')}
        </div>
        ${generateRecommendations(result, formData)}
    `;

    resultContainer.innerHTML = html;
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getFactorColor(value) {
    if (value >= 70) return 'var(--success)';
    if (value >= 40) return 'var(--info)';
    return 'var(--accent-primary)';
}

function generateRecommendations(result, formData) {
    if (!result.recommendations || result.recommendations.length === 0) return '';

    return `
        <div class="recommendations mt-5 p-4 border border-yellow-500 rounded bg-yellow-50 dark:bg-yellow-900/10">
            <h4 class="text-base font-semibold text-yellow-800 dark:text-yellow-300 mb-2">Recommendations to Improve Eligibility</h4>
            <ul class="list-disc pl-5 text-sm text-yellow-900 dark:text-yellow-200">
                ${result.recommendations.map(r => `<li>${r}</li>`).join('')}
            </ul>
        </div>
    `;
}

function showError(error) {
    console.error('Error:', error);
    const troubleshooting = generateTroubleshootingMessage(error);
    resultContainer.innerHTML = `
        <div class="error-alert animate-shake">
            <div class="error-header">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>${troubleshooting.title}</h3>
            </div>
            <div class="error-body">
                <p>${troubleshooting.description}</p>
                ${troubleshooting.steps}
                ${error.details ? `<div class="technical-details"><small>${error.details}</small></div>` : ''}
            </div>
            <div class="error-footer">
                <button class="btn btn-retry" onclick="window.location.reload()">
                    <i class="fas fa-sync-alt"></i> Try Again
                </button>
            </div>
        </div>
    `;
    resultContainer.style.display = 'block';
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function generateTroubleshootingMessage(error) {
    const base = {
        title: "Prediction Failed",
        description: "We couldn't process your request. Please check the following:",
        steps: `
            <ul class="troubleshooting-steps">
                <li><i class="fas fa-check-circle"></i> All required fields are filled</li>
                <li><i class="fas fa-check-circle"></i> Entered values are valid</li>
                <li><i class="fas fa-wifi"></i> Internet connection is working</li>
            </ul>`
    };

    const types = {
        NETWORK_ERROR: {
            title: "Connection Problem",
            description: "Cannot connect to the prediction server.",
            steps: `
                <ul class="troubleshooting-steps">
                    <li><i class="fas fa-wifi"></i> Check internet access</li>
                    <li><i class="fas fa-server"></i> API server might be down</li>
                    <li><i class="fas fa-shield-alt"></i> Check firewall/VPN settings</li>
                </ul>`
        },
        VALIDATION_ERROR: {
            title: "Input Error",
            description: "There were issues with your input.",
            steps: `
                <ul class="troubleshooting-steps">
                    ${error.fields.map(field => `<li><i class="fas fa-times-circle"></i> ${field}</li>`).join('')}
                </ul>`
        },
        SERVER_ERROR: {
            title: "Server Issue",
            description: "The server encountered an error.",
            steps: `
                <ul class="troubleshooting-steps">
                    <li><i class="fas fa-redo"></i> Try again later</li>
                    <li><i class="fas fa-bug"></i> Report to technical team</li>
                </ul>`
        }
    };

    return types[error.type] || base;
}

// Handle Prediction History Click
// Handle Prediction History Click
const historyLink = document.querySelector('.sidebar-menu-link[data-page="history"]');
if (historyLink) {
    historyLink.addEventListener('click', async function (e) {
        e.preventDefault();
        
        // Show loading state
        const resultContainer = document.getElementById('result');
        resultContainer.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 2rem;"></i>
                    </div>
                </div>
            </div>
        `;
        resultContainer.style.display = 'block';

        try {
            const response = await fetch('/history');
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();

            if (!Array.isArray(data)) {
                throw new Error("Invalid data format received");
            }

            if (data.length === 0) {
                resultContainer.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <p>No prediction history found.</p>
                        </div>
                    </div>
                `;
                return;
            }

            // Build table HTML with your existing styles
            let html = `
                <div class="card animate-slide-in-left">
                    <div class="card-header">
                        <h2 class="card-title">
                            <i class="fas fa-history"></i>
                            Prediction History
                        </h2>
                    </div>
                    <div class="card-body" style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="background-color: var(--bg-tertiary);">
                                    <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color);">Date</th>
                                    <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color);">Status</th>
                                    <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color);">Amount</th>
                                    <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color);">Probability</th>
                                    <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color);">Risk</th>
                                    <th style="padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color);">Rate</th>
                                </tr>
                            </thead>
                            <tbody>
            `;

            data.forEach(row => {
                const statusColor = row.approved === 'Approved' ? 'var(--success)' : 'var(--danger)';
                html += `
                    <tr style="border-bottom: 1px solid var(--border-color);">
                        <td style="padding: 0.75rem;">${row.prediction_date}</td>
                        <td style="padding: 0.75rem; color: ${statusColor}">${row.approved}</td>
                        <td style="padding: 0.75rem;">${row.loan_amount}</td>
                        <td style="padding: 0.75rem;">${row.probability}</td>
                        <td style="padding: 0.75rem;">${row.risk_level}</td>
                        <td style="padding: 0.75rem;">${row.interest_rate}</td>
                    </tr>
                `;
            });

            html += `</tbody></table></div></div>`;

            resultContainer.innerHTML = html;
        } catch (error) {
            console.error("Error fetching history:", error);
            resultContainer.innerHTML = `
                <div class="error-alert">
                    <div class="error-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h3>Error Loading History</h3>
                    </div>
                    <div class="error-body">
                        <p>Failed to load prediction history: ${error.message}</p>
                        <div class="error-footer">
                            <button class="btn btn-secondary" onclick="location.reload()">
                                <i class="fas fa-sync-alt"></i> Try Again
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
    });
}