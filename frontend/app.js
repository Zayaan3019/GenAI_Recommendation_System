// API Configuration
const API_BASE_URL = 'http://localhost:3000';

// DOM Elements
const queryInput = document.getElementById('queryInput');
const topKInput = document.getElementById('topK');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const resultsMeta = document.getElementById('resultsMeta');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const apiStatus = document.getElementById('apiStatus');

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL.replace(':3000', ':8000')}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            apiStatus.textContent = '✓ Online';
            apiStatus.className = 'online';
        } else {
            apiStatus.textContent = '⚠ Not Ready';
            apiStatus.className = 'offline';
        }
    } catch (error) {
        apiStatus.textContent = '✗ Offline';
        apiStatus.className = 'offline';
    }
}

// Search function
async function searchAssessments() {
    const query = queryInput.value.trim();
    const topK = parseInt(topKInput.value);

    if (!query) {
        showError('Please enter a job description or search query');
        return;
    }

    if (topK < 5 || topK > 10) {
        showError('Number of recommendations must be between 5 and 10');
        return;
    }

    // Show loading state
    searchBtn.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    document.querySelector('.loader').style.display = 'block';
    hideResults();
    hideError();

    try {
        // Use port 8000 for API
        const apiUrl = API_BASE_URL.replace(':3000', ':8000');
        
        const response = await fetch(`${apiUrl}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(`Error: ${error.message}. Please ensure the API server is running on port 8000.`);
    } finally {
        // Reset loading state
        searchBtn.disabled = false;
        document.querySelector('.btn-text').style.display = 'inline';
        document.querySelector('.loader').style.display = 'none';
    }
}

// Display results with ABSOLUTE relative scoring
function displayResults(data) {
    const { recommendations, total_results, processing_time_ms, metadata } = data;

    // Update meta information
    resultsMeta.innerHTML = `
        ${total_results} results in ${processing_time_ms}ms
        <br>
        <small>Method: ${metadata.retrieval_method}</small>
    `;

    // Clear previous results
    resultsContainer.innerHTML = '';

    // Get the top (first) result's score as the baseline
    if (recommendations.length > 0) {
        const topScore = recommendations[0].score;
        
        console.log(`Top result score: ${topScore}`);
        
        // Calculate relative percentages based on top score
        recommendations.forEach((rec) => {
            // If top score is the baseline, calculate percentage relative to it
            if (topScore > 0) {
                rec.displayPercentage = (rec.score / topScore) * 100;
            } else {
                rec.displayPercentage = 100;
            }
            
            // Clamp to reasonable display range
            rec.displayPercentage = Math.max(0, Math.min(100, rec.displayPercentage));
            
            console.log(`${rec.assessment_name}: score=${rec.score.toFixed(4)}, display=${rec.displayPercentage.toFixed(1)}%`);
        });
    }

    // Create assessment cards
    recommendations.forEach((assessment, index) => {
        const card = createAssessmentCard(assessment, index + 1);
        resultsContainer.appendChild(card);
    });

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Create assessment card with ABSOLUTE relative scoring
function createAssessmentCard(assessment, rank) {
    const card = document.createElement('div');
    card.className = 'assessment-card';

    // Use the display percentage (relative to top result)
    const scorePercentage = assessment.displayPercentage || 0;
    
    // Determine score color based on value
    let scoreColor = '#10b981'; // green (80%+)
    if (scorePercentage < 50) scoreColor = '#ef4444'; // red (< 50%)
    else if (scorePercentage < 75) scoreColor = '#f59e0b'; // orange (50-75%)
    
    card.innerHTML = `
        <div class="card-header">
            <div>
                <div class="assessment-title">${rank}. ${assessment.assessment_name}</div>
            </div>
            <div class="score-badge" style="background: linear-gradient(135deg, ${scoreColor}, ${scoreColor}cc);">
                Match: ${scorePercentage.toFixed(1)}%
            </div>
        </div>
        <div class="card-body">
            <p class="assessment-description">${assessment.description || 'No description available'}</p>
            
            <div class="metadata">
                ${assessment.test_type ? `
                    <div class="metadata-item">
                        <strong>Type:</strong> ${assessment.test_type}
                    </div>
                ` : ''}
                ${assessment.duration ? `
                    <div class="metadata-item">
                        <strong>Duration:</strong> ${assessment.duration}
                    </div>
                ` : ''}
            </div>

            ${assessment.skills && assessment.skills.length > 0 ? `
                <div class="skills-container">
                    <strong>Skills Assessed:</strong><br>
                    ${assessment.skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            ` : ''}
        </div>
        <a href="${assessment.url}" target="_blank" class="assessment-link">
            View Assessment Details →
        </a>
    `;

    return card;
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    hideResults();
}

// Hide error
function hideError() {
    errorSection.style.display = 'none';
}

// Hide results
function hideResults() {
    resultsSection.style.display = 'none';
}

// Clear function
function clearSearch() {
    queryInput.value = '';
    topKInput.value = 10;
    hideResults();
    hideError();
}

// Event listeners
searchBtn.addEventListener('click', searchAssessments);
clearBtn.addEventListener('click', clearSearch);

// Allow Enter key in textarea (with Ctrl/Cmd) to submit
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        searchAssessments();
    }
});

// Initialize
checkAPIHealth();


