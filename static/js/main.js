// Main JavaScript for handling the UI and data flow

// Global variables
let simulationInterval = null;
let isSimulationRunning = false;
let fuelConsumptionChart = null;
let classificationChart = null;
let classificationMetricsChart = null;
let regressionMetricsChart = null;

// Custom parameter handling
function updateCustomField() {
    const customParamSelect = document.getElementById('custom_param_name');
    const customParamValue = document.getElementById('custom_param_value');
    const selectedParam = customParamSelect.value;
    
    if (selectedParam) {
        // Find the corresponding dropdown in the form
        const paramDropdown = document.getElementById(selectedParam);
        if (paramDropdown) {
            // Disable the dropdown since we'll use the custom value
            paramDropdown.disabled = true;
            
            // Focus on the custom value field
            customParamValue.focus();
            
            // Add a data attribute to remember which dropdown is disabled
            customParamValue.dataset.linkedParam = selectedParam;
        }
    } else {
        // If no parameter is selected, re-enable all dropdowns
        const linkedParam = customParamValue.dataset.linkedParam;
        if (linkedParam) {
            const paramDropdown = document.getElementById(linkedParam);
            if (paramDropdown) {
                paramDropdown.disabled = false;
            }
            // Clear the data attribute
            delete customParamValue.dataset.linkedParam;
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    fuelConsumptionChart = window.chartFunctions.initFuelConsumptionChart();
    classificationChart = window.chartFunctions.initClassificationChart();
    
    // Initialize metrics charts if metrics data is available
    if (typeof classificationMetrics !== 'undefined' && typeof regressionMetrics !== 'undefined') {
        classificationMetricsChart = window.chartFunctions.initClassificationMetricsChart(classificationMetrics);
        regressionMetricsChart = window.chartFunctions.initRegressionMetricsChart(regressionMetrics);
        
        // Populate metrics tables
        populateMetricsTables();
    } else {
        console.warn('Metrics data not available. Tables and charts will not be populated.');
    }
    
    // Set up event listeners
    const toggleSimBtn = document.getElementById('toggleSimulation');
    if (toggleSimBtn) {
        toggleSimBtn.addEventListener('click', toggleSimulation);
    }
    
    // Add event listener for form submission to handle custom parameters
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            // If custom parameter is set, make sure it's included in the form
            const customParamName = document.getElementById('custom_param_name');
            const customParamValue = document.getElementById('custom_param_value');
            
            if (customParamName && customParamValue && 
                customParamName.value && customParamValue.value) {
                // Make sure the custom parameter is included even if its regular dropdown is disabled
                const hiddenInput = document.createElement('input');
                hiddenInput.type = 'hidden';
                hiddenInput.name = customParamName.value;
                hiddenInput.value = customParamValue.value;
                predictionForm.appendChild(hiddenInput);
            }
        });
    }
});

// Populate metrics tables with data from model_metrics.js
function populateMetricsTables() {
    // Populate classification table
    const classificationTable = document.getElementById('classificationTable');
    for (const [modelName, metrics] of Object.entries(classificationMetrics)) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${modelName}</td>
            <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
            <td>${(metrics.precision * 100).toFixed(2)}%</td>
            <td>${(metrics.recall * 100).toFixed(2)}%</td>
            <td>${(metrics.f1_score * 100).toFixed(2)}%</td>
            <td id="${modelName.replace(/\s+/g, '-')}-prediction">-</td>
        `;
        classificationTable.appendChild(row);
    }
    
    // Populate regression table
    const regressionTable = document.getElementById('regressionTable');
    for (const [modelName, metrics] of Object.entries(regressionMetrics)) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${modelName}</td>
            <td>${metrics.mse.toFixed(3)}</td>
            <td>${metrics.mae.toFixed(3)}</td>
            <td>${metrics.r2.toFixed(3)}</td>
            <td id="${modelName.replace(/\s+/g, '-')}-prediction">-</td>
        `;
        regressionTable.appendChild(row);
    }
}

// Toggle simulation start/stop
function toggleSimulation() {
    const button = document.getElementById('toggleSimulation');
    
    if (isSimulationRunning) {
        // Stop simulation
        clearInterval(simulationInterval);
        isSimulationRunning = false;
        button.innerHTML = '<i class="fas fa-play me-1"></i> Start Simulation';
        button.classList.remove('btn-danger');
        button.classList.add('btn-primary');
    } else {
        // Start simulation
        fetchData(); // Fetch initial data immediately
        simulationInterval = setInterval(fetchData, 1000); // Then fetch every second
        isSimulationRunning = true;
        button.innerHTML = '<i class="fas fa-stop me-1"></i> Stop Simulation';
        button.classList.remove('btn-primary');
        button.classList.add('btn-danger');
    }
}

// Fetch data from the server
function fetchData() {
    fetch('/get_data')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateUI(data);
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            // Stop simulation on error
            if (isSimulationRunning) {
                toggleSimulation();
            }
        });
}

// Update UI with fetched data
function updateUI(data) {
    // Update vehicle parameters
    document.getElementById('rpm-value').textContent = Math.round(data.features.rpm);
    document.getElementById('speed-value').textContent = Math.round(data.features.speed) + ' km/h';
    document.getElementById('throttle-value').textContent = Math.round(data.features.throttle_position) + '%';
    document.getElementById('acceleration-value').textContent = data.features.acceleration.toFixed(1) + ' m/s²';
    document.getElementById('engine-load-value').textContent = Math.round(data.features.engine_load) + '%';
    document.getElementById('fuel-consumption-value').textContent = data.actual.fuel_consumption.toFixed(1) + ' L/h';
    
    // Update progress bars
    document.getElementById('rpm-progress').style.width = (data.features.rpm / 6000 * 100) + '%';
    document.getElementById('speed-progress').style.width = (data.features.speed / 150 * 100) + '%';
    document.getElementById('throttle-progress').style.width = data.features.throttle_position + '%';
    
    // Acceleration progress bar is special: it's centered at 0
    const accPercent = 50 + (data.features.acceleration * 10);
    document.getElementById('acceleration-progress').style.width = Math.min(Math.max(accPercent, 0), 100) + '%';
    document.getElementById('acceleration-progress').className = data.features.acceleration >= 0 
        ? 'progress-bar bg-success' 
        : 'progress-bar bg-warning';
    
    document.getElementById('engine-load-progress').style.width = data.features.engine_load + '%';
    document.getElementById('fuel-consumption-progress').style.width = (data.actual.fuel_consumption / 15 * 100) + '%';
    document.getElementById('fuel-consumption-progress').className = data.actual.is_economic === 1 
        ? 'progress-bar bg-success' 
        : 'progress-bar bg-danger';
    
    // Update classification model predictions
    for (const [modelName, probability] of Object.entries(data.classification_results)) {
        const elementId = modelName.replace(/\s+/g, '-') + '-prediction';
        const element = document.getElementById(elementId);
        if (element) {
            const isEconomic = probability >= 0.5;
            element.textContent = `${(probability * 100).toFixed(1)}% ${isEconomic ? '(Economic)' : '(Non-Economic)'}`;
            element.className = isEconomic ? 'text-success' : 'text-danger';
        }
    }
    
    // Update regression model predictions
    for (const [modelName, prediction] of Object.entries(data.regression_results)) {
        const elementId = modelName.replace(/\s+/g, '-') + '-prediction';
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = prediction.toFixed(2) + ' L/h';
        }
    }
    
    // Update charts
    // Find the best regression model (using the one with highest R²)
    let bestRegressionModel = Object.keys(regressionMetrics).reduce((a, b) => {
        return regressionMetrics[a].r2 > regressionMetrics[b].r2 ? a : b;
    });
    
    // Find the best classification model (using the one with highest F1 score)
    let bestClassificationModel = Object.keys(classificationMetrics).reduce((a, b) => {
        return classificationMetrics[a].f1_score > classificationMetrics[b].f1_score ? a : b;
    });
    
    // Update fuel consumption chart with actual and best predicted values
    window.chartFunctions.updateFuelConsumptionChart(
        fuelConsumptionChart,
        data.timestamp,
        data.actual.fuel_consumption,
        data.regression_results[bestRegressionModel]
    );
    
    // Update classification chart with best model's probability
    window.chartFunctions.updateClassificationChart(
        classificationChart,
        data.timestamp,
        data.classification_results[bestClassificationModel]
    );
}
