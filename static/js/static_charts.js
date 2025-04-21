// Static charts implementation based on the provided image

document.addEventListener('DOMContentLoaded', function() {
    // Initialize static charts
    initStaticFuelConsumptionChart();
    initStaticClassificationChart();
    
    // Initialize metrics charts if metrics data is available
    if (typeof classificationMetrics !== 'undefined' && typeof regressionMetrics !== 'undefined') {
        initClassificationMetricsChart(classificationMetrics);
        initRegressionMetricsChart(regressionMetrics);
        
        // Populate metrics tables
        populateMetricsTables();
    } else {
        console.warn('Metrics data not available. Tables and charts will not be populated.');
    }
});

// Initialize static fuel consumption chart (similar to the image)
function initStaticFuelConsumptionChart() {
    const ctx = document.getElementById('staticFuelConsumptionChart').getContext('2d');
    
    // Generate realistic time labels (x-axis)
    const timeLabels = [];
    const now = new Date();
    for (let i = 0; i < 60; i++) {
        const time = new Date(now - (60 - i) * 1000);
        timeLabels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    }
    
    // Generate realistic fuel consumption data
    // This mimics the jagged pattern from the image
    const fuelData = [];
    let baseValue = 10;
    for (let i = 0; i < 60; i++) {
        if (i < 40) {
            // First part with moderate fluctuations
            baseValue += (Math.random() - 0.5) * 3;
            baseValue = Math.max(5, Math.min(15, baseValue));
        } else {
            // Latter part with higher values and wider fluctuations (as in the image)
            baseValue = 12 + (Math.random() - 0.3) * 8;
            baseValue = Math.max(8, Math.min(20, baseValue));
        }
        fuelData.push(baseValue);
    }
    
    // Create the chart
    const fuelConsumptionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: 'Fuel Consumption',
                    data: fuelData,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Economic Threshold (7.5 L/h)',
                    data: Array(60).fill(7.5), // Using 7.5 L/h as the threshold
                    borderColor: 'rgba(255, 159, 64, 0.8)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    },
                    ticks: {
                        maxTicksLimit: 6,
                        maxRotation: 0
                    },
                    grid: {
                        display: true,
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Fuel Consumption (L/h)'
                    },
                    min: 0,
                    max: 20,
                    ticks: {
                        stepSize: 5
                    },
                    grid: {
                        display: true,
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            animation: false,
            elements: {
                line: {
                    tension: 0.1 // Slightly smoother line (not too smooth as in the image)
                }
            }
        }
    });
}

// Initialize static classification chart
function initStaticClassificationChart() {
    const ctx = document.getElementById('staticClassificationChart').getContext('2d');
    
    // Generate realistic time labels
    const timeLabels = [];
    const now = new Date();
    for (let i = 0; i < 60; i++) {
        const time = new Date(now - (60 - i) * 1000);
        timeLabels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    }
    
    // Generate classification probability data (ECG-like)
    const classData = [];
    let value = 0.5;
    for (let i = 0; i < 60; i++) {
        if (i < 20) {
            // Oscillating around economic threshold (0.5)
            value = 0.5 + Math.sin(i * 0.5) * 0.3;
        } else if (i < 40) {
            // Clearly economic (above 0.5)
            value = 0.7 + Math.sin(i * 0.4) * 0.2;
        } else {
            // More variable in last section
            value = 0.4 + Math.sin(i * 0.6) * 0.5;
        }
        value = Math.max(0.1, Math.min(0.9, value));
        classData.push(value);
    }
    
    // Create the classification chart (ECG-like)
    const classificationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: 'Economic Driving Probability',
                    data: classData,
                    borderColor: function(context) {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;
                        if (!chartArea) return 'rgba(54, 162, 235, 1)';
                        
                        // Create gradient for the line
                        const gradient = ctx.createLinearGradient(0, 0, chartArea.width, 0);
                        gradient.addColorStop(0, 'rgba(54, 162, 235, 1)');
                        gradient.addColorStop(0.3, 'rgba(75, 192, 192, 1)');
                        gradient.addColorStop(0.6, 'rgba(54, 162, 235, 1)');
                        gradient.addColorStop(0.8, 'rgba(255, 99, 132, 1)');
                        gradient.addColorStop(1, 'rgba(54, 162, 235, 1)');
                        return gradient;
                    },
                    backgroundColor: 'transparent',
                    tension: 0.2,
                    borderWidth: 1.5,
                    pointRadius: 0
                },
                {
                    label: 'Economic Threshold',
                    data: Array(60).fill(0.5), // Constant threshold line
                    borderColor: 'rgba(255, 159, 64, 0.8)',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0,
                    borderWidth: 1.5,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const classification = value >= 0.5 ? 'Economic' : 'Non-Economic';
                            return `Probability: ${(value * 100).toFixed(1)}% (${classification})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    },
                    ticks: {
                        maxTicksLimit: 6,
                        maxRotation: 0
                    },
                    grid: {
                        display: true,
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Economic Driving Probability'
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            animation: false
        }
    });
}

// Initialize the Classification Metrics Chart (reused from original)
function initClassificationMetricsChart(metrics) {
    const ctx = document.getElementById('classificationMetricsChart').getContext('2d');
    
    // Extract model names and metrics
    const modelNames = Object.keys(metrics);
    const accuracy = modelNames.map(name => metrics[name].accuracy);
    const precision = modelNames.map(name => metrics[name].precision);
    const recall = modelNames.map(name => metrics[name].recall);
    const f1Score = modelNames.map(name => metrics[name].f1_score);
    
    const classificationMetricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'Accuracy',
                    data: accuracy,
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Precision',
                    data: precision,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Recall',
                    data: recall,
                    backgroundColor: 'rgba(255, 159, 64, 0.7)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                },
                {
                    label: 'F1 Score',
                    data: f1Score,
                    backgroundColor: 'rgba(153, 102, 255, 0.7)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            return `${context.dataset.label}: ${(value * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Models'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Metric Value'
                    },
                    min: 0.8,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            animation: false,
            // Prevent chart from growing by setting a fixed height
            onResize: function(chart, size) {
                if (size.height > 300) {
                    chart.height = 300;
                }
            }
        }
    });
    
    return classificationMetricsChart;
}

// Initialize the Regression Metrics Chart (reused from original)
function initRegressionMetricsChart(metrics) {
    const ctx = document.getElementById('regressionMetricsChart').getContext('2d');
    
    // Extract model names and metrics
    const modelNames = Object.keys(metrics);
    const r2Scores = modelNames.map(name => metrics[name].r2);
    const mseValues = modelNames.map(name => metrics[name].mse);
    const maeValues = modelNames.map(name => metrics[name].mae);
    
    // Normalize MSE and MAE for better visualization (they can be large)
    const maxMSE = Math.max(...mseValues);
    const normalizedMSE = mseValues.map(value => value / maxMSE);
    
    const maxMAE = Math.max(...maeValues);
    const normalizedMAE = maeValues.map(value => value / maxMAE);
    
    const regressionMetricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [
                {
                    label: 'R² Score',
                    data: r2Scores,
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    yAxisID: 'y'
                },
                {
                    label: 'Normalized MSE',
                    data: normalizedMSE,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                },
                {
                    label: 'Normalized MAE',
                    data: normalizedMAE,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const datasetLabel = context.dataset.label;
                            const value = context.raw;
                            
                            if (datasetLabel === 'R² Score') {
                                return `${datasetLabel}: ${value.toFixed(3)}`;
                            } else if (datasetLabel === 'Normalized MSE') {
                                return `MSE: ${mseValues[context.dataIndex].toFixed(3)}`;
                            } else if (datasetLabel === 'Normalized MAE') {
                                return `MAE: ${maeValues[context.dataIndex].toFixed(3)}`;
                            }
                            return `${datasetLabel}: ${value}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Models'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'R² Score'
                    },
                    min: 0.8,
                    max: 1
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Normalized Error'
                    },
                    min: 0,
                    max: 1,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            },
            animation: false,
            // Prevent chart from growing by setting a fixed height
            onResize: function(chart, size) {
                if (size.height > 300) {
                    chart.height = 300;
                }
            }
        }
    });
    
    return regressionMetricsChart;
}

// Populate metrics tables with data from model_metrics.js
function populateMetricsTables() {
    // Populate classification table
    const classificationTable = document.getElementById('classificationTable');
    for (const [modelName, metrics] of Object.entries(classificationMetrics)) {
        const row = document.createElement('tr');
        
        // Calculate a prediction value (for static display)
        const predictionValue = 0.5 + (metrics.f1_score - 0.9) * 2;
        const isEconomic = predictionValue >= 0.5;
        
        row.innerHTML = `
            <td>${modelName}</td>
            <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
            <td>${(metrics.precision * 100).toFixed(2)}%</td>
            <td>${(metrics.recall * 100).toFixed(2)}%</td>
            <td>${(metrics.f1_score * 100).toFixed(2)}%</td>
            <td class="${isEconomic ? 'text-success' : 'text-danger'}">
                ${(predictionValue * 100).toFixed(1)}% ${isEconomic ? '(Economic)' : '(Non-Economic)'}
            </td>
        `;
        classificationTable.appendChild(row);
    }
    
    // Populate regression table
    const regressionTable = document.getElementById('regressionTable');
    for (const [modelName, metrics] of Object.entries(regressionMetrics)) {
        const row = document.createElement('tr');
        
        // Calculate a prediction value (for static display)
        const predictionValue = 8 + metrics.r2 * 5;
        
        row.innerHTML = `
            <td>${modelName}</td>
            <td>${metrics.mse.toFixed(3)}</td>
            <td>${metrics.mae.toFixed(3)}</td>
            <td>${metrics.r2.toFixed(3)}</td>
            <td>${predictionValue.toFixed(2)} L/h</td>
        `;
        regressionTable.appendChild(row);
    }
}