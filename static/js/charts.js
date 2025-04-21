// Initialize charts and configuration

// Helper function to format timestamps
function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// Initialize the Fuel Consumption Chart
function initFuelConsumptionChart() {
    const ctx = document.getElementById('fuelConsumptionChart').getContext('2d');
    
    const fuelConsumptionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Actual Fuel Consumption',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    borderWidth: 2
                },
                {
                    label: 'Predicted (Best Model)',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    tension: 0.4,
                    borderWidth: 2,
                    borderDash: [5, 5]
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
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Fuel Consumption (L/h)'
                    },
                    beginAtZero: true
                }
            },
            animation: {
                duration: 500
            }
        }
    });
    
    return fuelConsumptionChart;
}

// Initialize the Classification Chart (ECG-like)
function initClassificationChart() {
    const ctx = document.getElementById('classificationChart').getContext('2d');
    
    const classificationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Economic Driving Probability',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: function(context) {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;
                        if (!chartArea) return null;
                        
                        // Create gradient for the background
                        const gradient = ctx.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
                        gradient.addColorStop(0, 'rgba(54, 162, 235, 0)');
                        gradient.addColorStop(0.5, 'rgba(54, 162, 235, 0.1)');
                        gradient.addColorStop(1, 'rgba(54, 162, 235, 0.2)');
                        return gradient;
                    },
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: function(context) {
                        // Color points based on economic/non-economic threshold
                        const value = context.raw;
                        return value >= 0.5 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)';
                    }
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
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Probability'
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            animation: {
                duration: 500
            }
        }
    });
    
    return classificationChart;
}

// Initialize the Classification Metrics Chart
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
            }
        }
    });
    
    return classificationMetricsChart;
}

// Initialize the Regression Metrics Chart
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
            }
        }
    });
    
    return regressionMetricsChart;
}

// Update the fuel consumption chart with new data
function updateFuelConsumptionChart(chart, timestamp, actualValue, predictedValue) {
    // Format timestamp
    const timeLabel = formatTime(timestamp);
    
    // Add new data
    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(actualValue);
    chart.data.datasets[1].data.push(predictedValue);
    
    // Keep only last 20 data points for better visibility
    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }
    
    // Update chart
    chart.update();
}

// Update the classification chart with new data
function updateClassificationChart(chart, timestamp, economicProbability) {
    // Format timestamp
    const timeLabel = formatTime(timestamp);
    
    // Add new data
    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(economicProbability);
    
    // Keep only last 20 data points for better visibility
    if (chart.data.labels.length > 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    // Update chart
    chart.update();
}

// Export chart functions
window.chartFunctions = {
    initFuelConsumptionChart,
    initClassificationChart,
    initClassificationMetricsChart,
    initRegressionMetricsChart,
    updateFuelConsumptionChart,
    updateClassificationChart
};
