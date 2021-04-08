function showInitialChart(chartData1){
var ctx = document.getElementById('stockchart').getContext('2d');
var stockchart = new Chart(ctx, {
    type: 'line',
    data: {
        datasets: [{
            label: "Test",
            data: chartData1
        }]
    },
    options: {
        responsive: true,
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true,
                    stepSize: 50
                }
            }]
        }
    }
});
}