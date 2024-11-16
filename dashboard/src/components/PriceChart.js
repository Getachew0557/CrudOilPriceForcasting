// src/components/PriceChart.js
import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { fetchPriceData } from '../api/api';

function PriceChart() {
  const [priceData, setPriceData] = useState(null);

  useEffect(() => {
    fetchPriceData().then((data) => setPriceData(data));
  }, []);

  return (
    <div className="chart-container">
      {priceData ? (
        <Line
          data={{
            labels: priceData.dates,
            datasets: [
              {
                label: 'Brent Oil Price',
                data: priceData.prices,
                borderColor: 'rgba(75,192,192,1)',
                fill: false,
              },
            ],
          }}
        />
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default PriceChart;
