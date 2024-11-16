// src/components/EventChart.js
import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { fetchEventData } from '../api/api';

function EventChart() {
  const [eventData, setEventData] = useState(null);

  useEffect(() => {
    fetchEventData().then((data) => setEventData(data));
  }, []);

  return (
    <div className="chart-container">
      {eventData ? (
        <Bar
          data={{
            labels: eventData.events,
            datasets: [
              {
                label: 'Price Impact',
                data: eventData.impact,
                backgroundColor: 'rgba(255,99,132,0.6)',
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

export default EventChart;
