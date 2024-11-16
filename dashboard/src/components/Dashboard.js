// src/components/Dashboard.js
import React from 'react';
import PriceChart from './PriceChart';
import EventChart from './EventChart';

function Dashboard() {
  return (
    <div>
      <h2>Dashboard</h2>
      <div className="charts">
        <PriceChart />
        <EventChart />
      </div>
    </div>
  );
}

export default Dashboard;
