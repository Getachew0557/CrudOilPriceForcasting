// src/components/Filters.js
import React, { useState } from 'react';

function Filters() {
  const [dateRange, setDateRange] = useState('2021-01-01 to 2021-12-31');
  const [eventType, setEventType] = useState('All');

  return (
    <div className="filters">
      <h4>Filters</h4>
      <label>
        Date Range:
        <input
          type="text"
          value={dateRange}
          onChange={(e) => setDateRange(e.target.value)}
        />
      </label>
      <label>
        Event Type:
        <select
          value={eventType}
          onChange={(e) => setEventType(e.target.value)}
        >
          <option value="All">All</option>
          <option value="Political">Political</option>
          <option value="Economic">Economic</option>
          <option value="Natural">Natural</option>
        </select>
      </label>
    </div>
  );
}

export default Filters;
