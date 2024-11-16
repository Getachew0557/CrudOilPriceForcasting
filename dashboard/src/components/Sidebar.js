// src/components/Sidebar.js
import React from 'react';
import { Nav } from 'react-bootstrap';

function Sidebar() {
  return (
    <div className="sidebar">
      <h3>Filters</h3>
      <Nav defaultActiveKey="/home" className="flex-column">
        <Nav.Link href="#price">Oil Price</Nav.Link>
        <Nav.Link href="#event-type">Event Type</Nav.Link>
        <Nav.Link href="#date-range">Date Range</Nav.Link>
      </Nav>
    </div>
  );
}

export default Sidebar;
