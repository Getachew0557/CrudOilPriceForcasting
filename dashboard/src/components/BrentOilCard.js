import React, { useEffect, useState } from 'react';
import { Table } from 'react-bootstrap';

function BrentOilCard() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('/api/brent_oil')
      .then((response) => response.json())
      .then((data) => setData(data))
      .catch((error) => console.error('Error fetching data:', error));
  }, []);

  return (
    <div className="card">
      <div className="card-header">Brent Oil Prices</div>
      <div className="card-body">
        <Table striped bordered hover>
          <thead>
            <tr>
              {data.length > 0 &&
                Object.keys(data[0]).map((key) => <th key={key}>{key}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr key={index}>
                {Object.values(row).map((value, i) => (
                  <td key={i}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    </div>
  );
}

export default BrentOilCard;
