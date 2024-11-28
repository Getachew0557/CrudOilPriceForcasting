import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, Row, Col } from 'react-bootstrap';
import { Line, Pie, Bar } from 'react-chartjs-2';
import 'chart.js/auto';
import './Dashboard.css'; // Optional: Custom styles for dashboard

const Dashboard = () => {
  const [datasets, setDatasets] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const datasetNames = [
      "BrentOilPrices",
      "CalendarEvents",
      "ExchangeRatesAlpha",
      "WorldGDPGrowth"
    ];

    const fetchDatasets = async () => {
      try {
        const dataPromises = datasetNames.map(name =>
          axios.get(`http://localhost:5000/api/data/${name}`).then(res => ({ name, data: res.data }))
        );
        const results = await Promise.all(dataPromises);
        const dataMap = results.reduce((acc, { name, data }) => ({ ...acc, [name]: data }), {});
        setDatasets(dataMap);
      } catch (error) {
        console.error("Error fetching datasets:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, []);

  const renderPieChart = (data, labelKey, valueKey) => {
    if (!Array.isArray(data)) {
      return <p>Invalid data format for Pie Chart</p>;
    }

    const labels = data.map(item => item[labelKey]);
    const values = data.map(item => item[valueKey]);
    return (
      <Pie
        data={{
          labels,
          datasets: [
            {
              data: values,
              backgroundColor: [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
              ],
              hoverBackgroundColor: [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
              ],
            },
          ],
        }}
      />
    );
  };

  const renderBarChart = (data, xKey, yKey) => {
    if (!Array.isArray(data)) {
      return <p>Invalid data format for Bar Chart</p>;
    }
  
    // Dynamically extract the keys based on actual column names
    const labels = data.map(item => item[xKey]);  // xKey will be the column you want to use as x-axis (like 'Unnamed: 0' for dates)
    const values = data.map(item => item[yKey]);  // yKey will be the column to plot on y-axis (like 'Close' for exchange rate)
  
    return (
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: yKey,  // Dynamic label for the bar chart
              data: values,
              backgroundColor: '#36A2EB',
              borderColor: '#2980b9',
              borderWidth: 1,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
        }}
      />
    );
  };
  

  const renderLineChart = (data, xKey, yKey) => {
    if (!Array.isArray(data)) {
      return <p>Invalid data format for Line Chart</p>;
    }

    const labels = data.map(item => item[xKey]);
    const values = data.map(item => item[yKey]);
    return (
      <Line
        data={{
          labels,
          datasets: [
            {
              label: yKey,
              data: values,
              fill: false,
              backgroundColor: '#FF6384',
              borderColor: '#FF6384',
              tension: 0.1,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
        }}
      />
    );
  };
  
  const renderTable = (data) => {
    if (!Array.isArray(data)) {
      return <p>Invalid data format for Table</p>;
    }
  
    return (
      <div className="scrollable-table-container">
        <table className="table table-striped table-bordered">
          <thead>
            <tr>
              {Object.keys(data[0]).slice(0, 5).map((key, i) => (
                <th key={i}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 2597).map((row, idx) => (
              <tr key={idx}>
                {Object.values(row).slice(0, 5).map((value, i) => (
                  <td key={i}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  

  const renderDataset = (name, data) => {
    if (!data) return <p>No data available for {name}</p>;
    switch (name) {
      case "BrentOilPrices":
        return renderLineChart(data, "Date", "Price");
      case "CalendarEvents":
        return renderTable(data);
      case "ExchangeRatesAlpha":
        return renderBarChart(data, "Unnamed: 0", "Close");
      case "WorldGDPGrowth":
        return renderLineChart(data, "date", "GDP growth (annual %)");
      default:
        return <p>No visualization available</p>;
    }
  };

  if (loading) {
    return <div className="text-center mt-5">Loading data...</div>;
  }

  return (
    <div className="container">
      <h1 className="text-center my-4">Oil Price Dashboard</h1>
      <Row>
        {Object.entries(datasets).map(([name, data]) => (
          <Col md={6} lg={4} className="mb-4" key={name}>
            <Card>
              <Card.Header as="h5" className="text-center">{name}</Card.Header>
              <Card.Body style={{ height: "300px" }}>
                {data && data.length > 0 ? renderDataset(name, data) : <p>No data available</p>}
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default Dashboard;
