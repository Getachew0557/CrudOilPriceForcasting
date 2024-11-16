// src/api/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const fetchPriceData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/oil-prices`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching price data:', error);
  }
};

export const fetchEventData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/events`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching event data:', error);
  }
};
