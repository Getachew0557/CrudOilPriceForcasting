import React, { useState } from 'react';
import axios from 'axios';
import { Box, Button, TextField, Typography, Paper } from '@mui/material';

const PriceRange = () => {
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [filteredData, setFilteredData] = useState([]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const result = await axios.get('http://localhost:5000/api/prices/range', {
            params: { start_date: startDate, end_date: endDate },
        });
        setFilteredData(result.data);
    };

    return (
        <Paper elevation={3} sx={{ padding: 2, margin: 2 }}>
            <Typography variant="h6" gutterBottom>Filter Prices by Date Range</Typography>
            <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', gap: 2 }}>
                <TextField
                    type="date"
                    label="Start Date"
                    InputLabelProps={{ shrink: true }}
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    required
                />
                <TextField
                    type="date"
                    label="End Date"
                    InputLabelProps={{ shrink: true }}
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    required
                />
                <Button type="submit" variant="contained" color="primary">Filter</Button>
            </Box>
            {filteredData.length > 0 && (
                <Box mt={2}>
                    <Typography variant="subtitle1">Filtered Prices</Typography>
                    <ul>
                        {filteredData.map((item, index) => (
                            <li key={index}>{`${item.Date}: $${item.Price}`}</li>
                        ))}
                    </ul>
                </Box>
            )}
        </Paper>
    );
};

export default PriceRange;
