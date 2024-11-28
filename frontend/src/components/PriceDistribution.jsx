import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Box, Typography, Paper } from '@mui/material';

const PriceDistribution = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            const result = await axios.get('http://localhost:5000/api/prices');
            const priceCounts = result.data.reduce((acc, curr) => {
                const price = Math.round(curr.Price);
                acc[price] = (acc[price] || 0) + 1;
                return acc;
            }, {});
            const formattedData = Object.keys(priceCounts).map((key) => ({
                price: key,
                count: priceCounts[key],
            }));
            setData(formattedData);
        };
        fetchData();
    }, []);

    return (
        <Paper elevation={3} sx={{ padding: 2, margin: 2 }}>
            <Typography variant="h6" gutterBottom>Price Distribution</Typography>
            <Box display="flex" justifyContent="center">
                <BarChart width={600} height={300} data={data}>
                    <XAxis dataKey="price" />
                    <YAxis />
                    <Tooltip />
                    <CartesianGrid strokeDasharray="3 3" />
                    <Bar dataKey="count" fill="#1976d2" />
                </BarChart>
            </Box>
        </Paper>
    );
};

export default PriceDistribution;
