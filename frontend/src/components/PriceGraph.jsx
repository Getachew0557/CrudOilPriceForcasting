import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Box, Typography, Paper } from '@mui/material';

import { ResponsiveLine } from "@nivo/line";
import { useTheme } from "@mui/material";
import { tokens } from "../theme";

const PriceGraph = ({ isCustomLineColors = false, isDashboard = false }) => {
    const theme = useTheme();
    const colors = tokens(theme.palette.mode);
    const [data, setData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            const result = await axios.get('http://localhost:5000/api/prices');
            setData(result.data);
        };
        fetchData();
    }, []);

    return (
        <Paper elevation={3} sx={{ padding: 2, margin: 2 }}>
            <Typography variant="h6" gutterBottom>Brent Oil Price Graph</Typography>
            <Box display="flex" justifyContent="center">
                <LineChart width={600} height={300} data={data}>
                    <XAxis dataKey="Date" />
                    <YAxis />
                    <Tooltip />
                    <CartesianGrid strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="Price" stroke="#1976d2" />
                </LineChart>
            </Box>
        </Paper>
        
    );
};

export default PriceGraph;
