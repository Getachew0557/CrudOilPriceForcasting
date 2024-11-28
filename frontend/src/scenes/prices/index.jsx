import React from 'react';
import PriceGraph from '../../components/PriceGraph';
// import PriceRange from '../../components/PriceRange';
import PriceDistribution from '../../components/PriceDistribution';
import { Container, Typography } from '@mui/material';

const Prices = () => {
    return (
        <Container>
            <Typography variant="h4" align="center" gutterBottom>Brent Oil Price Analysis</Typography>
            <PriceGraph />
            <PriceDistribution />
        </Container>
    );
};

export default Prices;
