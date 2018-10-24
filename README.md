# Hydrological-tools

## Overview
Python object to run the CemaNeige snow accounting routine (Valéry et al 2014, see reference)

## Prerequisites
Python3

Packages: pandas, numpy, csv and numba 

## Data
Synthetic data for an imaginary catchment, called "MyCatchment".

**MyCatchment_CemaNeigeInfo.csv**: datafile containing CemaNeige parameters (QNBV: average annual snow accumulation [mm], AltiBand: quantiles of elevation [0, 0.25, 0.50, 0.75, 1], Z50: median altitude )

**MyCatchment_data.csv**: Time series of Dates (Date), total precipitation (p, [mm]) and air temperature (tair, [°C])

## Reference
Valéry A., Andréassian, V. and Perrin, C.: "As simple as possible but not simpler": What is useful in a temperature-based snow-accounting routine? Part 2 - Sensitivity analysis of the Cemaneige snow accounting routine on 380 catchments, Journal of Hydrology, 517, 1176-1187, doi={10.1016/j.jhydrol.2014.04.058}, 2014.

## Working exemple
```python

# Load objects
import pandas as pd
import CemaNeige

# Declarations
CtchName = 'myCatchment'

# Load data
data = pd.read_csv(datafile,sep=',',header='infer')
data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d %H:%M:%S')
Date = data['Date']
inSnow = np.array( (data['p'], data['tair']) )

# Run CemaNeige
SnowM = CemaNeige(CtchName)
melt = SnowM.RunModel(inSnow,Date)        
                
```
