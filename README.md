# Weather-Based Flight Delay Prediction Model 
## Mhairi Crooks, Gwennan Drouillet, Craig Miller
This model and associated research was conducted as part of the Machine Learning Practical Course (2024/25) at the University of Edinburgh within the School of Informatics.

Our model attempts to predict the existence and severity of any flight delays using a combination of weather/flight data describing conditions between January 2010 and July 2018 at New York's John F. Kennedy Airport. We apply an ensemble of Random Forrest and LSTM (Long-Short Term Memory) methods in an effort to improve previous individual attempts at the problem.

The final dataset was gathered manually and consists of two separate sources:
1. National Oceanic and Atmospheric Administration (NOAA) - JFK Weather Data: https://developer.ibm.com/exchanges/data/all/jfk-weather-data/
2. Bureau of Transportation Statistics (BTS) - Airline On-Time Performance Data: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr