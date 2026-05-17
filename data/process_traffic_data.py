import pandas as pd     # used to read the excel file and manipulate traffic count data
import numpy as np      # used for math operations when converting counts to arrival rates



# load the raw traffic count data from the excel file
df = pd.read_excel("data/Vol16_26 Rural Rd btwn Rio Salado Pkwy and University Dr.xlsx", 
                   sheet_name="Day 1", 
                   header=None)



# extract 15-minute interval data from columns 23 (time), 24 (northbound), 25 (southbound)
records = []                # empty list to fill with one dictionary per 15-minute interval
for i in range(4, 100):     # starts at row 4 because the first few rows are header metadata
    row = df.iloc[i]
    time_val = row[23]      # timestamp of the 15-minute interval
    nb_val = row[24]        # northbound vehicle count
    sb_val = row[25]        # southbound vehicle count

    # only process rows that are actual time entries
    # ':' check ensures we skip metadata rows and only grab timestamps
    if pd.notna(time_val) and ':' in str(time_val):
        try:
            nb = float(nb_val)
            sb = float(sb_val)
            records.append({'time': str(time_val), 'NB': nb, 'SB': sb})
        except:
            pass    # skip rows with non-numeric count values


# convert list of dictionaries into a clean dataframe
data = pd.DataFrame(records)


# extract hour from time string and group into hourly totals
# groups all rows by hour (ex: hour=7 [07:00, 07:15, 07:30, 07:45])
# -> gives the total NB and SB vehicles for that entire hour
data['hour'] = data['time'].apply(lambda x: int(x.split(':')[0]))
hourly = data.groupby('hour')[['NB', 'SB']].sum()


# 1 timestep = 15 seconds, 5760 timesteps = 24 hour episode
# divide hourly totals by 240 to get average vehicles arriving per 15 seconds
hourly['NS_rate'] = hourly['NB'] / 240
hourly['EW_rate'] = hourly['SB'] / 240


print(hourly[['NS_rate', 'EW_rate']].to_string())   # used to view the per minute traffic rates (cars arriving at the intersection)


# save hourly rates to a CSV file for the traffic environment to load
hourly[['NS_rate', 'EW_rate']].to_csv('data/arrival_rates.csv')     # saves the NS_rate and EW_rate columns to a CSV file
print("Arrival rates saved to data/arrival_rates.csv")              # prints a confirmation that arrival rates were saved successfully