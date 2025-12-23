import streamlit as st
import pandas as pd
import folium 
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt

# HAVERSINE #

from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r


# SUODATUS #

from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# ACCELERATION #

df = pd.read_csv('Linear Acceleration.csv')

data = df['Linear Acceleration y (m/s^2)']
T_tot = df['Time (s)'].max() 
n = len(df['Time (s)'])
fs = n / T_tot
nyq = 0.5 * fs
order = 3
cutoff = 1/0.4
data_filt = butter_lowpass_filter(data, cutoff, nyq, order)

fig_filtered = plt.subplots(figsize=(12, 4))
plt.plot(df['Time (s)'], data_filt)
plt.axis([0, 100, -5, 5])
plt.title('Suodatettu data')
plt.xlabel('Aika (s)')
plt.ylabel('Kiihtyvyys (m/s²)')

jaksot = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] < 0:
        jaksot = jaksot + 1/2


# FOURIER #

signal =  df['Linear Acceleration y (m/s^2)']
t = df['Time (s)']
N = len(signal)
dt = np.max(t)/N

fourier = np.fft.fft(signal, N)
psd = fourier*np.conj(fourier)/N
freq = np.fft.fftfreq(N,dt)
L = np.arange(1,int(N/2))

fig_fourier = plt.subplots(figsize=(15,6))
plt.plot(freq[L],psd[L].real)
plt.xlabel('Taajuus (Hz)')
plt.ylabel('Teho')
plt.axis([0, 10, 0, 400])

f_max = freq[L][psd[L] == np.max(psd[L])][0]
T = 1/f_max

steps = f_max*np.max(t)


# GPS #

df_location = pd.read_csv('Location.csv')
df_location.head()

df_location = df_location[df_location['Horizontal Accuracy (m)'] < 4]
df_location = df_location.reset_index(drop=True)

lat1 = df_location['Latitude (°)'].mean()
lon1 = df_location['Longitude (°)'].mean()

my_map = folium.Map(location = [lat1, lon1], zoom_start = 17)

folium.PolyLine(df_location[['Latitude (°)', 'Longitude (°)']], color = 'red', weight = 3).add_to(my_map)

df_location['Distance_calc'] = np.zeros(len(df_location))

for i in range (len(df_location)-1):
    lon1 = df_location['Longitude (°)'][i]
    lon2 = df_location['Longitude (°)'][i+1]
    lat1 = df_location['Latitude (°)'][i]
    lat2 = df_location['Latitude (°)'][i+1]
    df_location.loc[i+1, 'Distance_calc'] = haversine(lon1, lat1, lon2, lat2)

df_location['total_distance'] = df_location['Distance_calc'].cumsum()

total_time = df_location['Time (s)'].iloc[-1] - df_location['Time (s)'].iloc[0]
keskinopeus_ms = (df_location['total_distance'].iloc[-1] * 1000) / total_time 
keskinopeus_kmh = keskinopeus_ms * 3.6

askelpituus = (df_location['total_distance'].iloc[-1] * 1000) / steps




# STREAMLIT #

st.title('Fysiikan loppuprojekti')

st.write('Askelmäärä laskettuna suodatuksen avulla:', jaksot)
st.write('Askelmäärä laskettuna Fourier-analyysin avulla:', steps)
st.write('Keskinopeus:', round(keskinopeus_kmh, 2), 'km/h')
st.write('Kokonaismatka:', round(df_location['total_distance'].iloc[-1], 2), 'km')
st.write('Askelpituus:', round(askelpituus, 2), 'm')

st.subheader('Suodatetun kiihtyvyysdatan y-komponentin kuvaaja')

st.pyplot(fig_filtered[0])

st.subheader('Tehospektri')

st.pyplot(fig_fourier[0])

st.subheader('Kartta GPS-datasta')

st_folium(my_map, width=700, height=500)