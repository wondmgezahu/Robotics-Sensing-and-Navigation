import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utm

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions, ConverterOptions
from std_msgs.msg import Header
from gps_msg.msg import GPSmsg
from geopy.distance import geodesic
import folium

# Initialize ROS 2
rclpy.init()
path_to_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
path_to_bag = os.path.join(path_to_src, 'Data/gps_walk_250m/gps_bag_0.db3')

# Define storage and converter options for bag file
storage_options = StorageOptions(uri=path_to_bag, storage_id='sqlite3')
converter_options = ConverterOptions()

# Create the bag reader
reader = SequentialReader()
reader.open(storage_options, converter_options)

# Initialize an empty list to store messages
data = []

# Define the topic name (replace with actual topic)
topic_name = 'gps'

# Read messages from the bag file
while reader.has_next():
    topic, msg, t = reader.read_next()
    gps_msg = deserialize_message(msg, GPSmsg)
    if topic == topic_name:
        print(gps_msg)
        # Extract values from the custom message
        # if isinstance(msg, GPSmsg):
        #     print("Extracting values")
        header = gps_msg.header
        latitude = gps_msg.latitude
        longitude = -1 * gps_msg.longitude
        altitude = gps_msg.altitude
        hdop = gps_msg.hdop
        # Forgot to do -1 * long in driver for walking data, changed for other data collection so we had to do conversion again
        utm_easting, utm_northing, zone, letter = utm.from_latlon(latitude=latitude, longitude=longitude)
        utc = gps_msg.utc

        # Append data as a dictionary to the list
        data.append({
            'header_stamp': header.stamp.nanosec,  # to store the timestamp
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'hdop': hdop,
            'utm_easting': utm_easting,
            'utm_northing': utm_northing,
            'utc': utc,
            'zone': zone,
            'letter': letter
        })

# Convert the data list into a DataFrame
df = pd.DataFrame(data)

# Show the DataFrame
print(df)

m = folium.Map(location=[df['latitude'][0], df['longitude'][0]], zoom_start=15)

# Add markers for each GPS point
for _, row in df.iterrows():
    folium.Marker([row['latitude'], row['longitude']]).add_to(m)
m.save("gps_map_walking.html")

# Scaling the data
numeric_columns = df.select_dtypes(include=['number'])
df_scaled = numeric_columns - numeric_columns.iloc[0]

# Getting true line
known_start_utm_northing = 4690068.53
known_start_utm_easting = 328256.10
known_end_utm_northing = 4690219.59
known_end_utm_easting = 328403.14
known_pos_northing = [0,known_end_utm_northing - known_start_utm_northing]
known_pos_easting = [0,known_end_utm_easting - known_start_utm_easting]

# Scatterplot of Northing vs Easting
df_scaled.plot.scatter(x='utm_northing',y='utm_easting')
plt.plot(np.unique(df_scaled['utm_northing']),np.poly1d(np.polyfit(df_scaled['utm_northing'],df_scaled['utm_easting'],1))(np.unique(df_scaled['utm_northing'])))
plt.plot(known_pos_northing,known_pos_easting,'k')
plt.title('Scaled Northing vs Easting Data (Walking)')
plt.xlabel('Northing')
plt.ylabel('Easting')
plt.savefig('northing_walking.png')
plt.show()

# Calculating position error
# known_lat = 42.33869
# known_long = 71.08834
# pos_error = []
#
# for i, row in df.iterrows():
#     measured_lat = row['latitude']
#     measured_long = row['longitude']
#     error_lat = known_lat - measured_lat
#     error_long = known_long - measured_long
#     error_tot = np.sqrt(error_lat*error_lat + error_long*error_long)
#     pos_error.append(error_tot)

# Histogram of error between known and measured positions
# plt.hist(pos_error,bins=10,edgecolor='black')
# plt.title('Histogram of Error from Known to Measured Positions (Occluded)')
# plt.xlabel('Distance (Euclidian)')
# plt.ylabel('Frequency')
# plt.show()

# Scatterplot of altitude vs time
df.plot.scatter(x='utc',y='altitude')
plt.title('Altitude vs Time (Walking)')
plt.xlabel('Time (UTC)')
plt.ylabel('Altitude')
plt.savefig('altitude_walking.png')
plt.show()

# pos error analysis:

path_slope=(known_end_utm_easting-known_start_utm_easting)/(known_end_utm_northing-known_start_utm_northing)
pos_error=[]
for _, row in df.iterrows():
     measured_north = row['utm_northing']
     measured_east = row['utm_easting']
     c=	known_start_utm_easting-path_slope*known_start_utm_northing
     distance=abs(path_slope*measured_north-measured_east +c) /np.sqrt(path_slope**2 +1)
     pos_error.append(distance) 
pos_error=np.array(pos_error)
mean_error=np.mean(pos_error)
std_error=np.std(pos_error)
max_error=np.max(pos_error)
min_error=np.min(pos_error)

print(f"Mean Error: {mean_error} meters")
print(f"Std Error: {std_error} meters")
print(f"Max Error: {max_error} meters")
print(f"Min Error: {min_error} meters")

# Shutdown rclpy
rclpy.shutdown()
