"""
Dead Reckoning Navigation 
"""
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from gps_msg.msg import GPSmsg
from imu_msg.msg import IMUmsg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import rclpy
import os
import math

rclpy.init()

CIRCLE_BAG_PATH = "data/data_circle/data_going_in_circles.db3"
DRIVING_BAG_PATH = "data/data_driving/data_driving.db3"  
OUTPUT_DIR='plotsnew'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPS_TOPIC = "/gps"
IMU_TOPIC = "/vectornav"
STATIONARY_WINDOW = 0.5  
STATIONARY_STD_THRESH = 0.08  
ACCEL_THRESH = 0.05  
COMP_FILTER_ALPHA = 0.2  
YAW_FILTER_CUTOFF = 0.05  
GRAVITY = 9.81  


def setup_reader(bag_path):
    """Set up a ROS2 bag reader for the specified bag path."""
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    return reader

def wrap_to_pi(angle_rad):
    """Wrap angle to [-π, π]"""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def wrap_angle_deg(angle_deg):
    """Wrap angle to [-180, 180]"""
    return (angle_deg + 180) % 360 - 180

def low_pass_filter(data, cutoff_hz, fs):
    """Apply a low-pass Butterworth filter to the data."""
    b, a = butter(N=2, Wn=cutoff_hz / (fs / 2), btype='low')
    return filtfilt(b, a, data)

def high_pass_filter(data, cutoff_hz, fs):
    """Apply a high-pass Butterworth filter to the data."""
    b, a = butter(N=2, Wn=cutoff_hz / (fs / 2), btype='high')
    return filtfilt(b, a, data)

def bearing_between(lat1, lon1, lat2, lon2):
    """Calculate bearing between two lat/lon points in radians."""
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dLon)
    return np.arctan2(x, y)

def proper_unwrap(angles_deg):
    """Unwrap angles properly for integration, keeping the total rotation count."""
    angles_rad = np.radians(angles_deg)
    unwrapped_rad = np.unwrap(angles_rad)
    # Convert back to degrees
    return np.degrees(unwrapped_rad)

def calibrate_magnetometer(mag_data):
    """
    Calibrate magnetometer to correct for hard-iron and soft-iron effects.
    
    Parameters:
    mag_data (np.array): Nx2 array with magnetometer X and Y values
    
    Returns:
    corrected_data (np.array): Calibrated magnetometer data
    offset (np.array): Hard-iron correction vector (bias)
    transform (np.array): Soft-iron correction matrix
    """
    # Hard-iron correction 
    offset = np.array([
        (np.max(mag_data[:, 0]) + np.min(mag_data[:, 0])) / 2,
        (np.max(mag_data[:, 1]) + np.min(mag_data[:, 1])) / 2
    ])
    centered = mag_data - offset
    
    # Soft-iron correction 
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    eigvals = np.maximum(eigvals, 1e-6)
    scale = np.diag(1 / np.sqrt(eigvals))
    transform = eigvecs @ scale @ eigvecs.T
 
    corrected = (transform @ centered.T).T
    radius_corrected = np.sqrt(np.sum(corrected**2, axis=1))    
    return corrected, offset, transform

def plot_magnetometer_calibration(raw_data, corrected_data):
    """Plot raw vs. corrected magnetometer data."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(raw_data[:, 0], raw_data[:, 1], s=1, c='orange', alpha=0.5)
    plt.title(f"Magnetometer X-Y (Before Calibration)")
    plt.xlabel("Magnetic Field X (gauss)")
    plt.ylabel("Magnetic Field Y (gauss)")
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(corrected_data[:, 0], corrected_data[:, 1], s=1, c='blue', alpha=0.5)
    
    # Add a reference circle just to compare the corrected magnetometer
    circle_radius = np.mean(np.sqrt(np.sum(corrected_data**2, axis=1)))
    theta = np.linspace(0, 2*np.pi, 100)
    x = circle_radius * np.cos(theta)
    y = circle_radius * np.sin(theta)
    plt.plot(x, y, 'r--', label='Reference Circle')
    plt.scatter(0, 0, c='red', s=30, marker='+')
    
    plt.title(f"Magnetometer X-Y (After Calibration)")
    plt.xlabel("Magnetic Field X (gauss)")
    plt.ylabel("Magnetic Field Y (gauss)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mag_xy.png'))
    #plt.show()

def calibrate_from_circle_data():
    
    reader = setup_reader(CIRCLE_BAG_PATH)
    mag_list = []
    reader.seek(0)
    while reader.has_next():
        t, data, _ = reader.read_next()
        if t == IMU_TOPIC:
            msg = deserialize_message(data, IMUmsg)
            mx = msg.mag_field.magnetic_field.x * 1e4  # Convert to Gauss
            my = msg.mag_field.magnetic_field.y * 1e4
            mz = msg.mag_field.magnetic_field.z * 1e4
            mag_list.append([mx, my, mz])
    mag_data = np.array(mag_list)
    mag_data_2d = mag_data[:, 0:2]
    corrected_2d, offset_2d, transform_2d = calibrate_magnetometer(mag_data_2d)
    plot_magnetometer_calibration(mag_data_2d, corrected_2d)
    return offset_2d, transform_2d

def plot_magnetometer_time_series(times, raw_mag, calibrated_mag):
    """Plot raw vs calibrated magnetometer data over time."""
    plt.figure(figsize=(15, 9))
    
    # X component
    plt.subplot(3, 1, 1)
    plt.plot(times, raw_mag[:, 0], label='Raw', color='orange', alpha=0.7)
    plt.plot(times, calibrated_mag[:, 0], label='Calibrated', color='blue')
    plt.title("Magnetometer X over Time")
    plt.ylabel("Magnetic Field X (gauss)")
    plt.grid(True)
    plt.legend()
    
    # Y component
    plt.subplot(3, 1, 2)
    plt.plot(times, raw_mag[:, 1], label='Raw', color='orange', alpha=0.7)
    plt.plot(times, calibrated_mag[:, 1], label='Calibrated', color='blue')
    plt.title("Magnetometer Y over Time")
    plt.ylabel("Magnetic Field Y (gauss)")
    plt.grid(True)
    plt.legend()
    if raw_mag.shape[1] > 2:
        plt.subplot(3, 1, 3)
        plt.plot(times, raw_mag[:, 2], label='Raw', color='orange', alpha=0.7)
        plt.plot(times, calibrated_mag[:, 2], label='Calibrated', color='blue')
        plt.title("Magnetometer Z over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic Field Z (gauss)")
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mag_series.png'))
    #plt.show()
    plt.close()

def compute_corrected_yaw(offset, transform):
    reader = setup_reader(DRIVING_BAG_PATH)
    raw_mags = []
    corrected_mags = []
    raw_yaws = []
    corrected_yaws = []
    timestamps = []
    reader.seek(0)
    while reader.has_next():
        t, data, _ = reader.read_next()
        if t == IMU_TOPIC:
            msg = deserialize_message(data, IMUmsg)
            time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            mx = msg.mag_field.magnetic_field.x * 1e4
            my = msg.mag_field.magnetic_field.y * 1e4
            mz = msg.mag_field.magnetic_field.z * 1e4
            
            raw_mags.append([mx, my, mz])
            raw_yaw = np.arctan2(my, mx)
            raw_yaws.append(np.degrees(raw_yaw))
            
            # Apply correction
            centered = np.array([mx, my]) - offset
            corrected = transform @ centered
            corrected_mags.append(corrected)
            corrected_yaw = np.arctan2(corrected[1], corrected[0])
            corrected_yaws.append(np.degrees(corrected_yaw))
            timestamps.append(time)
    
    timestamps = np.array(timestamps)
    start_time = timestamps[0]
    timestamps -= start_time
    raw_mags = np.array(raw_mags)
    corrected_mags_2d = np.array(corrected_mags)
    raw_yaws = np.array(raw_yaws)
    corrected_yaws = np.array(corrected_yaws)
    plot_magnetometer_time_series(timestamps, raw_mags[:,:2], corrected_mags_2d)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, raw_yaws, label='Raw Yaw', color='salmon')
    plt.plot(timestamps, corrected_yaws, label='Corrected Yaw', color='green')
    plt.title("Raw vs. Calibrated Magnetometer Yaw")
    plt.ylabel("Yaw (degrees)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, proper_unwrap(raw_yaws), label='Raw Yaw (unwrapped)', color='salmon')
    plt.plot(timestamps, proper_unwrap(corrected_yaws), label='Corrected Yaw (unwrapped)', color='green')
    plt.title("Unwrapped Yaw Angle (for cumulative rotation analysis)")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw (degrees)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'yaw_cum.png'))
    #plt.show()
    plt.close()
    
    df = pd.DataFrame({
        "Time (s)": timestamps,
        "RawMagX": raw_mags[:, 0],
        "RawMagY": raw_mags[:, 1],
        "RawMagZ": raw_mags[:, 2] if raw_mags.shape[1] > 2 else np.zeros_like(timestamps),
        "CorrectedMagX": [m[0] for m in corrected_mags],
        "CorrectedMagY": [m[1] for m in corrected_mags],
        "RawYaw (rad)": np.radians(raw_yaws),
        "CorrectedYaw (rad)": np.radians(corrected_yaws),
        "RawYaw (deg)": raw_yaws,
        "CorrectedYaw (deg)": corrected_yaws
    })
    
    return df

def compare_mag_vs_gyro_yaw(mag_df):

    reader = setup_reader(DRIVING_BAG_PATH)
    
    gyro_timestamps = []
    gyro_z = []
    orientations = []
    
    reader.seek(0)
    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name == IMU_TOPIC:
            msg = deserialize_message(data, IMUmsg)
            
            # Timestamp (s)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            gyro_timestamps.append(t)
            
            # Gyroscope Z-axis (rad/s)
            gz = msg.imu.angular_velocity.z
            gyro_z.append(gz)
            
            # Orientation quaternion (for comparison)
            q = msg.imu.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            yaw_imu = np.arctan2(siny_cosp, cosy_cosp)
            orientations.append(np.degrees(yaw_imu))
    gyro_timestamps = np.array(gyro_timestamps)
    start_time = gyro_timestamps[0]
    gyro_timestamps -= start_time
    gyro_z = np.array(gyro_z)
    orientations = np.array(orientations)

    dt = np.mean(np.diff(gyro_timestamps))
    gyro_yaw_rad = cumulative_trapezoid(gyro_z, gyro_timestamps, initial=0)
    gyro_yaw_deg = np.degrees(gyro_yaw_rad)

    mag_yaw_interp = interp1d(
        mag_df['Time (s)'], 
        mag_df['CorrectedYaw (deg)'], 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    
    aligned_mag_yaw = mag_yaw_interp(gyro_timestamps)
    gyro_yaw_deg -= gyro_yaw_deg[0]
    aligned_mag_yaw -= aligned_mag_yaw[0]
    orientations -= orientations[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(gyro_timestamps, wrap_angle_deg(aligned_mag_yaw), 
             label='Magnetometer Yaw', color='blue')
    plt.plot(gyro_timestamps, wrap_angle_deg(gyro_yaw_deg), 
             label='Integrated Gyro Yaw', color='orange', linestyle='--')
    plt.plot(gyro_timestamps, wrap_angle_deg(orientations), 
             label='IMU Orientation (quaternion)', color='green', linestyle=':')
    plt.title("Yaw Angle Comparison: Magnetometer vs Integrated Gyro vs IMU")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'imu_yaw.png'))
    #plt.show()
    plt.close()

    df = pd.DataFrame({
        "Time (s)": gyro_timestamps,
        "GyroZ (rad/s)": gyro_z,
        "GyroYaw_Integrated (deg)": gyro_yaw_deg,
        "CorrectedMagYaw (deg)": aligned_mag_yaw,
        "IMU_Orientation (deg)": orientations
    })
    return df

def complementary_filter_yaw(mag_gyro_df):

    timestamps = mag_gyro_df["Time (s)"].values
    gyro_z = mag_gyro_df["GyroZ (rad/s)"].values
    mag_yaw_deg = mag_gyro_df["CorrectedMagYaw (deg)"].values
    imu_yaw_deg = mag_gyro_df["IMU_Orientation (deg)"].values
    
    # Sampling frequency
    dt = np.mean(np.diff(timestamps))
    fs = 1 / dt
    #breakpoint()
    #fs=0.1
    # Magnetometer low-pass filter
    mag_yaw_lp_deg = low_pass_filter(mag_yaw_deg, YAW_FILTER_CUTOFF, fs)
    
    # Gyro high-pass filter (the integrated angle)
    gyro_yaw_deg = mag_gyro_df["GyroYaw_Integrated (deg)"].values
    gyro_yaw_hp_deg = high_pass_filter(gyro_yaw_deg, YAW_FILTER_CUTOFF, fs)
    
    # Complementary filter
    # Method 1: Using low/high pass filters
    fused_yaw_1 = mag_yaw_lp_deg + gyro_yaw_hp_deg
    # Plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, mag_yaw_lp_deg, label="Low-pass Magnetometer", color='green', linewidth=1)
    plt.plot(timestamps, gyro_yaw_hp_deg, label="High-pass Gyro Integration", color='blue', linewidth=1)
    plt.plot(timestamps, fused_yaw_1, label="Complementary Filter (Method 1)", color='red', linewidth=2)
    plt.plot(timestamps, imu_yaw_deg, label="IMU Yaw (from orientation)", color='purple', linestyle='--', alpha=0.6)
    
    plt.title("Components of the Complementary Filter")
    text = f"Yaw_comp = {COMP_FILTER_ALPHA} * Mag_yaw + (1 - {COMP_FILTER_ALPHA}) * Gyro_yaw"
    plt.annotate(text, xy=(0.5, 0.9), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
                 ha='center', va='center')
    plt.ylabel("Yaw Angle (degrees)")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout() 
    plt.savefig(os.path.join(OUTPUT_DIR, 'compare_yaw.png'))
    plt.close()
    
    df = pd.DataFrame({
        "Time (s)": timestamps,
        "MagYaw_LP (deg)": mag_yaw_lp_deg,
        "GyroYaw_HP (deg)": gyro_yaw_hp_deg,
        "FusedYaw_Method1 (deg)": fused_yaw_1,
        #"FusedYaw_Method2 (deg)": fused_yaw_2,
        "IMU_Yaw (deg)": imu_yaw_deg,
        "GyroZ (rad/s)": gyro_z
    })
    
    return df

def estimate_velocity_from_acceleration():

    reader = setup_reader(DRIVING_BAG_PATH)
    
    timestamps = []
    accel_x = []
    reader.seek(0)
    while reader.has_next():
        topic_name, data, _ = reader.read_next()
        if topic_name == IMU_TOPIC:
            msg = deserialize_message(data, IMUmsg)
            
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            timestamps.append(t)
            ax = msg.imu.linear_acceleration.x # forward a
            accel_x.append(ax)
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    start_time = timestamps[0]
    timestamps -= start_time
    accel_x = np.array(accel_x)
    
    accel_x_detrended = detrend(accel_x)
    #  gravity correction (assuming slight tilt of IMU)
    gravity_component = np.mean(accel_x) * np.ones_like(accel_x)
    accel_x_corrected = accel_x - gravity_component
 
    dt = np.mean(np.diff(timestamps))
    
    velocity_raw = np.cumsum(accel_x) * dt
    velocity_detrended = np.cumsum(accel_x_detrended) * dt
    velocity_corrected = np.cumsum(accel_x_corrected) * dt
    # ZUPT
    accel_thresh = ACCEL_THRESH  
    velocity_clamped = np.zeros_like(accel_x_corrected)
    v = 0.0
    for i, a in enumerate(accel_x_corrected):
        if abs(a) < accel_thresh:
            v = 0.0  
        else:
            v += a * dt
        velocity_clamped[i] = v
    
    # High-pass filter 
    fs = 1 / dt
    cutoff_hz = 0.01  
    velocity_filtered = high_pass_filter(velocity_clamped, cutoff_hz, fs)
    
    # Calculate std
    window_size = int(1 / dt)  
    accel_std = np.zeros_like(accel_x_corrected)
    
    for i in range(len(accel_x_corrected)):
        start_idx = max(0, i - window_size)
        end_idx = i + 1
        accel_std[i] = np.std(accel_x_corrected[start_idx:end_idx])
        #accel_mag=np.sqrt(accel_x_corrected**2 +accel_y_corrected)

    std_thresh =0.03  
    is_stationary = accel_std < std_thresh
    

    velocity_zupt_improved = np.copy(velocity_corrected)
    for i in range(len(velocity_zupt_improved)):
        if is_stationary[i]:
            velocity_zupt_improved[i] = 0

    stationary_markers = np.zeros_like(timestamps)
    stationary_markers[is_stationary] = 1
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, accel_x, label='Raw X Acceleration', color='blue')
    plt.plot(timestamps, gravity_component, label='Gravity Component', color='red')
    plt.plot(timestamps, accel_x_corrected, label='Corrected Acceleration', color='green')
    plt.title("Forward Acceleration and Gravity Correction")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, velocity_raw, label='Raw Velocity (Uncorrected)', color='red')
    plt.plot(timestamps, velocity_detrended, label='Velocity (Bias Corrected)', color='green')
    plt.plot(timestamps, velocity_clamped, label='Velocity (ZUPT)', color='blue')

    
    plt.title("IMU Velocity Estimation (Raw, Corrected, and Scaled)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, velocity_clamped, label='Velocity with ZUPT', color='blue')
    plt.plot(timestamps, velocity_filtered, label='Velocity with ZUPT + Filtering', color='purple')
    plt.plot(timestamps, velocity_zupt_improved, label='Velocity with Improved ZUPT', color='green')
    
    plt.title("Comparison of ZUPT Methods")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'zupt.png'))
    #plt.show()
    plt.close()
    df = pd.DataFrame({
        "Time (s)": timestamps,
        "Acceleration_X_Raw (m/s²)": accel_x,
        "Acceleration_X_Detrended (m/s²)": accel_x_detrended,
        "Acceleration_X_Corrected (m/s²)": accel_x_corrected,
        "Velocity_X_Raw (m/s)": velocity_raw,
        "Velocity_X_Detrended (m/s)": velocity_detrended,
        "Velocity_X_ZUPT (m/s)": velocity_clamped,
        "Velocity_X_Filtered (m/s)": velocity_filtered,
        "Velocity_X_ZUPT_Improved (m/s)": velocity_zupt_improved,
        "Is_Stationary": stationary_markers
    })
    return df

def compare_gps_vs_imu_velocity(vel_df, driving_gps_data=None):    
    if driving_gps_data is None:
        # Read data from driving bag
        reader = setup_reader(DRIVING_BAG_PATH)
        gps_time, gps_lat, gps_lon = [], [], []
        reader.seek(0)
        while reader.has_next():
            topic, data, _ = reader.read_next()
            
            if topic == GPS_TOPIC:
                msg = deserialize_message(data, GPSmsg)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                gps_time.append(t)
                
                #original_longitude = msg.longitude
                #lon_deg = -math.floor(original_longitude)  # For negative L, floor(L) gives the next lower integer
                #corrected_longitude = (2 * lon_deg + original_longitude)
                # Get correct longitude format
                original_longitude = msg.longitude
                corrected_longitude = original_longitude
                
                # Handle negative longitude properly
                if original_longitude < 0:
                    lon_deg = -math.floor(abs(original_longitude))
                else:
                    lon_deg = math.floor(original_longitude)
                #lon_deg=corrected_longitude
                gps_lat.append(msg.latitude)
                gps_lon.append(lon_deg)
                
        gps_time = np.array(gps_time)
        start_time = gps_time[0]
        gps_time -= start_time
        gps_lat = np.array(gps_lat)
        gps_lon = np.array(gps_lon)
    else:
        gps_time = driving_gps_data.index.values
        gps_lat = driving_gps_data['latitude'].values
        gps_lon = driving_gps_data['longitude'].values
    
    gps_velocity = []
    gps_velocity_time = []
    
    for i in range(1, len(gps_time)):
        dt = gps_time[i] - gps_time[i - 1]
        if dt <= 0.001:  
            continue
        d = geodesic((gps_lat[i-1], gps_lon[i-1]), (gps_lat[i], gps_lon[i])).meters
        v = d / dt
        if v < 30:  
            gps_velocity.append(v)
            gps_velocity_time.append(gps_time[i])
    
    gps_velocity = np.array(gps_velocity)
    gps_velocity_time = np.array(gps_velocity_time)    
    # Smooth GPS velocity with moving average method
    window_size = 5  # Adjust based on GPS data rate
    gps_velocity_smoothed = np.zeros_like(gps_velocity)
    for i in range(len(gps_velocity)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(gps_velocity), i + window_size // 2 + 1)
        gps_velocity_smoothed[i] = np.mean(gps_velocity[start_idx:end_idx])
    gps_interp = interp1d(gps_velocity_time, gps_velocity_smoothed, 
                          bounds_error=False, fill_value=0)
    
    imu_time = vel_df["Time (s)"].values
    gps_velocity_on_imu_time = gps_interp(imu_time)
    if "Velocity_X_ZUPT (m/s)" in vel_df.columns:
        imu_velocity = vel_df['Velocity_X_ZUPT (m/s)'].values  
    else:
        imu_velocity = vel_df['Velocity_X_Raw (m/s)'].values
    
    gps_peaks = []
    imu_peaks = []
    for i in range(1, len(gps_velocity_on_imu_time)-1):
        if gps_velocity_on_imu_time[i] > 1.0 and \
        gps_velocity_on_imu_time[i] > gps_velocity_on_imu_time[i-1] and \
        gps_velocity_on_imu_time[i] > gps_velocity_on_imu_time[i+1]:
            gps_peaks.append(gps_velocity_on_imu_time[i])
            window = 5  # samples
            start = max(0, i-window)
            end = min(len(imu_velocity), i+window)
            if np.max(imu_velocity[start:end]) > 0.01:  
                imu_peaks.append(np.max(imu_velocity[start:end]))
                
    if len(gps_peaks) > 3 and len(imu_peaks) == len(gps_peaks):
        scale_factors = [g/i for g, i in zip(gps_peaks, imu_peaks)]
        scale_factor = np.median(scale_factors)
        print(f"Using peak-based scaling: {scale_factor:.4f}")
    else:
        valid = (gps_velocity_on_imu_time > 0.5) & (imu_velocity > 0.01)
        if np.sum(valid) > 10:
            scale_factor = np.mean(gps_velocity_on_imu_time[valid]) / np.mean(imu_velocity[valid])
        else:
            scale_factor = 1.0
    imu_velocity_scaled = imu_velocity * scale_factor
    
    df = pd.DataFrame({
        "Time (s)": imu_time,
        "GPS_Velocity (m/s)": gps_velocity_on_imu_time,
        "IMU_Velocity_Raw (m/s)": imu_velocity,
        "IMU_Velocity_Scaled (m/s)": imu_velocity_scaled,
        "Scale_Factor": scale_factor
    })
    
    if "Is_Stationary" in vel_df.columns:
        df["Is_Stationary"] = vel_df["Is_Stationary"].values
    plot_velocity_comparison(
        imu_time, 
        gps_velocity_time, 
        gps_velocity, 
        gps_velocity_smoothed,
        gps_velocity_on_imu_time,
        imu_velocity, 
        imu_velocity_scaled,
        "Is_Stationary" in vel_df.columns and vel_df["Is_Stationary"].values
    )
    return df

def plot_velocity_comparison(imu_time, gps_time, gps_vel, gps_vel_smooth, 
                           gps_on_imu_time, imu_vel, imu_vel_scaled, stationary=None):
    
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(gps_time, gps_vel, 'o', label="Raw GPS Velocity", 
             color='red', markersize=4, alpha=0.5)
    plt.plot(gps_time, gps_vel_smooth, label="GPS Velocity (Smoothed)", 
             color='green', linewidth=2)
    plt.title("Raw GPS Velocity vs Time")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    # Plot GPS vs IMU velocity
    plt.subplot(2, 1, 2)
    plt.plot(imu_time, gps_on_imu_time, label="GPS Velocity", 
             color='blue', linewidth=2)
    plt.plot(imu_time, imu_vel, '--', label="IMU Velocity (Raw)", 
             color='orange', alpha=0.6)
    plt.plot(imu_time, imu_vel_scaled, label="IMU Velocity (Adjusted)", 
             color='green', linewidth=2)
    
    # Mark stationary periods if available
    # if stationary is not None:
    #     for i in range(len(imu_time)):
    #         if stationary[i]:
    #             plt.axvline(x=imu_time[i], color='gray', linestyle=':', alpha=0.1)
    
    plt.title("Velocity Comparison: GPS vs. IMU (Adjusted)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vel_comp.png'))
    
    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(gps_on_imu_time, imu_vel_scaled, alpha=0.5)
    
    max_vel = max(np.max(gps_on_imu_time), np.max(imu_vel_scaled))
    plt.plot([0, max_vel], [0, max_vel], 'r--', label="1:1 Line")
    
    plt.title("Correlation between GPS and IMU Velocity Estimates")
    plt.xlabel("GPS Velocity (m/s)")
    plt.ylabel("IMU Velocity (m/s)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(OUTPUT_DIR, 'gps_imu_vel.png'))
    plt.close()

def compare_motion_model(vel_df, comp_filter_df):
    reader = setup_reader(DRIVING_BAG_PATH)
    
    imu_time = []
    accel_y = []
    reader.seek(0)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == IMU_TOPIC:
            msg = deserialize_message(data, IMUmsg)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            imu_time.append(t)
            accel_y.append(msg.imu.linear_acceleration.y)  
    
    imu_time = np.array(imu_time)
    start_time = imu_time[0]
    imu_time -= start_time
    accel_y = np.array(accel_y)
    accel_y = detrend(accel_y)
    
    comp_time = comp_filter_df["Time (s)"].values
    gyro_z = comp_filter_df["GyroZ (rad/s)"].values
    min_time = max(comp_time[0], imu_time[0])
    max_time = min(comp_time[-1], imu_time[-1])
    
    mask_imu = (imu_time >= min_time) & (imu_time <= max_time)
    imu_time_overlap = imu_time[mask_imu]
    accel_y_overlap = accel_y[mask_imu]

    omega_interp = interp1d(comp_time, gyro_z, bounds_error=False, fill_value=0)
    omega = omega_interp(imu_time_overlap)
    
    # Interpolate forward velocity
    vel_time = vel_df["Time (s)"].values
    vel_x = vel_df["IMU_Velocity_Scaled (m/s)"].values
    
    vel_interp = interp1d(vel_time, vel_x, bounds_error=False, fill_value=0)
    velocity_x = vel_interp(imu_time_overlap)
    omega_x_dot = omega * velocity_x
    
    correlation = np.corrcoef(omega_x_dot, accel_y_overlap)[0, 1]
    correlation_text = f"Correlation: {correlation:.2f}" if not np.isnan(correlation) else "Correlation: nan"
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(imu_time_overlap, omega, label="Angular Velocity (ω)", color='red')
    plt.title("Angular Velocity from Gyroscope")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(imu_time_overlap, velocity_x, label="Forward Velocity (Ẋ)", color='green')
    plt.title("Forward Velocity (from acceleration integration)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.subplot(3, 1, 3)
    plt.plot(imu_time_overlap, omega_x_dot, label="ω·Ẋ (expected y-accel)", color='blue')
    plt.plot(imu_time_overlap, accel_y_overlap, label="y̋obs (measured y-accel)", 
             color='red', alpha=0.5)
    plt.title(f"Comparison of ω*Ẋ vs ÿobs ({correlation_text})")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(True)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'acc.png'))
    #plt.show()
    plt.close()
    # Create DataFrame
    df = pd.DataFrame({
        "Time (s)": imu_time_overlap,
        "omega (rad/s)": omega,
        "velocity_x (m/s)": velocity_x,
        "omega_x_dot (m/s²)": omega_x_dot,
        "accel_y (m/s²)": accel_y_overlap,
        "correlation": correlation
    })
    
    return df

def estimate_trajectory(yaw_df, vel_df):
    reader = setup_reader(DRIVING_BAG_PATH)
    
    gps_data = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == GPS_TOPIC:
            msg = deserialize_message(data, GPSmsg)
            gps_data.append({
                "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "utm_easting": msg.utm_easting,
                "utm_northing": msg.utm_northing
            })
    
    gps_df = pd.DataFrame(gps_data)
    gps_df["timestamp"] -= gps_df["timestamp"].iloc[0]
    
    theta_rad = np.radians(yaw_df["FusedYaw_Method1 (deg)"].values)
    yaw_time = yaw_df["Time (s)"].values
    velocity_x = vel_df["IMU_Velocity_Scaled (m/s)"].values
    vel_time = vel_df["Time (s)"].values
    
    velocity_x = np.clip(velocity_x, -15, 15)  
    min_time = max(yaw_time[0], vel_time[0], gps_df["timestamp"].min())
    max_time = min(yaw_time[-1], vel_time[-1], gps_df["timestamp"].max())
    
    dt = np.diff(vel_time)
    dt = np.append(dt[0], dt)  
    east = np.zeros_like(velocity_x)
    north = np.zeros_like(velocity_x)
    yaw_at_vel_time = np.interp(vel_time, yaw_time, theta_rad)
    for i in range(1, len(velocity_x)):
        v_east = velocity_x[i] * np.sin(yaw_at_vel_time[i])
        v_north = velocity_x[i] * np.cos(yaw_at_vel_time[i])
        
        east[i] = east[i-1] + v_east * dt[i]
        north[i] = north[i-1] + v_north * dt[i]

    gps_east = gps_df["utm_easting"].values - gps_df["utm_easting"].iloc[0]
    gps_north = gps_df["utm_northing"].values - gps_df["utm_northing"].iloc[0]
    gps_time = gps_df["timestamp"].values
    gps_east_interp = np.interp(vel_time, gps_time, gps_east)
    gps_north_interp = np.interp(vel_time, gps_time, gps_north)
    
    position_error = np.sqrt((east - gps_east_interp)**2 + (north - gps_north_interp)**2)
    plt.figure(figsize=(10, 5))
    plt.plot(vel_time, position_error)
    plt.axhline(y=2.0, color='r', linestyle='--', label="2.0m Error Threshold")
    plt.title("Dead Reckoning Position Error Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deadreck.png'))
    plt.close()
    
    if np.any(position_error > 2.0):
        idx = np.where(position_error > 2.0)[0][0]    
    plt.figure(figsize=(10, 8))
    plt.plot(east, north, '-', label="Dead Reckoning (IMU)", color='blue', linewidth=2)
    plt.plot(gps_east, gps_north, '-', label="GPS", color='red', linewidth=2)
    plt.plot(0, 0, 'ko', markersize=10, label="Start")
    
    plt.title("Trajectory Comparison: Dead Reckoning vs. GPS")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'traj.png'))
    plt.close()
    return east, north, vel_time

def main():
    offset, transform = calibrate_from_circle_data()
    yaw_df = compute_corrected_yaw(offset, transform)
    mag_vs_gyro_df = compare_mag_vs_gyro_yaw(yaw_df)
    comp_filter_df = complementary_filter_yaw(mag_vs_gyro_df)
    
    vel_from_acc_df = estimate_velocity_from_acceleration()
    vel_comparison_df = compare_gps_vs_imu_velocity(vel_from_acc_df)
    
    model_comparison_df = compare_motion_model(vel_comparison_df, comp_filter_df)
    trajectory_df = estimate_trajectory(comp_filter_df, vel_comparison_df)
    return {
        "magnetometer_calibration": (offset, transform),
        "yaw_data": {
            "corrected_yaw": yaw_df,
            "mag_vs_gyro": mag_vs_gyro_df,
            "complementary_filter": comp_filter_df
        },
        "velocity_data": {
            "from_acceleration": vel_from_acc_df,
            "comparison": vel_comparison_df
        },
        "dead_reckoning": {
            "model_comparison": model_comparison_df,
            "trajectory": trajectory_df
        }
    }
if __name__ == "__main__":
    main()