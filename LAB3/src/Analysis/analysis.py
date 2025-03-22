import pandas as pd
import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy import stats
from rclpy.serialization import deserialize_message
from imu_msg.msg import IMUmsg
import seaborn as sns


def load_bag_data(bag_path, is_stationary=True):
    """Load data from a ROS2 bag file"""
    print(f"Loading data from {bag_path}...")
    
    # Open the ROS 2 bag
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))

    # Find the IMU topic
    topic_name = None
    topics = reader.get_all_topics_and_types()
    for topic in topics:
        if "imu" in topic.name.lower():  
            topic_name = topic.name
            break

    if not topic_name:
        raise ValueError("No IMU topic found in the bag.")
    
    print(f"Found IMU topic: {topic_name}")

    # Read the IMU data
    timestamps, imu_data = [], []
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            data = deserialize_message(data, IMUmsg)

            timestamps.append(t / 1e9)  # Convert nanoseconds to seconds
            imu_data.append({
                "gyro_x": data.imu.angular_velocity.x,
                "gyro_y": data.imu.angular_velocity.y,
                "gyro_z": data.imu.angular_velocity.z,
                "accel_x": data.imu.linear_acceleration.x,
                "accel_y": data.imu.linear_acceleration.y,
                "accel_z": data.imu.linear_acceleration.z + 9.81,
                "orient_x": data.imu.orientation.x,
                "orient_y": data.imu.orientation.y,
                "orient_z": data.imu.orientation.z,
                "orient_w": data.imu.orientation.w,
                "mag_x": data.mag_field.magnetic_field.x,
                "mag_y": data.mag_field.magnetic_field.y,
                "mag_z": data.mag_field.magnetic_field.z,
            })

    # Adjust timestamps to start at 0
    start_time = timestamps[0] if timestamps else 0
    #breakpoint()
    timestamps = [t - start_time for t in timestamps]
    print(f"Loaded {len(timestamps)} IMU samples covering {timestamps[-1] - timestamps[0]:.2f} seconds")
    #print(len(timestamps)/40)
    #breakpoint()
    imu_df = pd.DataFrame(imu_data, index=timestamps)
    file_label = "stationary" if is_stationary else "dynamic"
    return imu_df, file_label
def quaternion_to_euler(df):
    """Convert quaternion orientation to Euler angles (roll, pitch, yaw)"""
    euler_angles = []
    for _, row in df.iterrows():
        quat = [row["orient_x"], row["orient_y"], row["orient_z"], row["orient_w"]]
        r = R.from_quat(quat)
        euler = r.as_euler("xyz", degrees=True)  # Convert to degrees
        euler_angles.append(euler)

    euler_df = pd.DataFrame(euler_angles,
                           columns=["roll", "pitch", "yaw"],
                           index=df.index)
    return euler_df    

def plot_time_series(df, title, columns, ylabel, filename, ylim=None, time_range=None):
    """
    Plot and save time series data with improved readability
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the data
    title (str): Plot title
    columns (list): List of column names to plot
    ylabel (str): Y-axis label
    filename (str): Output filename
    ylim (tuple): Optional y-axis limits (min, max)
    time_range (tuple): Optional time range to plot (start_time, end_time) in seconds
    """
    plt.figure(figsize=(14, 8))
    
    # Apply time range filter if specified
    if time_range is not None:
        start_time, end_time = time_range
        plot_df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
    else:
        plot_df = df
    
    # Plot each column 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, col in enumerate(columns):
        plt.plot(plot_df.index, plot_df[col], label=col, linewidth=2.5, 
                 color=colors[i % len(colors)])
    

    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    

    if ylim:
        plt.ylim(ylim)
    

    plt.legend(fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_histogram(df, columns, title, filename, bins=30):
    """
    Plot and save histograms for each column with improved readability
    """
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(columns):
        plt.subplot(1, len(columns), i+1)
        
        # Calculate values around median
        values = df[col] - df[col].median()
        
        # Plot histogram
        plt.hist(values, bins=bins, alpha=0.5, color='skyblue', label='Histogram')
        
        # Add KDE curve
        kde_x = np.linspace(values.min(), values.max(), 100)
        kde = stats.gaussian_kde(values)
        plt.plot(kde_x, kde(kde_x)*len(values) * (values.max()-values.min()) /bins,
                'b-', linewidth=2.5, label='KDE')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(values)
        x = np.linspace(values.min(), values.max(), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, pdf * len(values) * (values.max() - values.min()) / bins, 
                 'r-', linewidth=2.5, label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')  
        

        plt.xlabel(f"{col} Deviation from Median", fontsize=13)
        plt.ylabel("Frequency", fontsize=13)
        plt.title(f"{col} Distribution", fontsize=14, fontweight='bold')
        plt.legend(loc="upper left", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=11)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_motion_clip(df, euler_df, start_time, end_time, clip_title, filename_prefix):
    """
    Create detailed plots for a specific time clip to match with video frames
    
    Parameters:
    df (pandas.DataFrame): DataFrame with IMU data
    euler_df (pandas.DataFrame): DataFrame with Euler angles
    start_time (float): Start time of clip in seconds
    end_time (float): End time of clip in seconds
    clip_title (str): Title for the clip (e.g., "Forward Motion", "Rotation")
    filename_prefix (str): Prefix for the output filenames
    """
    # Ensure the time range is valid
    if start_time >= end_time or start_time < df.index.min() or end_time > df.index.max():
        print(f"Invalid time range: {start_time}-{end_time}s")
        return
    
    # Create a plot 
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Time range for filtering
    time_range = (start_time, end_time)
    
    # Filter data for the specific time range
    #breakpoint()
    #print(df.index)
    clip_df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
    clip_euler = euler_df.loc[(euler_df.index >= start_time) & (euler_df.index <= end_time)]
    
    # 1. Plot gyroscope data
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        axes[0].plot(clip_df.index, clip_df[col], linewidth=2.5, label=col)
    axes[0].set_ylabel("Angular Velocity (rad/s)", fontsize=14)
    axes[0].set_title("Gyroscope Data", fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=16)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='y', labelsize=12)
    
    # 2. Plot accelerometer data
    for col in ["accel_x", "accel_y", "accel_z"]:
        axes[1].plot(clip_df.index, clip_df[col], linewidth=2.5, label=col)
    axes[1].set_ylabel("Acceleration (m/s²)", fontsize=14)
    axes[1].set_title("Accelerometer Data", fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=16)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='y', labelsize=12)
    
    # 3. Plot Euler angles
    for col in ["roll", "pitch", "yaw"]:
        axes[2].plot(clip_euler.index, clip_euler[col], linewidth=2.5, label=col)
    axes[2].set_ylabel("Angle (degrees)", fontsize=14)
    axes[2].set_title("Orientation (Euler Angles)", fontsize=15, fontweight='bold')
    axes[2].legend(fontsize=16)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].tick_params(axis='y', labelsize=12)
    
    # 4. Plot magnetometer data
    for col in ["mag_x", "mag_y", "mag_z"]:
        axes[3].plot(clip_df.index, clip_df[col], linewidth=2.5, label=col)
    axes[3].set_xlabel("Time (s)", fontsize=14)
    axes[3].set_ylabel("Magnetic Field (T)", fontsize=14)
    axes[3].set_title("Magnetometer Data", fontsize=15, fontweight='bold')
    axes[3].legend(fontsize=16)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].tick_params(axis='both', labelsize=12)
    
    # Add overall title and timestamp
    plt.suptitle(f"Motion Clip: {clip_title} ({start_time:.1f}s - {end_time:.1f}s)", 
                fontsize=18, fontweight='bold', y=0.98)
    #plt.figtext(0.02, 0.01, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
     #          fontsize=9, alpha=0.6)
    
    # Adjust spacing
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(f"{filename_prefix}_{int(start_time)}_{int(end_time)}.png", dpi=300)
    plt.close()
    
    print(f"Created clip visualization for {start_time:.1f}s - {end_time:.1f}s")    

def analyze_imu_data(bag_path, is_stationary=True):
    """Analyze IMU data from a bag file"""
    # Load data
    imu_df, file_label = load_bag_data(bag_path, is_stationary)
    #breakpoint()
    # Convert quaternion to Euler angles
    euler_df = quaternion_to_euler(imu_df)
    #combined_df=pd.concat([imu_df,euler_df],axis=1)
    # Plot time series
    print("\nGenerating time series plots...")
    plot_time_series(imu_df, "Gyroscope Data", ["gyro_x", "gyro_y", "gyro_z"],
                    "Angular Velocity (rad/s)", f"{file_label}_gyro.png")
    
    plot_time_series(imu_df, "Accelerometer Data", ["accel_x", "accel_y", "accel_z"],
                    "Acceleration (m/s²)", f"{file_label}_accel.png")
    
    plot_time_series(imu_df, "Magnetometer Data", ["mag_x", "mag_y", "mag_z"],
                    "Magnetic Field (T)", f"{file_label}_mag.png")
    
    plot_time_series(euler_df, "Orientation (Euler Angles)", ["roll", "pitch", "yaw"],
                    "Angle (degrees)", f"{file_label}_euler.png")
    
    # Calculate mean and median
    mean_euler = euler_df.mean()
    median_euler = euler_df.median()
    
    print("\n===== Orientation Statistics =====")
    print("Mean Euler Angles:")
    print(mean_euler)
    print("\nMedian Euler Angles:")
    print(median_euler)
    
    # Plot histograms of orientation around median
    print("\nGenerating histogram plots...")
    plot_histogram(euler_df, ["roll", "pitch", "yaw"], 
                  "Euler Angle Distribution Around Median",
                  f"{file_label}_euler_histogram.png")

    return euler_df

def analyze_dynamic_data_with_video(bag_path):
    """
    Analyze dynamic motion data and create clip visualizations for video matching
    
    Parameters:
    bag_path (str): Path to the ROS bag file with dynamic motion data
    """
    print(f"Loading dynamic motion data from {bag_path}...")
    
    # Load the dynamic data
    imu_df, _ = load_bag_data(bag_path, is_stationary=False)
    
    # Convert quaternion to Euler angles
    euler_df = quaternion_to_euler(imu_df)
    #breakpoint()
    # 1. First, plot the full time series 
    print("\nGenerating full dynamic time series plots...")
    plot_time_series(imu_df, "Dynamic Motion - Gyroscope Data", 
                    ["gyro_x", "gyro_y", "gyro_z"],
                    "Angular Velocity (rad/s)", "dynamic_gyro_full.png")
    
    plot_time_series(imu_df, "Dynamic Motion - Accelerometer Data", 
                    ["accel_x", "accel_y", "accel_z"],
                    "Acceleration (m/s²)", "dynamic_accel_full.png")
    
    plot_time_series(euler_df, "Dynamic Motion - Orientation (Euler Angles)", 
                    ["roll", "pitch", "yaw"],
                    "Angle (degrees)", "dynamic_euler_full.png")
    
    # 2. Identify three motion clips 
    clip_segments = [
        # Format: (start_time, end_time, description)
        (50, 55, "First Motion Segment"),   
        (150, 155, "Second Motion Segment"),  
        (170, 175, "Third Motion Segment"), 
    ]
    
    print("\nGenerating detailed clip visualizations for video matching...")
    for start_time, end_time, description in clip_segments:
        plot_motion_clip(
            imu_df, 
            euler_df, 
            start_time, 
            end_time, 
            description, 
            "dynamic_clip"
        )
    
    print("\nDynamic data analysis complete. Generated full time series plots and clips for video matching.")
    
    return imu_df, euler_df

if __name__ == "__main__":
    # Analyze stationary data 
    stationary_bag_path = "src/Data/stationary_b.db3"
    stationary_euler = analyze_imu_data(stationary_bag_path, is_stationary=True)
    
    # Analyze dynamic data 
    dynamic_bag_path = "src/Data/5_min.db3"
    dynamic_imu, dynamic_euler = analyze_dynamic_data_with_video(dynamic_bag_path)
    
    print("\nAnalysis complete!")

  