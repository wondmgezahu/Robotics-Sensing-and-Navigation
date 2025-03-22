import numpy as np
from matplotlib import pyplot as plt
from rosbags.typesys import get_types_from_msg
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import os


# Create a typestore for ROS 1 Noetic
typestore = get_typestore(Stores.ROS1_NOETIC)

# Define missing message types
STRIDX_MSG = """
Header header
string data
"""
typestore.register(get_types_from_msg(STRIDX_MSG, 'rospy_tutorials/msg/HeaderString'))

def load_data_from_bag(bag_path, topic='/vectornav'):
    """Extract IMU data from ROS1 bag file"""
    timestamps = []
    gyro_x, gyro_y, gyro_z = [], [], []
    acc_x, acc_y, acc_z = [], [], []
    
    print(f"Loading data from {bag_path}...")
    
    with Reader(bag_path) as reader:
        # Topic and msgtype information
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)

        # Iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic and "VNYMR" in str(rawdata):
                try:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    fields = msg.data.split(",")
                    fields[-1] = fields[-1].split("*")[0]
                    
                    gyro_x.append(float(fields[10]) if fields[10] else None)
                    gyro_y.append(float(fields[11]) if fields[11] else None)
                    gyro_z.append(float(fields[12]) if fields[12] else None)

                    acc_x.append(float(fields[7]) if fields[7] else None)
                    acc_y.append(float(fields[8]) if fields[8] else None)
                    acc_z.append(float(fields[9]) if fields[9] else None)

                    timestamps.append(timestamp)
                except Exception as e:
                    print(f"Error parsing data: {e}")
    
    print(f"Loaded {len(timestamps)} data points")
    
    # Convert lists to NumPy arrays
    timestamps = np.array(timestamps)
    gyro_x = np.array(gyro_x)
    gyro_y = np.array(gyro_y)
    gyro_z = np.array(gyro_z)
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)
    
    return timestamps, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z

def allan_variance(theta, t0, maxNumM=100):
    """
    Computes Allan variance using the overlapping method.
    
    Parameters:
        theta (np.array): Integrated angular velocity data (cumulative sum of gyro).
        t0 (float): Sampling time interval.
        maxNumM (int): Number of logarithmically spaced averaging windows.
    
    Returns:
        tau (np.array): Averaging time intervals.
        avar (np.array): Allan variance values.
    """
    L = len(theta)
    maxM = 2**np.floor(np.log2(L / 2))
    m = np.logspace(np.log10(1), np.log10(maxM), maxNumM)  # Log-spaced m
    m = np.unique(np.ceil(m).astype(int))  # Ensure integers and remove duplicates
    
    tau = m * t0  # Compute tau values
    avar = np.zeros(len(m))
    
    for i, mi in enumerate(m):
        avar[i] = np.sum(
            (theta[2*mi:L] - 2*theta[mi:L-mi] + theta[0:L-2*mi])**2
        )
    
    avar = avar / (2 * tau**2 * (L - 2 * m))
    
    return tau, avar

def extract_allan_params(tau, adev):
    """
    Extracts Rate Random Walk (K), Angle Random Walk (N), and Bias Stability (B)
    from Allan variance using the approach from the original code.
    
    Parameters:
        tau (numpy array): Averaging times
        adev (numpy array): Allan deviation values
        
    Returns:
        tuple: (N, K, B) for Angle Random Walk, Rate Random Walk, and Bias Stability
    """
    # Convert to log scale
    logtau = np.log10(tau)
    logadev = np.log10(adev)
    dlogadev = np.diff(logadev) / np.diff(logtau)
    
    # Angle Random Walk (N)
    slope_arw = -0.5
    idx_arw = np.argmin(np.abs(dlogadev - slope_arw))
    b_arw = logadev[idx_arw] - slope_arw * logtau[idx_arw]
    N = 10**b_arw
    
    # Rate Random Walk (K)
    slope_rrw = 0.5
    idx_rrw = np.argmin(np.abs(dlogadev - slope_rrw))
    b_rrw = logadev[idx_rrw] - slope_rrw * logtau[idx_rrw]
    log_K = slope_rrw * np.log10(3) + b_rrw
    K = 10**log_K
    
    # Bias Stability (B) - using the original method
    slope = 0
    idx = np.argmin(np.abs(dlogadev - slope))
    b = logadev[idx] - slope * logtau[idx]
    scf_b = np.sqrt(2*np.log(2)/np.pi)
    log_b = b - np.log10(scf_b)
    B = 10**log_b
    
    return N, K, B


def analyze_imu_data(bag_path):
    """Analyze IMU data from a bag file"""
    # Load data
    _, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z = load_data_from_bag(bag_path)
    
    # Sampling time ( 40 Hz as mentioned in the lab)
    t0 = 1/40
    fig_gyro = plt.figure(figsize=(12, 8))
    plt.title("Gyroscope Allan Deviation", fontsize=16, fontweight='bold')
    plt.xlabel("Averaging Time (s)", fontsize=14)
    plt.ylabel("Allan Deviation (rad/s)", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    # Analyze gyroscope data
    print("\n--- Gyroscope Analysis ---")
    for axis, data, label in zip(['X', 'Y', 'Z'], [gyro_x, gyro_y, gyro_z], ['gyro_x', 'gyro_y', 'gyro_z']):
        # Integrate gyro data to get theta (angle)
        theta = np.cumsum(data) * t0
        
        # Compute Allan variance
        tau, avar = allan_variance(theta, t0)
        adev = np.sqrt(avar)  # Allan deviation
        
        # Extract parameters
        N, K, B = extract_allan_params(tau, adev)
    
        print(f"\n{axis}-axis Gyroscope Parameters:")
        print(f"  Angle Random Walk (N): {N:.2e} rad/√s")
        print(f"  Rate Random Walk (K): {K:.23} rad/s/√s")
        print(f"  Bias Stability (B): {B:.2e} rad/s")
    
        # Plot Allan deviation
        #plt.figure(figsize=(10, 6))
        plt.loglog(tau, adev, label=f"{axis}-axis")
        
        # # Mark key points
        # plt.scatter(1, N, color='r', marker='o', label='ARW (N)')
        # plt.scatter(3, 3*K, color='g', marker='s', label='RRW (K)')
        
        # # Add reference slopes
        # tau_range = np.logspace(-1, 3, 100)
        # plt.loglog(tau_range, N/np.sqrt(tau_range), 'r--', alpha=0.5, label='Slope = -1/2')
        # plt.loglog(tau_range, K*np.sqrt(tau_range), 'g--', alpha=0.5, label='Slope = +1/2')

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("gyro_combined_allan_deviation.png", dpi=300)
    plt.close()

    fig_accel = plt.figure(figsize=(12, 8))
    plt.title("Accelerometer Allan Deviation", fontsize=16, fontweight='bold')
    plt.xlabel("Averaging Time (s)", fontsize=14)
    plt.ylabel("Allan Deviation (m/s²)", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)

    # Analyze accelerometer data
    print("\n--- Accelerometer Analysis ---")
    for axis, data, label in zip(['X', 'Y', 'Z'], [acc_x, acc_y, acc_z], ['acc_x', 'acc_y', 'acc_z']):
        # Integrate accelerometer data
        theta = np.cumsum(data) * t0
        
        # Compute Allan variance
        tau, avar = allan_variance(theta, t0)
        adev = np.sqrt(avar)
        
        # Extract parameters
        N, K, B = extract_allan_params(tau, adev)
        
        print(f"\n{axis}-axis Accelerometer Parameters:")
        print(f"  Velocity Random Walk (N): {N:.2e} m/s/√s")
        print(f"  Acceleration Random Walk (K): {K:.2e} m/s²/√s")
        print(f"  Bias Stability (B): {B:.2e} m/s²")
        
        # Plot Allan deviation
        #plt.figure(figsize=(10, 6))
        plt.loglog(tau, adev, label=f"{axis}-axis")
        
        # # Mark key points
        # plt.scatter(1, N, color='r', marker='o', label='VRW (N)')
        # plt.scatter(3, 3*K, color='g', marker='s', label='ARW (K)')
        
        # # Add reference slopes
        # tau_range = np.logspace(-1, 3, 100)
        # plt.loglog(tau_range, N/np.sqrt(tau_range), 'r--', alpha=0.5, label='Slope = -1/2')
        # plt.loglog(tau_range, K*np.sqrt(tau_range), 'g--', alpha=0.5, label='Slope = +1/2')

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("accel_combined_allan_deviation.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Path to the bag file
    bag_path = "src/Data/LocationB.bag"
    
    # Analyze the data
    analyze_imu_data(bag_path)
