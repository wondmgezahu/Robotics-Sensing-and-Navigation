#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from imu_msg.msg import IMUmsg
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion
from scipy.spatial.transform import Rotation as R
import serial

class IMUReader(Node):
    def __init__(self):
        super().__init__('imu_reader')
        self.start_time=time.time()
        # Parameters for serial port and baudrate
        self.serial_port = self.declare_parameter('port', '/dev/ttyUSB0').value
        self.get_logger().info(f'Using serial port: {self.serial_port}')
        self.serial_baud = self.declare_parameter('baudrate', 115200).value
        
        # Create publisher
        self.publisher = self.create_publisher(IMUmsg, 'imu', 10)
        
        # Initialize serial connection
        try:
            self.serial = serial.Serial(self.serial_port, self.serial_baud, timeout=1.0)
            # Configure IMU to output at 40 Hz
            self.configure_imu_rate(40)
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise
            
        # Create timer for reading IMU data (slightly faster than 40Hz to not miss messages)
        self.timer = self.create_timer(0.02, self.read_imu_data)
        self.get_logger().info('IMU driver initialized successfully')

    def calculate_checksum(self, message):
        """Calculate the checksum for VectorNav messages"""
        checksum = 0
        for char in message[1:]:  # Skip the leading $
            checksum ^= ord(char)
        return f"{checksum:02X}"  # Return as two-digit hexadecimal

    def configure_imu_rate(self, rate_hz):
        """Configure the IMU output rate in Hz"""
        # Command to set the async output frequency (register 07)
        command = f"$VNWRG,07,{rate_hz}"
        
        # Calculate and append checksum
        checksum = self.calculate_checksum(command)
        command = f"{command}*{checksum}\r\n"
        
        self.get_logger().info(f'Sending command: {command.strip()}')
        
        try:
            self.serial.write(command.encode())
            time.sleep(0.1)  # Short delay to allow IMU to process
            
            # Read response
            response = self.serial.readline().decode(errors='ignore').strip()
            self.get_logger().info(f'Response: {response}')
            
            if not response.startswith('$VNWRG'):
                self.get_logger().warn(f'Unexpected response: {response}')
        except serial.SerialException as e:
            self.get_logger().error(f'Error configuring IMU: {e}')
            raise

    def read_imu_data(self):
        """Timer callback to read and process IMU data"""
        try:
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith('$VNYMR'):
                    imu_data = self.parse_raw_data(line)
                    if imu_data:
                        self.publish_imu_data(imu_data)
        except serial.SerialException as e:
            self.get_logger().error(f'Serial read error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def parse_raw_data(self, line):
        """
        Parse the VNYMR string:
        $VNYMR,<Yaw>,<Pitch>,<Roll>,<MagX>,<MagY>,<MagZ>,<AccX>,<AccY>,<AccZ>,<GyroX>,<GyroY>,<GyroZ>*<Checksum>
        """
        try:
            # Split string and remove checksum
            parts = line.split(',')
            #if len(parts) < 13:  # Expect at least 13 parts (including header)
            #    self.get_logger().warn(f'Invalid VNYMR string format: {line}')
            #    return None
                
            # Remove checksum from last field
            parts[-1] = parts[-1].split('*')[0]
            
            # Parse values
            yaw = float(parts[1])
            pitch = float(parts[2])
            roll = float(parts[3])
            
            magX = float(parts[4])
            magY = float(parts[5])
            magZ = float(parts[6])
            
            accX = float(parts[7])
            accY = float(parts[8])
            accZ = float(parts[9])
            
            gyroX = float(parts[10])
            gyroY = float(parts[11])
            gyroZ = float(parts[12])
            
            # Create custom message
            msg = IMUmsg()
            
            # Get current time
            current_time = self.get_clock().now().to_msg()
            # Set main header
            header = Header(
                stamp=current_time,
                frame_id="IMU1_Frame")
            """
            header=Header()
            header.frame_id="IMU1_Frame"
            header.stamp.sec=int(elapsed_time)
            header.stamp.nanosec=int((elapsed_time-int(elapsed_time)*1e9))
            """
            # Convert Euler angles (in degrees) to quaternion
            # VectorNav uses aerospace convention (ZYX order: yaw, pitch, roll)
            rpy_rad = np.radians([roll, pitch, yaw])
            quaternion = R.from_euler('xyz', rpy_rad).as_quat()
            #quaternion = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_quat()
            #@breakpoint()
            # Create IMU message
            imu_msg = Imu()
            imu_msg.header = header
            imu_msg.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
            imu_msg.angular_velocity = Vector3(x=gyroX, y=gyroY, z=gyroZ)
            imu_msg.linear_acceleration = Vector3(x=accX, y=accY, z=accZ)
            
            # Create MagneticField message
            mag_msg = MagneticField()
            mag_msg.header = header
            mag_msg.magnetic_field = Vector3(x=magX, y=magY, z=magZ)
            
            # Combine into custom message
            msg.header = header
            msg.imu = imu_msg
            msg.mag_field = mag_msg
            msg.raw_str = line
            
            return msg
            
        except ValueError as e:
            self.get_logger().warn(f'Error parsing VNYMR string: {e} - {line}')
            return None
        except Exception as e:
            self.get_logger().error(f'Unexpected error parsing VNYMR string: {e}')
            return None

    def publish_imu_data(self, imu_msg):
        """Publish the IMU data message"""
        self.publisher.publish(imu_msg)
        self.get_logger().debug(f'Published IMU data')

def main(args=None):
    rclpy.init(args=args)
    try:
        imu_reader = IMUReader()
        rclpy.spin(imu_reader)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()