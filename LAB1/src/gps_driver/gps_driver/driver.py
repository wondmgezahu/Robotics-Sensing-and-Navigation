#!/usr/bin/env python3
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from gps_msg.msg import GPSmsg
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message
import serial
import utm
import rosbag2_py


class GPSReader(Node):
    def __init__(self):
        super().__init__('gps_reader')

        # ROS publisher
        self.publisher = self.create_publisher(GPSmsg, '/gps', 10)

        # Parameters for serial port and baudrate
        self.serial_port = self.declare_parameter('port', '/dev/ttyUSB0').value
        self.serial_baud = self.declare_parameter('baudrate', 4800).value
        self.serial = serial.Serial(self.serial_port, self.serial_baud)

        # Timer for publishing data
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Setup rosbag writer 
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri='gps_bag',
                                                    storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        topic_info = rosbag2_py.TopicMetadata(
            name='gps',
            type='gps_msg/msg/GPSmsg',
            serialization_format='cdr')
        self.writer.create_topic(topic_info)

        # Shared data
        self.gps_data_queue = queue.Queue(maxsize=1)

        # Start a thread to handle serial data
        self.processor_thread = threading.Thread(target=self.main_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def main_loop(self):
        """
        Main loop for managing serial communication
        """

        try:
            while rclpy.ok():
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    self.get_logger().info(f'Read line: {line}')

                    # Parse the NMEA sentence here (for example, a GPGGA sentence)
                    if line.startswith('$GPGGA'):
                        gps_data = self.parse_gpgga(line)
                        self.get_logger().info(f"Parsed GPS Data: {gps_data}")
			#self.publisher.publish(msg)
                        if gps_data:
                            if self.gps_data_queue.full():
                                self.gps_data_queue.get()
                            self.gps_data_queue.put(gps_data)
                # time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f"Error in main loop: {e}")
            self.serial.close()

    def timer_callback(self):
        # Every second, check if we have new GPS data
        self.get_logger().info(f"Timer triggered at: {time.time()}")

        if not self.gps_data_queue.empty():
            gps_data = self.gps_data_queue.get()
            self.get_logger().info(f"Publishing GPS data: {gps_data}")
            self.publish_gps_data(gps_data)

    def parse_gpgga(self, sentence):
        # Example parsing for GPGGA: $GPGGA,123456.78,3723.2475,N,12202.2551,W,1,08,1.0,10.0,M,-34.0,M,,*7C
        # sentence = "$GPGGA,123456.78,3723.2475,N,12202.2551,W,1,08,1.0,10.0,M,-34.0,M,,*7C"
        fields = sentence.split(",")
        print(fields)

        time_UTC = float(fields[1]) if fields[1] else 0.0
        latitude = float(fields[2]) if fields[2] else None
        lat_dir=fields[3] if fields[3] else ""
        longitude = float(fields[4]) if fields[4] else None
        lon_dir=fields[5] if fields[5] else ""
        HDOP = float(fields[8]) if fields[8] else None
        altitude = float(fields[9])  if fields[9] else None

        if not latitude or not longitude or not altitude:
            return None

        lat_deg = int(latitude / 100)
        lat_min = latitude - (lat_deg * 100)
        lat_decimal = lat_deg + (lat_min / 60)
        if lat_dir=='S':
            lat_decimal=-1*lat_decimal
        lon_deg = int(longitude / 100)
        lon_min = longitude - (lon_deg * 100)
        lon_decimal = lon_deg + (lon_min / 60)
        if lon_dir=='W':
            lon_decimal=-1*lon_decimal
        print(lat_decimal)
        print(lon_decimal)
        # print(utm.from_latlon(latitude=lat_decimal, longitude=lon_decimal))
        utm_easting, utm_northing, zone, letter = utm.from_latlon(latitude=lat_decimal, longitude=lon_decimal)
        return {
            'latitude': lat_decimal,
            'longitude': lon_decimal,
            'altitude': altitude,
            'hdop': HDOP,
            'utm_easting': utm_easting,
            'utm_northing': utm_northing,
            'utc': time_UTC,
            'zone': zone,
            'letter': letter
        }
    def publish_gps_data(self, gps_data):
        # Create a NavSatFix message
        seconds = int(gps_data['utc'])  # Integer part gives seconds
        nanoseconds = int((gps_data['utc'] - seconds) * 1e9)
        msg = GPSmsg()
        msg.header = Header(frame_id="GPS1_Frame", stamp=Time(sec=seconds, nanosec=nanoseconds))
        # Loop through the fields and set the values in the message
        # for field in gps_data.keys():
        #    if field in gps_data:
        #        setattr(msg, field, gps_data[field])
        #    else:
        #        setattr(msg, field, 0.0)

        msg.latitude= gps_data['latitude']
        msg.longitude= gps_data['longitude']
        msg.altitude= gps_data['altitude']
        msg.hdop= gps_data['hdop']
        msg.utm_easting= gps_data['utm_easting']
        msg.utm_northing= gps_data['utm_northing']
        msg.zone=gps_data['zone']
        msg.letter=gps_data['letter']
        msg.utc=gps_data['utc']
        # Publish the message
        self.publisher.publish(msg)
        self.writer.write(
            'gps',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)
        self.get_logger().info(
            f"writer triggered at: {time.time()} and wrote: lat={msg.latitude}, long={msg.longitude},UTM=({msg.utm_easting},{msg.utm_northing})")


def main(args=None):
    rclpy.init(args=args)
    gps_reader = GPSReader()
    rclpy.spin(gps_reader)
    gps_reader.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
