import threading
import time
import queue
import logging
import csv
import os
from datetime import datetime
from payload import RadarDataParser
from plots import RadarVisualizer
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import zoom
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RadarApplication:
   def __init__(self):
       self._initialize_variables()
       self._setup_visualizations()


   def _initialize_variables(self):
       """Initialize shared variables and components."""
       self.data_queue = queue.Queue(maxsize=100)
       self.parser = RadarDataParser()
       self.radar_visualizer = RadarVisualizer()
       self.stop_event = threading.Event()
       # self.window = MainWindow()
       self.window = []


       self.epsilon = 0.005
       self.range_profile = []
       self.noise_profile = []
       self.range_doppler_heatmap = []
       self.azimuth_static_heatmap = []
       self.detected_points = []
       self.side_info_for_detected_points = []


       self.pos = None
       self.doppler = None
      
       # CSV logging variables
       self.csv_directory = None
       self.csv_files = {}
       self.csv_writers = {}
       self.csv_file_handles = {}
       self.frame_count = 0
       self.start_time_s = None  # Start time in seconds for stopwatch
       self.custom_folder_name = None  # Custom folder name for data
       self.received_frames = 0
       self.parsed_frames = 0


   def _setup_visualizations(self):
       """Setup plots for visualizing radar data."""


       self.enable_range_profile_plot = True
       if self.enable_range_profile_plot:
           self.range_profile_figure, self.range_profile_plot_objects = (
               self.radar_visualizer.generate_range_profile_plot())


       self.enable_range_doppler_heatmap = True
       if self.enable_range_doppler_heatmap:
           self.range_doppler_heatmap_figure, self.range_doppler_heatmap_item = (
               self.radar_visualizer.generate_heatmap_plot(
                   "Range-Doppler Heatmap", "Range [m]", "Speed [m/s]"
               )
           )


       self.enable_range_azimuth_heatmap = True
       if self.enable_range_azimuth_heatmap:
           self.range_azimuth_heatmap_figure, self.range_azimuth_heatmap_item = (
               self.radar_visualizer.generate_heatmap_plot(
                   "Range-Azimuth Static Heatmap (at Doppler = 0)",
                   "Distance [m]", "Range [m]"
           )
       )


   def set_csv_directory(self, csv_dir, folder_name=None):
       """Set the CSV directory for data logging.


       Args:
           csv_dir: Base directory path (usually "Data")
           folder_name: Custom name for the data folder (if None, uses timestamp)
       """
       self.csv_directory = csv_dir
       self.custom_folder_name = folder_name
       self.start_time_s = None  # Will be set on first frame
       print(f"✓ CSV logging enabled. Directory: {os.path.abspath(csv_dir)}")
       self._setup_csv_files()


   def _setup_csv_files(self):
       """Setup CSV files for different data types."""
       if not self.csv_directory:
           return


       # Use custom folder name or generate timestamp
       if self.custom_folder_name:
           folder_name = self.custom_folder_name
       else:
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           folder_name = f"radar_data_{timestamp}"


       # Create Data directory with subfolder
       data_dir = os.path.join("Data", folder_name)
       if not os.path.exists(data_dir):
           os.makedirs(data_dir)
           print(f"✓ Created data directory: {os.path.abspath(data_dir)}")
      
       # Define CSV file names in the timestamped Data folder
       self.csv_files = {
           'detected_points': os.path.join(data_dir, "points_cloud.csv"),
           'range_doppler_heatmap': os.path.join(data_dir, "range_doppler.csv"),
           'side_info': os.path.join(data_dir, "noise_snr.csv")
       }


       # Store the data directory for later use
       self.data_dir = data_dir


       # Create CSV files with headers
       self._create_csv_headers()
      
   def _create_csv_headers(self):
       """Create CSV files with appropriate headers."""
       try:
           # Detected points CSV (points_cloud.csv)
           with open(self.csv_files['detected_points'], 'w', newline='') as f:
               writer = csv.writer(f)
               writer.writerow(['frame_number', 'point_index', 'x', 'y', 'z', 'doppler', 'range', 'aoa'])


           # Range-Doppler heatmap CSV
           with open(self.csv_files['range_doppler_heatmap'], 'w', newline='') as f:
               writer = csv.writer(f)
               writer.writerow(['timestamp_s', 'frame_number', 'range_bin', 'doppler_bin', 'signal_strength'])


           # Side info CSV (noise_snr.csv)
           with open(self.csv_files['side_info'], 'w', newline='') as f:
               writer = csv.writer(f)
               writer.writerow(['timestamp_s', 'frame_number', 'point_index', 'snr', 'noise'])


           logging.info(f"CSV files created in directory: {self.csv_directory}")
           print(f"✓ CSV files created successfully:")
           print(f"  - detected_points: points_cloud.csv")
           print(f"  - range_doppler_heatmap: range_doppler.csv")
           print(f"  - side_info: noise_snr.csv")
          
       except Exception as e:
           logging.error(f"Error creating CSV files: {e}")
           print(f"✗ Error creating CSV files: {e}")


   def _write_to_csv(self, filename, data):
       """Write data to CSV file."""
       try:
           with open(filename, 'a', newline='') as f:
               writer = csv.writer(f)
               if isinstance(data, list):
                   for row in data:
                       writer.writerow(row)
               else:
                   writer.writerow(data)
           # Print confirmation for first few writes
           if self.frame_count <= 5:
               print(f"✓ Data written to {os.path.basename(filename)} (Frame {self.frame_count})")
       except Exception as e:
           logging.error(f"Error writing to CSV file {filename}: {e}")
           print(f"✗ Error writing to CSV: {e}")


   def read_data(self, device):
       """Read data from the radar device and put complete frames into the queue."""
       buffer = b""  # Initialize an empty buffer to accumulate data
       magic_word_pattern = b"\x02\x01\x04\x03\x06\x05\x08\x07"
       expected_header_length = 40  # Length of the frame header
       max_buffer_size = 10 * 1024 * 1024  # Set a reasonable buffer size limit (10 MB)
       discard_threshold = 1024  # Log significant discards only
       debug_log_interval = 100  # Log buffer state every 100 iterations
       iteration_count = 0
       # Variables for timing
       start_time = time.time()
       total_frames = 0


       while not self.stop_event.is_set():
           try:
               if device.data_port_connected:
                   byte_count = device.data_port.in_waiting
                   if byte_count > 0:
                       raw_data = device.data_port.read(byte_count)
                       # print("Data received ... ")
                       buffer += raw_data  # Append new data to the buffer


                       # Prevent buffer overflow
                       if len(buffer) > max_buffer_size:
                           logging.warning("Buffer size exceeded limit. Resetting buffer to prevent memory overflow.")
                           buffer = b""


                       while True:
                           # Look for the magic word in the buffer
                           magic_word_index = buffer.find(magic_word_pattern)
                           if magic_word_index == -1:
                               break  # No valid frame header found yet


                           if magic_word_index > 0:
                               if magic_word_index > discard_threshold:
                                   logging.warning(f"Discarding {magic_word_index} bytes of unaligned data.")
                               buffer = buffer[magic_word_index:]


                           if len(buffer) < expected_header_length:
                               break  # Wait for more data


                           frame_header = buffer[:expected_header_length]
                           total_packet_length = int.from_bytes(frame_header[12:16], byteorder='little')


                           if len(buffer) < total_packet_length:
                               break  # Wait for more data


                           frame_data = buffer[:total_packet_length]
                           buffer = buffer[total_packet_length:]  # Remove processed frame


                           if not self.data_queue.full():
                               self.data_queue.put(frame_data)
                               self.received_frames += 1
                               total_frames += 1
                               # logging.info(f"Complete frame added to queue. Queue size: {self.data_queue.qsize()}")
                           else:
                               logging.warning("Queue is full. Frame not added.")


               radar_params = getattr(device, "radar_params", None)
               frame_periodicity = radar_params.get("Frame Periodicity [ms]", 100) / 1000 if radar_params else 0.1
               # self.epsilon = max(self.epsilon, 0.001)
               # time.sleep(frame_periodicity/10 + self.epsilon)
               if frame_periodicity != 0:
                   time.sleep(min(frame_periodicity / 10, 0.01))
                   # print("frame_periodicity = ", frame_periodicity)
               else:
                   time.sleep(0.5)


               # Periodic debug logging
               iteration_count += 1
               if iteration_count % debug_log_interval == 0:
                   elapsed_time = time.time() - start_time
                   avg_processing_time = elapsed_time / max(total_frames, 1)  # Avoid division by zero
                   logging.debug(f"Buffer size: {len(buffer)} bytes, Total frames: {total_frames}, "
                                 f"Avg processing time per frame: {avg_processing_time:.4f} s")
                   start_time = time.time()  # Reset start time for the next interval
                   total_frames = 0  # Reset frame count for the next interval


           except AttributeError as e:
               logging.error(f"Device attribute error: {e}")
           except ValueError as e:
               logging.error(f"Value error during data parsing: {e}")
           except IOError as e:
               logging.error(f"I/O error during data reading: {e}")
           except Exception as e:
               logging.error(f"Unexpected error: {e}")


   def process_data(self, window):
       """Process data from the queue."""
       while not self.stop_event.is_set():
           try:
               if not self.data_queue.empty():
                   raw_data = self.data_queue.get_nowait()  # Non-blocking retrieval
                   magic_word_pattern = b"\x02\x01\x04\x03\x06\x05\x08\x07"
                   magic_word_index = raw_data.find(magic_word_pattern)


                   if magic_word_index == -1:
                       continue  # No valid data found


                   data = raw_data[magic_word_index:]
                   frame_header = data[:40]
                   parsed_header = self.parser.parse_frame_header(frame_header)
                   num_tlvs = parsed_header.get("Num TLVs", 0) if parsed_header else 0
                   tlv_data = self.parse_tlvs(data[40:], window, num_tlvs=num_tlvs)
                   self.parsed_frames += 1

                   if self.parsed_frames % 20 == 0:
                       logging.info(
                           f"Radar parsing active: received_frames={self.received_frames}, "
                           f"parsed_frames={self.parsed_frames}, queue={self.data_queue.qsize()}"
                       )


                   # Handle parsed data with the window passed as a parameter
                   self.handle_data(parsed_header, tlv_data, window)


                   # Check queue size after processing
                   if self.data_queue.qsize() > 1:
                       logging.info(f"Queue size after processing: {self.data_queue.qsize()}")


           except queue.Empty:
               pass  # No data in the queue, continue to next cycle
           except Exception as e:
               logging.error(f"Error processing data: {e}")


           # Short sleep to avoid excessive CPU usage, adjust as necessary
           time.sleep(0.01)  # Keep this small for real-time processing


   def parse_tlvs(self, tlv_data, window, num_tlvs=None):
       """Parse TLV data from the payload."""
       target_tlvs = num_tlvs if isinstance(num_tlvs, int) and num_tlvs > 0 else None

       def parse_with_mode(length_includes_header):
           tlv_list_local = []
           tlv_index_local = 0
           parsed_count_local = 0

           while tlv_index_local + 8 <= len(tlv_data):
               try:
                   tlv_type = int.from_bytes(tlv_data[tlv_index_local:tlv_index_local + 4], byteorder='little')
                   tlv_length = int.from_bytes(tlv_data[tlv_index_local + 4:tlv_index_local + 8], byteorder='little')

                   payload_length = tlv_length - 8 if length_includes_header else tlv_length
                   if payload_length < 0:
                       break

                   end_index = tlv_index_local + 8 + payload_length
                   if end_index > len(tlv_data):
                       break

                   tlv_payload = tlv_data[tlv_index_local + 8:end_index]
                   try:
                       parsed_tlv = self.parser.parse_tlv(tlv_type, tlv_payload, window)
                   except Exception as parse_exc:
                       logging.warning(f"Skipping TLV type {tlv_type} due to parse error: {parse_exc}")
                       tlv_index_local = end_index
                       parsed_count_local += 1
                       if target_tlvs is not None and parsed_count_local >= target_tlvs:
                           break
                       continue

                   tlv_list_local.append({
                       'type': tlv_type,
                       'data': parsed_tlv,
                       'length': tlv_length,
                       'payload': tlv_payload
                   })

                   tlv_index_local = end_index
                   parsed_count_local += 1

                   if target_tlvs is not None and parsed_count_local >= target_tlvs:
                       break
               except Exception as e:
                   logging.error(f"Error parsing TLVs at index {tlv_index_local}: {e}")
                   break

           return tlv_list_local, parsed_count_local, tlv_index_local

       parsed_header_mode, count_header_mode, consumed_header_mode = parse_with_mode(True)
       parsed_payload_mode, count_payload_mode, consumed_payload_mode = parse_with_mode(False)

       if target_tlvs is not None:
           if count_header_mode == target_tlvs and count_payload_mode != target_tlvs:
               return parsed_header_mode
           if count_payload_mode == target_tlvs and count_header_mode != target_tlvs:
               logging.info("TLV parser selected payload-only length mode.")
               return parsed_payload_mode

       header_score = (count_header_mode, consumed_header_mode)
       payload_score = (count_payload_mode, consumed_payload_mode)
       if payload_score > header_score:
           if count_payload_mode > 0:
               logging.info("TLV parser selected payload-only length mode.")
           return parsed_payload_mode

       return parsed_header_mode


   def handle_data(self, parsed_header, tlv_data, window):
       """Handle parsed radar data."""
       try:
           # Initialize stopwatch on first frame when CSV logging is enabled
           if self.csv_directory and self.start_time_s is None:
               self.start_time_s = time.time()


           # Calculate elapsed time in seconds since start
           if self.start_time_s is not None:
               elapsed_s = round(time.time() - self.start_time_s, 6)  # 6 decimal places (microsecond precision)
           else:
               elapsed_s = 0.0
           self.frame_count += 1
          
           if parsed_header:
               window.update_specified_table('header_table', parsed_header)


           for tlv in tlv_data:
               tlv_info = tlv['data']
               tlv_type = tlv['type']


               if tlv_type == 6:  # Statistics
                   window.update_specified_table('statistics_table', tlv_info)
                      
               elif tlv_type == 9:  # Temperature
                   window.update_specified_table('temperature_table', tlv_info)
                      
               elif tlv_type == 2:  # Range Profile
                   self.range_profile = tlv_info
                          
               elif tlv_type == 3:  # Noise Profile
                   self.noise_profile = tlv_info
                          
               elif tlv_type == 4:  # Azimuth Static Heatmap
                   self.azimuth_static_heatmap = tlv_info
               elif tlv_type == 5:  # Range-Doppler Heatmap
                   self.range_doppler_heatmap = tlv_info
                   if self.csv_directory and 'range_doppler_heatmap' in self.csv_files and tlv_info:
                       heatmap_data = []
                       for range_bin in range(len(tlv_info)):
                           for doppler_bin in range(len(tlv_info[range_bin])):
                               heatmap_data.append([
                                   elapsed_s,
                                   self.frame_count,
                                   range_bin,
                                   doppler_bin,
                                   tlv_info[range_bin][doppler_bin]
                               ])
                       if heatmap_data:
                           self._write_to_csv(self.csv_files['range_doppler_heatmap'], heatmap_data)
               elif tlv_type == 1:  # Detected Points
                   self.detected_points = tlv_info
                   window.update_specified_table('detected_points_table', tlv_info)
                   if self.csv_directory and 'detected_points' in self.csv_files and tlv_info:
                       points_data = []
                       for i, point in enumerate(tlv_info):
                           x = point.get('X', 0)
                           y = point.get('Y', 0)
                           z = point.get('Z', 0)
                           # Calculate range as sqrt(x^2 + y^2 + z^2)
                           range_val = np.sqrt(x*x + y*y + z*z)
                           # Calculate Angle of Arrival (AoA) as arctan(y/x) in degrees
                           aoa = np.degrees(np.arctan2(y, x))  # arctan2 handles all quadrants correctly
                           points_data.append([
                               self.frame_count,
                               i,
                               x,
                               y,
                               z,
                               point.get('Doppler', 0),
                               range_val,
                               aoa
                           ])
                       if points_data:
                           self._write_to_csv(self.csv_files['detected_points'], points_data)
                          
               elif tlv_type == 7:  # Side Info
                   self.side_info_for_detected_points = tlv_info
                   window.update_specified_table('points_info_table', tlv_info)
                   if self.csv_directory and 'side_info' in self.csv_files and tlv_info:
                       side_data = []
                       for i, info in enumerate(tlv_info):
                           side_data.append([
                               elapsed_s,
                               self.frame_count,
                               i,
                               info.get('snr', 0),
                               info.get('noise', 0)
                           ])
                       if side_data:
                           self._write_to_csv(self.csv_files['side_info'], side_data)
                          
       except Exception as e:
           logging.error(f"Error handling data: {e}")


   def update_plots_wrapper(self, window):
       """Wrapper function to update plots with new data."""
       radar_params = getattr(window, "radar_params", None)
       # unambiguous_range_m = radar_params.get("Unambiguous Range [m]", 10)
       maximum_range = radar_params.get("Maximum Range [m]", 10)
       unambiguous_velocity_km_h = radar_params.get("Unambiguous Velocity [km/h]", 10)
       unambiguous_velocity_m_s = unambiguous_velocity_km_h / 3.6
       min_value_range_doppler = radar_params.get("Range-Doppler Heatmap Minimum Value", 0)
       max_value_range_doppler = radar_params.get("Range-Doppler Heatmap Maximum Value", 4096)
       num_range_bins = radar_params.get("Number of Range FFT Bins", 256)
       # num_doppler_bins = radar_params.get("Number of Doppler FFT Bins", 16)
       min_value_range_azimuth = radar_params.get("Azimuth Static Heatmap Minimum Value", 0)
       max_value_range_azimuth = radar_params.get("Azimuth Static Heatmap Maximum Value", 2000)
       range_doppler_heatmap_grid_size = int(
           radar_params.get("Range-Doppler Heatmap Grid Size", 250))  # Grid resolution for image
       azimuth_static_heatmap_grid_size = int(radar_params.get("Azimuth Static Heatmap Grid Size", 250))  # Grid resolution for image
       azimuth_limits = 90
       if self.range_profile:
           if radar_params is None:
               x1 = 0
           else:
               num_samples_per_chirp = radar_params.get("Number of Samples per Chirp")
               range_resolution = radar_params.get("Range Resolution [m]")


               if num_samples_per_chirp is not None and range_resolution is not None:
                   x1 = num_samples_per_chirp * range_resolution
               else:
                   x1 = 0
           range_profile = 10 * (np.array(self.range_profile, dtype=np.complex64) / 512.0)
           noise_profile = 10 * (np.array(self.noise_profile, dtype=np.complex64) / 512.0)


           # logging.info(f"Raw Range Profile Shape: {range_profile.shape}")
           # logging.info(f"Raw Noise Profile Shape: {noise_profile.shape}")


           # Update the radar plots
           self.radar_visualizer.update_range_profile_plot(self.range_profile_plot_objects, x1, range_profile, x1,
                                                           noise_profile)
       if self.range_doppler_heatmap:
           #
           range_doppler_heatmap = np.array(self.range_doppler_heatmap)  # Shape: (256, 16)
           range_doppler_heatmap = np.fft.fftshift(range_doppler_heatmap, axes=1)
           desired_rows = max(range_doppler_heatmap_grid_size, range_doppler_heatmap.shape[0])
           desired_cols = max(range_doppler_heatmap_grid_size, range_doppler_heatmap.shape[1])


           # Rescale factors for each dimension
           scale_y = desired_rows / range_doppler_heatmap.shape[0]
           scale_x = desired_cols / range_doppler_heatmap.shape[1]
           interpolated_heatmap = zoom(range_doppler_heatmap, (scale_y, scale_x), order=1)  # Bilinear interpolation


           primary_axis = np.linspace(0, maximum_range, desired_rows)
           secondary_axis = np.linspace(-unambiguous_velocity_m_s, unambiguous_velocity_m_s, desired_cols)


           self.radar_visualizer.update_heatmap_plot(self.range_doppler_heatmap_item, interpolated_heatmap,
                                                     primary_axis, secondary_axis, False,
                                                     min_value_range_doppler, max_value_range_doppler,
                                                     colormap='jet')


       if len(self.azimuth_static_heatmap) > 0:
           logging.debug(f"Azimuth static heatmap dimensions: {np.shape(self.azimuth_static_heatmap)}")


           heatmap_data = np.abs(self.azimuth_static_heatmap).T
           num_azimuth_bins = heatmap_data.shape[0]


           # Index creation
           range_index = np.linspace(0, maximum_range, num_range_bins)
           azimuth_index = np.linspace(-azimuth_limits / 2, azimuth_limits / 2, num_azimuth_bins)


           # Convert azimuth to radians for polar plot
           azimuth_rad = np.deg2rad(azimuth_index)


           # Create a grid of azimuth and range
           azimuth_grid, range_grid = np.meshgrid(azimuth_rad, range_index, indexing='ij')


           # Convert polar coordinates (range, azimuth) to Cartesian (x, y)
           xx = range_grid * np.cos(azimuth_grid)  # x = r * cos(θ)
           yy = range_grid * np.sin(azimuth_grid)  # y = r * sin(θ)


           # Create a grid for the image (rescale to grid_size)
           x_grid = np.linspace(np.min(xx), np.max(xx), azimuth_static_heatmap_grid_size)
           y_grid = np.linspace(np.min(yy), np.max(yy), azimuth_static_heatmap_grid_size)


           # Flatten the original data for interpolation
           points = np.array([xx.flatten(), yy.flatten()]).T
           values = heatmap_data.flatten()


           # Create a 2D grid for the interpolation
           x_grid_2d, y_grid_2d = np.meshgrid(x_grid, y_grid)


           # Interpolate the matrix onto the new grid
           matrix_rescaled = griddata(points, values, (x_grid_2d, y_grid_2d), method='cubic')


           primary_axis = np.linspace(-maximum_range/2, maximum_range/2, num_range_bins)
           secondary_axis = np.linspace(0, maximum_range, num_range_bins)


           self.radar_visualizer.update_heatmap_plot(
               self.range_azimuth_heatmap_item, matrix_rescaled,
               primary_axis, secondary_axis, False,
               min_value_range_azimuth, max_value_range_azimuth,
               colormap='jet'
           )
       if len(self.detected_points) > 0:
           pos = np.array([[point['X'], point['Y'], point['Z']] for point in self.detected_points])
           doppler = np.array([[point['Doppler']] for point in self.detected_points])
           color = np.ones((len(pos), 4))  # Initialize with white color
           color[:, 0] = 1.0  # Red channel
           color[:, 1] = 0.0  # Green channel
           color[:, 2] = 0.0  # Blue channel
           color[:, 3] = 1.0  # Full opacity
           size = np.ones(len(pos)) * 15  # Point size
           window.three_d_plot_item.setData(pos=pos, color=color, size=size)
           # scatter_3d.setData(pos=pos, color=color, size=size)


           # print("pos = ", pos)
           # print("doppler = ", doppler)
       if len(self.side_info_for_detected_points) > 0:
           snr = np.array([[point['snr']] for point in self.side_info_for_detected_points])
           noise = np.array([[point['noise']] for point in self.side_info_for_detected_points])
           # print("snr = ", self.snr)
           # print("noise = ", self.noise)




