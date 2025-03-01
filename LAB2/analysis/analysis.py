import os
import glob
import folium
import utm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Find data files in the current directory structure
data_dir = os.path.join(os.getcwd(), "data")
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
print(f"Found {len(txt_files)} text files in {data_dir}")
# breakpoint()
def main_loop():
    """
    Main loop for reading from .txt files in the Data directory
    """
    data = {}

    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        data[file_name] = []
        print(f"Processing {file_name}...")
        
        with open(file_path, 'rb') as file:
            for line in file:
                try:
                    line = line.decode('utf-8').strip()
                except Exception:
                    # Skip undecodable lines
                    continue

                # Parse the NMEA sentence 
                if line.startswith('$GNGGA'):
                    gps_data = parse_gngga(line)
                    if gps_data:
                        data[file_name].append(gps_data)
        
        print(f"  → Extracted {len(data[file_name])} GNGGA records")

    return data

def parse_gngga(sentence):
    # Parse GNGGA sentence
    fields = sentence.split(",")
    
    try:
        time_UTC = float(fields[1]) if fields[1] else None
        latitude = float(fields[2]) if fields[2] else None
        longitude = float(fields[4]) if fields[4] else None
        fix_quality = int(fields[6]) if fields[6] else None  # Added fix quality tracking
        HDOP = float(fields[8]) if fields[8] else None
        altitude = float(fields[9]) if fields[9] else None

        if not latitude or not longitude or not altitude:
            return None

        lat_deg = int(latitude / 100)
        lat_min = latitude - lat_deg * 100
        lat_decimal = lat_deg + lat_min / 60

        lon_deg = int(longitude / 100)
        lon_min = longitude - lon_deg * 100
        lon_decimal = -(lon_deg + lon_min / 60) if fields[5] == 'W' else (lon_deg + lon_min / 60)

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
            'letter': letter,
            'fix_quality': fix_quality  # Save the fix quality
        }
    except (ValueError, IndexError) as e:
        # Handle parsing errors silently
        return None

def point_to_segment_distance(p, a, b):
    """Compute the distance from point p to the line segment (a, b)"""
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # Clamp to segment
    closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)

# Main execution
print("Starting RTK GNSS data analysis...")
data = main_loop()

# Define dataset categories and truth values
stationary = ['occluded_stationary.txt', 'open_air_stationary.txt']
square = ['occluded_square.txt', 'open_air_square.txt']
truth = {
    'occluded_square.txt': [(42.33742, -71.08666), (42.33730, -71.08654), (42.33712, -71.08681), (42.33726, -71.08693)],
    'occluded_stationary.txt': (42.33737, -71.08715),
    'open_air_square.txt': [(42.33635, -71.08843), (42.33622, -71.08858), (42.33629, -71.08866), (42.33643, -71.08853)],
    'open_air_stationary.txt': (42.33636400510384, -71.08838515087764)
}

# Process and analyze stationary datasets
print("\nAnalyzing stationary datasets...")
error_results = {}

for key in stationary:
    if key not in data:
        print(f"Warning: {key} not found in processed data")
        continue
        
    print(f"\nAnalyzing {key}:")
    df = pd.DataFrame(data[key])
    true_lat, true_lon = truth[key]

    # Generate Folium Map
    m = folium.Map(location=[df['latitude'].iloc[0], df['longitude'].iloc[0]], zoom_start=18, min_zoom=0)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=False,
        max_zoom=21
    ).add_to(m)
    
    # Add true position marker
    folium.Marker(
        [true_lat, true_lon],
        popup="True Position",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)
    
    # Add data points
    for _, row in df.iterrows():
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=2,
            fill=True,
            fill_opacity=0.7,
            color='blue'
        ).add_to(m)
    
    m.save(f"gps_map_{key.replace('.txt', '')}.html")
    print(f"  → Map saved as gps_map_{key.replace('.txt', '')}.html")

    # Convert true lat/lon to UTM
    true_easting, true_northing, _, _ = utm.from_latlon(true_lat, true_lon)

    # Compute positional error (Euclidean distance in UTM)
    df['position_error'] = np.sqrt((df['utm_easting'] - true_easting)**2 + (df['utm_northing'] - true_northing)**2)

    # Compute statistical deviations
    mean_error = df['position_error'].mean()
    std_dev_error = df['position_error'].std()
    
    # Track fix quality statistics
    fix_quality_stats = df['fix_quality'].value_counts().to_dict() if 'fix_quality' in df.columns else {}

    error_results[key] = {
        'mean_position_error': mean_error,
        'std_dev_position_error': std_dev_error,
        'fix_quality': fix_quality_stats
    }
    
    print(f"  → Mean error: {mean_error:.2f}m, Std Dev: {std_dev_error:.2f}m")
    print(f"  → Fix quality distribution: {fix_quality_stats}")

    # Scatter plot of Northing vs Easting
    plt.figure(figsize=(8, 6))
    plt.scatter(df['utm_northing'], df['utm_easting'], alpha=0.6, s=3)
    plt.plot([true_northing], [true_easting], 'ro', markersize=8, label="True Position")
    plt.title(f"Northing vs Easting Data ({key})",fontsize=14, fontweight='bold')
    plt.xlabel('Northing (m)',fontsize=14)
    plt.ylabel('Easting (m)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{key.replace('.txt', '')}_scatter.png", dpi=300)
    plt.close()

    # 2D histogram for position distribution
    plt.figure(figsize=(8, 6))
    plt.hist2d(df['utm_northing'], df['utm_easting'], bins=20, cmap='Blues')
    plt.colorbar(label='Count')
    plt.plot([true_northing], [true_easting], 'r*', markersize=10)
    plt.title(f"2D Histogram of Position ({key})",fontsize=14, fontweight='bold')
    plt.xlabel('Northing (m)',fontsize=14)
    plt.ylabel('Easting (m)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{key.replace('.txt', '')}_2d_histogram.png", dpi=300)
    plt.close()

    # Error histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df['position_error'], bins=15, edgecolor='black')
    plt.title('Histogram of Error from Known to Measured Position',fontsize=14, fontweight='bold')
    plt.xlabel('Distance (m)',fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{key.replace('.txt', '')}_error_histogram.png", dpi=300)
    plt.close()

# Process and analyze square path datasets
print("\nAnalyzing square path datasets...")

for key in square:
    if key not in data:
        print(f"Warning: {key} not found in processed data")
        continue
        
    print(f"\nAnalyzing {key}:")
    df = pd.DataFrame(data[key])
    square_corners = truth[key]
    
    # Convert square corners to UTM
    square_utm = [utm.from_latlon(lat, lon)[:2] for lat, lon in square_corners]

    # Create Folium map
    m = folium.Map(location=[df['latitude'].iloc[0], df['longitude'].iloc[0]], zoom_start=18, min_zoom=0)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=False,
        max_zoom=21
    ).add_to(m)
    
    # Add true square path
    square_coords = [(lat, lon) for lat, lon in square_corners]
    square_coords.append(square_corners[0])  # Close the loop
    folium.PolyLine(
        square_coords,
        color="red",
        weight=4,
        opacity=0.8,
        popup="True Square Path"
    ).add_to(m)
    
    # Add GPS track
    folium.PolyLine(
        [(row['latitude'], row['longitude']) for _, row in df.iterrows()],
        color="blue",
        weight=2,
        opacity=0.6,
        popup="GPS Track"
    ).add_to(m)
    
    m.save(f"gps_map_{key.replace('.txt', '')}.html")
    print(f"  → Map saved as gps_map_{key.replace('.txt', '')}.html")

    # Compute standard deviation of measured UTM coordinates
    std_easting = df['utm_easting'].std()
    std_northing = df['utm_northing'].std()
    print(f"  → Standard deviations - Easting: {std_easting:.2f}m, Northing: {std_northing:.2f}m")

    # Compute deviation from square path (minimum distance to expected path)
    errors = []
    for _, row in df.iterrows():
        point = np.array([row['utm_easting'], row['utm_northing']])
        
        # Calculate distance to each segment of the square
        segment_errors = []
        for i in range(len(square_utm)):
            j = (i + 1) % len(square_utm)  # Next point, wrapping around
            segment_errors.append(point_to_segment_distance(
                point, 
                np.array(square_utm[i]), 
                np.array(square_utm[j])
            ))
        
        errors.append(min(segment_errors))

    # Track fix quality statistics
    fix_quality_stats = df['fix_quality'].value_counts().to_dict() if 'fix_quality' in df.columns else {}

    # Compute mean and std deviation of path deviation
    mean_dev = np.mean(errors)
    std_dev = np.std(errors)
    error_results[key] = {
        'mean_position_error': mean_dev,
        'std_dev_position_error': std_dev,
        'fix_quality': fix_quality_stats
    }
    
    print(f"  → Mean path deviation: {mean_dev:.2f}m, Std Dev: {std_dev:.2f}m")
    print(f"  → Fix quality distribution: {fix_quality_stats}")

    # Scatter plot of Northing vs Easting with true path
    plt.figure(figsize=(8, 6))
    plt.scatter(df['utm_northing'], df['utm_easting'], alpha=0.6, s=3, label="GPS Data")
    
    # Plot true square path
    square_northing = [coord[1] for coord in square_utm]
    square_easting = [coord[0] for coord in square_utm]
    square_northing.append(square_northing[0])  # Close the loop
    square_easting.append(square_easting[0])    # Close the loop
    plt.plot(square_northing, square_easting, 'r-', linewidth=2, label="True Square Path")
    
    plt.title(f"GPS Path vs Expected Square Path ({key})",fontsize=14, fontweight='bold')
    plt.xlabel("Northing (m)",fontsize=14)
    plt.ylabel("Easting (m)",fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{key.replace('.txt', '')}_path_comparison.png", dpi=300)
    plt.close()

    # # Histogram of deviation from path
    # plt.figure(figsize=(8, 6))
    # plt.hist(errors, bins=15, edgecolor='black')
    # plt.title(f'Histogram of Deviation from Expected Path ({key})')
    # plt.xlabel('Deviation Distance (m)')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"{key.replace('.txt', '')}_deviation_histogram.png", dpi=300)
    # plt.close()

# Print summary of all results
print("\nSummary of Results:")
print("\nStationary Datasets:")
for key in stationary:
    if key in error_results:
        print(f"\n{key}:")
        print(f"  Mean position error: {error_results[key]['mean_position_error']:.3f} meters")
        print(f"  Standard deviation: {error_results[key]['std_dev_position_error']:.3f} meters")
        print(f"  Fix quality stats: {error_results[key]['fix_quality']}")

print("\nSquare Path Datasets:")
for key in square:
    if key in error_results:
        print(f"\n{key}:")
        print(f"  Mean path deviation: {error_results[key]['mean_position_error']:.3f} meters")
        print(f"  Standard deviation: {error_results[key]['std_dev_position_error']:.3f} meters")
        print(f"  Fix quality stats: {error_results[key]['fix_quality']}")

print("\nAnalysis complete. Check the current directory for generated maps and plots.")