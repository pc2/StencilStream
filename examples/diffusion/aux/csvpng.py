import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv

# Function to process a single file and generate an image
def process_file(file_name, output_name):
    data = []
    # Read the CSV file into a 2D array
    try:
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append([float(x) for x in row if x.strip() != ''])  # Ensure empty values are ignored
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        return
    
    # Check if data is not empty
    if not data:
        print(f"No data found in {file_name}")
        return
    
    # Convert the data into a NumPy array
    data_array = np.array(data)

    # Check if data array is valid
    if data_array.size == 0:
        print(f"Empty data in file {file_name}")
        return
    
    # Calculate the minimum and maximum values
    max_value = np.max(data_array)
    min_value = np.min(data_array)
    
    # Normalize the data from 0 to 100 (Optional: based on `vmin` and `vmax`)
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    
    # Create a colormap that goes from blue to red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_red", ["blue", "cyan", "yellow", "red"]
    )
    
    # Plot the data as an image
    plt.imshow(data_array, cmap=cmap, norm=norm, origin='upper')
    plt.colorbar()  # Optional, to show the color scale
    
    # Add labels and title for clarity (Optional)
    plt.title("Heatmap")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # Save the plot as a PNG image
    try:
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_name}")
    except Exception as e:
        print(f"Error saving plot to {output_name}: {e}")
    finally:
        plt.close()  # Close the plot to free memory

# Example usage
input_file = "Output.csv"
output_file = "Output.png"
try:
    process_file(input_file, output_file)
    print(f"Processed {input_file} -> {output_file}")
except FileNotFoundError:
    print(f"File not found: {input_file}")
