# # change filename to different csv files
# # run "python3 plot_tensor_vs_lifetime.py" in terminal

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Read data from CSV
# filename = "tensor_size_vs_lifetime_12_3_21_14_29.csv"
# data = pd.read_csv(filename)

# # Create scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(data['Lifetime'], data['Size_MB'], c='blue', alpha=0.7, edgecolors='w', s=80)
# plt.title('Scatter Plot of Tensor Size vs Lifetime', fontsize=14)
# plt.xlabel('Lifetime', fontsize=12)
# plt.ylabel('Size (MB)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)

# # Generate output file name based on input file name
# output_file = os.path.splitext(filename)[0] + ".png"

# # Save the plot as a PNG file
# plt.savefig(output_file, dpi=300, bbox_inches='tight')
# print(f"Plot saved as {output_file}")

# ____________________________________________________________________________________________________________________________________
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Get the CSV filename from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python3 plot_tensor_vs_lifetime.py <csv_filename>")
    sys.exit(1)

filename = sys.argv[1]

# Read data from CSV
data = pd.read_csv(filename)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Lifetime'], data['Size_MB'], c='blue', alpha=0.7, edgecolors='w', s=80)
plt.title('Scatter Plot of Tensor Size vs Lifetime', fontsize=14)
plt.xlabel('Lifetime', fontsize=12)
plt.ylabel('Size (MB)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Generate output file name based on input file name
output_file = os.path.splitext(filename)[0] + ".png"

# Save the plot as a PNG file
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved as {output_file}")
