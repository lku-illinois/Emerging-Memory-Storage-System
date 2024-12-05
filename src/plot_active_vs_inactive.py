# # change filename to different csv files
# # run "python3 plot_active_vs_inactive.py" in terminal

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Read data from CSV
# filename = "active_time_vs_inactive_time22_37_43.csv"  # Change this to the actual filename
# data = pd.read_csv(filename)

# # Plot 1: Tensor ID vs Active Time
# plt.figure(figsize=(8, 6))
# plt.scatter(data['Tensor_id'], data['ActiveTime'], c='green', alpha=0.7, edgecolors='w', s=80)
# plt.title('Scatter Plot of Tensor ID vs Active Time', fontsize=14)
# plt.xlabel('Tensor ID', fontsize=12)
# plt.ylabel('Active Time', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)

# # Save the plot
# output_file_active = os.path.splitext(filename)[0] + "_active_time.png"
# plt.savefig(output_file_active, dpi=300, bbox_inches='tight')
# plt.close()  # Close the figure to avoid overlap

# print(f"Active Time plot saved as {output_file_active}")

# # Plot 2: Tensor ID vs Inactive Time
# plt.figure(figsize=(8, 6))
# plt.scatter(data['Tensor_id'], data['InactiveTime'], c='red', alpha=0.7, edgecolors='w', s=80)
# plt.title('Scatter Plot of Tensor ID vs Inactive Time', fontsize=14)
# plt.xlabel('Tensor ID', fontsize=12)
# plt.ylabel('Inactive Time', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)

# # Save the plot
# output_file_inactive = os.path.splitext(filename)[0] + "_inactive_time.png"
# plt.savefig(output_file_inactive, dpi=300, bbox_inches='tight')
# plt.close()  # Close the figure to avoid overlap

# print(f"Inactive Time plot saved as {output_file_inactive}")


# ____________________________________________________________________________________________________________________________
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Get the CSV filename from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python3 plot_active_vs_inactive.py <csv_filename>")
    sys.exit(1)

filename = sys.argv[1]

# Read data from CSV
data = pd.read_csv(filename)

# Plot 1: Tensor ID vs Active Time
plt.figure(figsize=(8, 6))
plt.scatter(data['Tensor_id'], data['ActiveTime'], c='green', alpha=0.7, edgecolors='w', s=80)
plt.title('Scatter Plot of Tensor ID vs Active Time', fontsize=14)
plt.xlabel('Tensor ID', fontsize=12)
plt.ylabel('Active Time', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot
output_file_active = os.path.splitext(filename)[0] + "_active_time.png"
plt.savefig(output_file_active, dpi=300, bbox_inches='tight')
plt.close()

print(f"Active Time plot saved as {output_file_active}")

# Plot 2: Tensor ID vs Inactive Time
plt.figure(figsize=(8, 6))
plt.scatter(data['Tensor_id'], data['InactiveTime'], c='red', alpha=0.7, edgecolors='w', s=80)
plt.title('Scatter Plot of Tensor ID vs Inactive Time', fontsize=14)
plt.xlabel('Tensor ID', fontsize=12)
plt.ylabel('Inactive Time', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot
output_file_inactive = os.path.splitext(filename)[0] + "_inactive_time.png"
plt.savefig(output_file_inactive, dpi=300, bbox_inches='tight')
plt.close()

print(f"Inactive Time plot saved as {output_file_inactive}")
