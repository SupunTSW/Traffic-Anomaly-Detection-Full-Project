import pandas as pd

def calculate_dx_dy_and_inverses(flitered_csv_file_x_y_write, dxdy_csv_file, average_csv_file, vehicle_id_column='Ids'):
    """
    Calculates dx, dy, and their inverses for each vehicle trajectory.

    Args:
        flitered_csv_file_x_y_write (str): Path to the input CSV file containing filtered trajectory data.
        dxdy_csv_file (str): Path to save the DataFrame with dx and dy values to a new CSV file.
        average_csv_file (str): Path to save the averages and reciprocal values to a new CSV file.
        vehicle_id_column (str): The name of the column that contains vehicle IDs. Default is 'Ids'.

    Returns:
        None
    """
    # Read the CSV file containing trajectory data
    data = pd.read_csv(flitered_csv_file_x_y_write)

    # Group the data by vehicle ID
    grouped_data = data.groupby(vehicle_id_column)

    # Create a new DataFrame to store the grouped data
    grouped_df = pd.DataFrame(columns=[vehicle_id_column, 'X', 'Y'])

    # Iterate over each group and concatenate the trajectory coordinates
    for vehicle_id, group in grouped_data:
        group_df = pd.DataFrame(group, columns=[vehicle_id_column, 'X', 'Y'])
        grouped_df = pd.concat([grouped_df, group_df])

    # Reset index
    grouped_df.reset_index(drop=True, inplace=True)

    # Initialize lists to store dx and dy values
    dx_values = []
    dy_values = []

    # Calculate dx and dy for each vehicle ID
    for _, group in grouped_df.groupby(vehicle_id_column):
        # Calculate dx
        dx = (group['X'].diff() ** 2).fillna(0)
        dx_values.extend(dx)

        # Calculate dy
        dy = (group['Y'].diff() ** 2).fillna(0)
        dy_values.extend(dy)

    # Add dx and dy columns to the DataFrame
    grouped_df['dx'] = dx_values
    grouped_df['dy'] = dy_values

    # Save the updated data as a new CSV file
    grouped_df.to_csv(dxdy_csv_file, index=False)

    # Calculate the average dx and dy for each vehicle ID
    average_dx_dy = grouped_df.groupby(vehicle_id_column).agg({'dx': 'mean', 'dy': 'mean'}).reset_index()

    # Calculate 1/average_dx and 1/average_dy
    average_dx_dy['dx_inv'] = 1 / average_dx_dy['dx']
    average_dx_dy['dy_inv'] = 1 / average_dx_dy['dy']

    # Save the averages and reciprocal values as a new CSV file
    average_dx_dy.to_csv(average_csv_file, index=False)
