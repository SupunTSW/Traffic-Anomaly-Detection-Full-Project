import pandas as pd

def filter_vehicles_by_presence(csv_file_x_y_write, flitered_csv_file_x_y_write, vehicle_id_column='Ids', min_presence_threshold=100):
    """
    Filters vehicles based on their presence in the dataset.

    Args:
        csv_file_x_y_write (str): Path to the input CSV file containing the dataset.
        flitered_csv_file_x_y_write (str): Path to save the filtered dataset to a new CSV file.
        vehicle_id_column (str): The name of the column that contains vehicle IDs. Default is 'Ids'.
        min_presence_threshold (int): The minimum number of occurrences for a vehicle to be considered valid. Default is 100.

    Returns:
        None
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_x_y_write)

    # Group by vehicle ID column and count the number of occurrences
    vehicle_presence = df.groupby(vehicle_id_column).size()

    # Filter out vehicles with low presence
    valid_vehicles = vehicle_presence[vehicle_presence >= min_presence_threshold].index

    # Remove rows corresponding to invalid vehicles
    filtered_df = df[df[vehicle_id_column].isin(valid_vehicles)]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(flitered_csv_file_x_y_write, index=False)
