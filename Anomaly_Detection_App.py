import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import cv2
import numpy as np
import csv
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from filter import filter_vehicles_by_presence
from dxdy_inv import calculate_dx_dy_and_inverses
import threading
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os 


class AnomalyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Highway Anomaly Detection System")
        
        # Make the window fullscreen
        #self.root.attributes('-fullscreen', True)
        self.root.geometry("1000x600")  # Resize window for better layout
        self.root.resizable(True, True)  # Allow window resizing

        # Set up style
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12), padding=6)
        style.configure('TLabel', font=('Arial', 12), padding=6)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), padding=10)

        self.video_path = None

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Configure grid to allow resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)  # Right frame gets more space

        # Create a frame for the left side (buttons, status)
        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create a frame for the right side (video output and plot)
        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure right frame subframes for video and plot
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)  # Set equal weight for video and plot sections
        right_frame.grid_columnconfigure(0, weight=1)

        # Subframes for right section: video output on top, plot on bottom
        right_frame_top = ttk.Frame(right_frame)
        right_frame_top.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        right_frame_bottom = ttk.Frame(right_frame)
        right_frame_bottom.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Header Label
        header_label = ttk.Label(left_frame, text="Highway Anomaly Detection System", style='Header.TLabel')
        header_label.pack(pady=10)

        # Video selection
        self.label = ttk.Label(left_frame, text="Select a video to analyze:")
        self.label.pack(pady=10)

        # Video Select button with Green background and White text
        self.select_button = tk.Button(left_frame, text="Select Video", command=self.select_video, bg="green", fg="white", font=('Arial', 16))
        self.select_button.pack(pady=5, padx=10)

        # Video Preview button with Yellow background and Black text
        self.preview_button = tk.Button(left_frame, text="Preview Video", command=self.preview_video, state=tk.DISABLED, bg="yellow", fg="black", font=('Arial', 16))
        self.preview_button.pack(pady=5, padx=10)

        # Analyze button with Red background and White text
        self.run_button = tk.Button(left_frame, text="Analyze", command=self.run_analysis, state=tk.DISABLED, bg="red", fg="white", font=('Arial', 16))
        self.run_button.pack(pady=10, padx=10)


        # # Progress bar
        # self.progress = ttk.Progressbar(left_frame, orient="horizontal", length=400, mode="determinate")
        # self.progress.pack(pady=10)

        # Anomaly status button (changes dynamically)
        self.status_button = tk.Button(left_frame, text="Waiting for Analysis", bg="gray", fg="white", font=("Arial", 16), padx=20, pady=10)
        self.status_button.pack(pady=10)

        # # Video output area (Video occupies 1/4th of the screen)
        self.video_output_label = ttk.Label(right_frame_top, text="", font=("Arial", 16))
        self.video_output_label.pack(expand=True, padx=5, pady=5)

        # # # Output plot area (Takes remaining space on the right)
        # self.plot_output_label = ttk.Label(right_frame_bottom, text="", font=("Arial", 16))
        # self.plot_output_label.pack(expand=True, padx=5, pady=5)

        # Close button to exit the application
        self.close_button = ttk.Button(left_frame, text="Close", command=self.close_app)
        self.close_button.pack(pady=10)

        # Anomaly Detection Message Label
        self.anomaly_message_label = ttk.Label(left_frame, text="", font=("Arial", 24, "bold"), foreground="black")
        self.anomaly_message_label.pack(pady=10)
        
        # Anomaly IDs Display Label (for big, bold numbers)
        self.anomaly_ids_label = ttk.Label(left_frame, text="", font=("Arial", 30, "bold"), foreground="black")
        self.anomaly_ids_label.pack(pady=10)

        # Store references to the right frame sections for video and plot
        self.right_frame_top = right_frame_top
        self.right_frame_bottom = right_frame_bottom

    def select_video(self):
        # Open file dialog to select a video file
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4 *.mov")])
        if self.video_path:
            video_name = os.path.basename(self.video_path)  # Extract the base name
            self.label.config(text=f"Selected Video: {video_name}")
            self.run_button.config(state=tk.NORMAL)
            self.preview_button.config(state=tk.NORMAL)

    def preview_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video file selected.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open the video file.")
            return

        def show_frame():
            ret, frame = cap.read()
            if ret:
                # Resize the video to fit 1/4th of the screen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()

                target_width = 640
                target_height = 360

                # Resize frame
                frame = cv2.resize(frame, (target_width, target_height))

                # Convert the frame to RGB
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_output_label.imgtk = imgtk
                self.video_output_label.config(image=imgtk)
                self.video_output_label.after(10, show_frame)  # Call this function again after 10ms
            else:
                cap.release()

        show_frame()

    def run_analysis(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        # self.progress.start(10)  # Start the progress bar
        self.status_button.config(text="Processing video... Please wait.", font=("Arial", 12),  bg="yellow", fg="black")

        # Run the analysis in a separate thread to avoid freezing the UI
        thread = threading.Thread(target=self.process_video)
        thread.start()

    def process_video(self):
        try:
            # CSV file paths
            csv_file_x_y_write = "./csv/yolo9_cctv6_normal.csv"
            flitered_csv_file_x_y_write = "./csv/yolo9_filtered_cctv6_normal.csv"
            dxdy_csv_file = "./csv/yolo9_cctv6_normal_dxdy.csv"
            average_csv_file = "./csv/yolo9_cctv6_normal_dxdy_avg_with_inverse.csv"
            graph_title = "Anomaly Detection"

            # Parameters
            min_threshold = 50 # Sudden Stop 130, Accident, Breakdown - 50

            # Initialize YOLO model
            model = YOLO('yolov9c.pt')

            # Open the video file
            cap = cv2.VideoCapture(self.video_path)
            track_history = defaultdict(lambda: [])
            ids, plot_value_x, plot_value_y = [], [], []

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = model.track(frame, persist=True, classes=[2, 5, 7])
                if results is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.cpu().tolist() if results[0].boxes.id is not None else []
                    annotated_frame = results[0].plot()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))

                        ids.append(track_id)
                        plot_value_x.append(float(x))
                        plot_value_y.append(float(y))

                        if len(track) > 30:
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=3)

                    # Show video in right_frame_top
                    cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_output_label.imgtk = imgtk
                    self.video_output_label.config(image=imgtk)
                    self.video_output_label.update()

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            cap.release()
            cv2.destroyAllWindows()

            data = zip(ids, plot_value_x, plot_value_y)
            with open(csv_file_x_y_write, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Ids', 'X', 'Y'])
                writer.writerows(data)

            # Call the processing functions
            filter_vehicles_by_presence(csv_file_x_y_write, flitered_csv_file_x_y_write, min_presence_threshold=min_threshold)
            calculate_dx_dy_and_inverses(flitered_csv_file_x_y_write, dxdy_csv_file, average_csv_file)

            # Run anomaly detection and check results
            num_anomalies, anomaly_ids = self.detect_anomalies_with_output(average_csv_file, graph_title)
            self.plot_trajectories(csv_file_x_y_write, anomaly_ids)
            if num_anomalies > 0:
                # Show anomaly message in large, bold red text
                self.anomaly_message_label.config(text="Anomaly Detected!", foreground="red")
                
                # Show the anomaly IDs in very large, bold format
                anomaly_ids_str = ", ".join(map(str, anomaly_ids))
                self.anomaly_ids_label.config(text=f"IDs: {anomaly_ids_str}", foreground="red")
            else:
                # Show no anomaly message in large, bold green text
                self.anomaly_message_label.config(text="No Anomalies Detected!", foreground="green")
                
                # Clear the anomaly IDs label when there are no anomalies
                self.anomaly_ids_label.config(text="")


        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_button.config(text="Error during processing", bg="red", fg="white")
        finally:
            self.progress.stop()

    def detect_anomalies_with_output(self, average_csv_file, graph_title):
        data = pd.read_csv(average_csv_file)
        
        x_co = data['dx_inv'].values
        y_co = data['dy_inv'].values
        data_arr = np.column_stack((data['Ids'].values, x_co, y_co))

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_arr[:, 1:3])
        scaled_data = np.column_stack((data_arr[:, 0], scaled_data))

        # Initialize DBSCAN model
        dbscan = DBSCAN(eps=0.57, min_samples=3)

        # Fit the model and predict anomaly labels
        anomaly_labels = dbscan.fit_predict(scaled_data[:, 1:3])
        anomaly_labels_mapped = np.where(anomaly_labels == -1, 1, 0)

        # Print the number of anomalies detected
        num_anomalies = np.sum(anomaly_labels_mapped)
        c1 = scaled_data[anomaly_labels_mapped == 0]
        c2 = scaled_data[anomaly_labels_mapped == 1]

        #plt.figure(figsize=(10, 6))
        # Plot normal data
        plt.scatter(c1[:, 1], c1[:, 2], label="Normal Data")

        # Plot anomaly data
        plt.scatter(c2[:, 1], c2[:, 2], label="Anomaly Data", color='r')

        plt.xlabel("dx_inv")
        plt.ylabel("dy_inv")
        plt.title(graph_title)
        plt.legend(title="Anomaly Vehicle IDs", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        # Return the number of anomalies and the anomaly IDs
        anomaly_ids = scaled_data[anomaly_labels == -1][:, 0].astype(int).tolist()
        return num_anomalies, anomaly_ids
    
    def plot_trajectories(self, csv_file_x_y_write, anomaly_ids):
        # Load the CSV data into a DataFrame
        data = pd.read_csv(csv_file_x_y_write)

        # Rotate 90 degrees counterclockwise
        angle = np.radians(180)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Filter data for anomaly and normal points
        anomaly_plot = data[data['Ids'].isin(anomaly_ids)]
        normal_plot = data[~data['Ids'].isin(anomaly_ids)]

        # Apply rotation transformation to anomalous points
        x_rotated_anomaly = anomaly_plot['X'] * cos_angle - anomaly_plot['Y'] * sin_angle
        y_rotated_anomaly = anomaly_plot['X'] * sin_angle + anomaly_plot['Y'] * cos_angle

        # Apply rotation transformation to normal points
        x_rotated_normal = normal_plot['X'] * cos_angle - normal_plot['Y'] * sin_angle
        y_rotated_normal = normal_plot['X'] * sin_angle + normal_plot['Y'] * cos_angle

        # Plot the rotated points
        plt.scatter(x_rotated_anomaly, y_rotated_anomaly, s=10, color='red', label='Anomaly')
        plt.scatter(x_rotated_normal, y_rotated_normal, s=1, color='blue', label='Normal')

        # Plot formatting
        plt.gca().invert_xaxis()
        plt.xlabel("X-axis reference to CCTV")
        plt.ylabel("Y-axis reference to CCTV")
        plt.title("Anomalous and Normal Trajectories")
        plt.legend()
        plt.show()


    def close_app(self):
        # Close the application gracefully
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    root.mainloop()
