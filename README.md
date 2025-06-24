# RF-data-generation-using-Nvidia-Sionna
This repository simulates RF signal tracks with Doppler effect using the NVIDIA Sionna ray tracing library. The generated data includes CSI matrices, receiver tracks, and transmitter orientations for multiple scene configurations.
project/
│
├── generate_rf_tracks.py      # Main simulation script  
├── /home/xml_scene/           # Folder containing XML files for scenes  
│   ├── scene1/  
│   │    └── scene1.xml  
│   ├── scene2/   
│   │    └── scene2.xml  
│   └── ...  
├── /home/sionna_csi_dataset/  # Folder where generated HDF5 datasets will be saved  
│  
└── README.md                  # This file  
  
This simulation:  
1. Loads predefined scenes from XML files.    
2. Simulates UE tracks under different transmitter orientations.  
3. Computes Doppler effect along the track.  
4. Generates CSI data using OFDM processing.  
5. Saves all outputs in HDF5 files.  
    
Inputs:  
Scene Files: Located in /home/xml_scene/scene{i}/scene{i}.xml for i = 1 to 9.  
Transmitter Positions: Fixed at [0, 0, 20] meters (can be adjsted based on application).  
Transmitter Orientations: Randomly generated yaw, pitch, roll combinations.  
UE Tracks: Zigzag patterns of 40 points per track, rotated based on TX orientation(coordinates need to change everytime according to the scene coordinates).  
OFDM Parameters:  
Subcarriers: 33  
Subcarrier spacing: 30 kHz  
OFDM symbol duration: 2 ms  
Sampling frequency: 500 Hz  
Time steps: 500 (1 second of data)  
Notes:  
All the above OFDM parameter can be adjust accordingly and ofdm grid can be formed.  
The code currently processes scenes with indexes 1 to 9.  
Doppler effect is applied using the velocity of the receiver at each time step(can be adjsted based on application).  
  
 Outputs:  
The generated datasets are stored in HDF5 files at:  
/home/sionna_csi_dataset/generated_dataset_scene{i}.h5  
  
Each file contains:  
csidata: Shape (samples, 500, 64, 33) – Complex64 CSI matrices  
tx_orientation: Shape (samples, 3) – Yaw, pitch, roll of the TX  
rx_positions: Shape (samples, 3) – Position of the RX at each time step  

 Code Flow  
GPU Setup: Configures TensorFlow to use GPUs with memory growth.  
Scene Loading: Iterates over 9 predefined scenes.  
Dataset Initialization: Creates an HDF5 file for each scene with resizable datasets.  
Transmitter Orientation: Selects specific TX orientations for each simulation.  
Track Generation: Simulates zigzag UE motion, rotated according to TX orientation.  
Doppler Calculation: Applies Doppler effect based on UE velocity.  
CSI Computation: Uses sionna functions to convert channel paths to OFDM CSI.  
Data Saving: Saves CSI, TX orientation, and RX positions into the HDF5 file.  
Scene Clean-up: Removes TX and RX after each iteration.   
   
Example Dataset Structure  
generated_dataset_scene1.h5  
├── csidata           # Shape: (N, 500, 64, 33)  
├── tx_orientation    # Shape: (N, 3)  
└── rx_positions      # Shape: (N, 3)  
  
Dependencies  
Python 3.8+  (tested with 3.9.21)
TensorFlow (tested with 2.15.1)  
Sionna 0.19.2  
h5py  
scipy  
numpy  
matplotlib  
  
Setup Instructions  
1. Clone the repository  
Install dependencies: pip install -r requirements.txt  
2. Additional CUDA Notes:  
Make sure your NVIDIA driver is compatible with CUDA 12.8.  
TensorFlow 2.15.1 will automatically use GPU if the CUDA runtime and cuDNN are properly installed.  
When using GPU make sure your TensorFlow installation matches your CUDA and cuDNN setup.  
  
3. Prepare scene files:  
Ensure your XML scene files are stored under: /home/xml_scene/scene{i}/scene{i}.xml  
Set the correct save path inside the Python script  
  
4. Running the Simulation  
Simply execute:  
python generate_rf_tracks.py  
The script will process all available scenes and save the results to the specified HDF5 files.  




