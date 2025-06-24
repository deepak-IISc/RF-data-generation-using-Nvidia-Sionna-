import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import h5py
import sionna
from sionna.rt import Transmitter, Receiver, PlanarArray, load_scene
import os
import numpy as np
from sionna.ofdm import ResourceGrid
from sionna.channel.utils import subcarrier_frequencies, cir_to_ofdm_channel


import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

yaw_angles = np.linspace(0, 360, 10, endpoint=False)
pitch_angles = np.linspace(0, 20, 10, endpoint=False)
roll_angles = np.linspace(0, 180, 20, endpoint=False)
tx_orientations = []
np.random.seed(42)
for yaw in yaw_angles:    
    pitch = np.random.choice(pitch_angles)
    roll = np.random.choice(roll_angles)
    tx_orientations.append([yaw, pitch, roll])
tx_orientations = np.array(tx_orientations)

def generate_track_in_local_frame(n_points=40, spacing=1, zigzag_amplitude=5, zigzag_frequency=2*np.pi/8):
    track = [[0, 0, 0]] 
    t=0.0
    def get_point(t):
        y = t
        x = zigzag_amplitude * np.sin(zigzag_frequency * t)
        z = 0
        return np.array([x, y, z])
    while len(track) < n_points:
        last_point = track[-1]
        t_next = t + 0.01  # small step
        while True:
            candidate = get_point(t_next)
            if np.linalg.norm(candidate - last_point) >= spacing:
                track.append(candidate)
                t = t_next
                break
            t_next += 0.01
    return np.array(track)

def generate_rotated_ue_track(tx_pos=[0,0,0],tx_ori=[0,0,0], n_points=40):
    orientation_deg=tx_ori
    local_track = generate_track_in_local_frame(n_points)
    rot = R.from_euler('Z', orientation_deg[0],degrees=True)
    rotated=rot.apply(local_track)
    world_track=rotated + np.array(tx_pos)

    velocity_vecs = []
    dt=0.01
    for i in range(1, len(world_track)):
        prev = world_track[i - 1]
        curr = world_track[i]
        vel = [dt*(round(curr[j] - prev[j], 5)) for j in range(3)]
        velocity_vecs.append(vel)

    # Duplicate first velocity
    velocity_vecs.insert(0, velocity_vecs[0])
    velocity_vecs=np.array(velocity_vecs)
    return world_track, orientation_deg, velocity_vecs



scene_path="/home/parmardee/deepak/Mtech_project/new_task_1/dynamic_scene_model2/xml_scene2"

for i in range(9, 10):
    all_arrays=[]
    save_path= f"/home/hdd1/dynamic_gen_data/datagen_test2/raw_data/dynamic_scene_{i}_data.h5"
    if not os.path.exists(save_path):
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('csidata', shape=(0,500, 64, 33), maxshape=(None,500, 64, 33), dtype='complex64')
            # f.create_dataset('tx_positions', shape=(0, 3), maxshape=(None, 3), dtype='float32')
            f.create_dataset('tx_orientation', shape=(0, 3), maxshape=(None, 3), dtype='float32')
            f.create_dataset('rx_positions', shape=(0, 3), maxshape=(None, 3), dtype='float32')

    #----------------scene load 
    folder_name = f"scene{i}"
    file_name=f"scene{i}"
    folder_path = os.path.join(scene_path, folder_name)
    xml_file = os.path.join(folder_path, f"{file_name}.xml")
    if os.path.isfile(xml_file):
        print(f"Importing {xml_file}")
        scene=load_scene(xml_file)
        print(f"Scene no: {i} loaded")
        # print("Number of objects:", len(scene.objects))

        scene.tx_array = PlanarArray(num_rows=8,
                                num_cols=8,
                                vertical_spacing=0.12,
                                horizontal_spacing=0.12,
                                pattern="dipole",
                                polarization="V")
        scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0,
                                horizontal_spacing=0,
                                pattern="dipole",
                                polarization="V")
        #---resource grid paramter
        ofdm_symbol_duration=10e-3/5
        sampling_frequency = 1 / ofdm_symbol_duration  # ≈ 1400 Hz as before
        num_time_steps = 500 
        frequency=5e9
        tx_positions = [
            [0, 0, 20]
            # [50, 50, 20],
            # [-50, 50, 20],
            # [50, -50, 20],
            # [-50, -50, 20]
        ]
        # tx_orientation = [
        #     [0, 0, 0],       # forward (x)
        #     [0, 90, 0],      # right (y)
        #     [0, 180, 0],     # back (-x)
        #     [0, 270, 0]    # downward tilt
        # ]
        with h5py.File(save_path, 'a') as f:
            csids = f['csidata']
            # txds=f['tx_positions']
            txods=f['tx_orientation']
            rxds = f['rx_positions']

            # for tx, tx_pos in enumerate(tx_positions):
            

            # for k, tx_or in enumerate(tx_orientations):
            for k in [3, 7, 8]:
                tx_or = tx_orientations[k]
                track, tx_ori, velocity_vecs = generate_rotated_ue_track(tx_pos=[0,0,0],tx_ori=tx_or, n_points=40)
                my_tx1 = Transmitter(name="my_tx1",
                        position=tx_positions[0],
                        orientation=tx_ori,
                        power_dbm=44)
                scene.add(my_tx1)
                scene.frequency=frequency

                for rx, rx_pos in enumerate(track):
                    my_rx1 = Receiver(name="my_rx1",
                                        position=rx_pos,
                                        orientation=(0,0,0)
                                        )
                    scene.add(my_rx1)
                    frequencies = subcarrier_frequencies(33, 30e3)
                    paths = scene.compute_paths(max_depth=3)
                    paths.apply_doppler(
                        sampling_frequency=500,    # samples per second
                        num_time_steps=500,        # 1 second worth of symbols
                        rx_velocities=velocity_vecs[rx]
                    )
                    a,tau=paths.cir()
                    with tf.device('/CPU:0'):
                        h = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
                    h_reshaped = np.squeeze(h,axis=None)
                    final_csi = np.transpose(h_reshaped, (1, 0, 2))
                    # print(final_csi.shape)

                    csids.resize((csids.shape[0] + 1, 500, 64, 33))
                    # txds.resize((txds.shape[0] + 1, 3))
                    txods.resize((txods.shape[0] + 1, 3))
                    rxds.resize((rxds.shape[0] + 1, 3))

                    csids[-1] = final_csi
                    # txds[-1]= tx_pos
                    txods[-1]= tx_or
                    rxds[-1] = track[rx]
                    scene.remove("my_rx1")
                scene.remove("my_tx1")
                print(f"tx orientation {k} is done")    
    else:
        print(f"File not found: {xml_file}")
    print(f"scene {i} is done")
print("Process complete")

