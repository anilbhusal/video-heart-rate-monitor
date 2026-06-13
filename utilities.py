
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows


def return_frames(video_path):
        
    """
    Input:
        video_path: String which contains the directory path of video
    Return Parameters
        frames:     A list of frames. Each frame is of type numpy array of size (frame_height, frame_width, 3) 
    """

    capture = cv.VideoCapture(video_path, cv.CAP_FFMPEG)

    # Checking whether the file opened successfully or not.
    if not capture.isOpened():
        print('Unable to open video file.')
        return False
    
    # Variable for storing frames as list.
    frames = []

    try:
        while True:
            # Read frame by frame.
            status, frame = capture.read()
        
            # If 'status' is False, then we have reached the end of video.
            if not status:
                break

            # Converting BGR color format to RGB format for Matplotlib and Appending each frame to 'frames' list defined earlier..
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame)

            # Plotting current frame.
            # print(f"Frame number: {frames_count}")
            # clear_output(wait=True)
            # plt.imshow(frame_rgb)
            # plt.axis('off')
            # plt.show()

            # 0.2 second pause.
            # time.sleep(0.001)
        
    except KeyboardInterrupt:
        print("Process terminated by keyboard interruption")
        return frames

    return frames




def calculate_intensity(frames, graph=False):

    """
    Input:
        frames:             A list of frames. Each frame is of type numpy array of size (frame_height, frame_width, 3)  
        graph:              Boolen value for graphing. Plots if 'True' is passed.                
    Return Parameters
        frame_intent_values: Numpy array of red intensity value of each pixel.  
    """


    # Total number of frames.
    no_frames = len(frames)

    # Dimension of each RGB pixel
    n_row = frames[0].shape[0]
    n_col = frames[0].shape[1]

    # Variable for storing intensity of each pixel.
    frame_intent_values = np.zeros(no_frames)
    for ith_frame in range(no_frames):

        # Extracting Red, Green and Blue channel values of a particular location of each frame.
        frame = frames[ith_frame]
        average_red_channel = np.mean(frame[:, :, 0])

        # Calculating intensity using formula.
        # frame_intensity = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
        frame_intensity = average_red_channel
        frame_intent_values[ith_frame] = frame_intensity


    if graph:
        # Plotting intensity values against each frame.
        plt.figure(figsize=(16, 5))
        plt.plot(frame_intent_values, color='red', linewidth=0.8, marker='.')
        plt.title("Red Color Intensity of each frame.")
        plt.xlabel("Frame no.")
        plt.ylabel("Average Red Color Intensity")
        plt.savefig('intensityGraph.png')
        plt.show()

    return frame_intent_values



def calculate_fft(frame_intent_values, graph=False):
    """
    Video Acquisition Device Specifications
    Apple iPhone 11.
    FPS = 60 frames/sec @ 1080p HD
    """

    """
    Input:
        frames_intent_values:    Numpy array of red intensity value of each pixel.
        graph:              Boolen value for graphing. Plots if 'True' is passed. 
    Return Parameters:
        bpm_range:              Frequencies obtained from fft transform of intensity values.
        fft_range:              Magnitude of each frequencies.
        estimated_heart_rate:   Heart beat rate from strongest frequency.


    """

    Fs = 60.0
    Ts = 1 / Fs
    min_bpm, max_bpm = 40.0, 200.0
    m = len(frame_intent_values)

    # Zero out the steady flashlight baseline (DC offset)
    intensity_detrended = frame_intent_values - np.mean(frame_intent_values)

    # Aligning edges with a Hamming window to prevent frequency leakage
    windowed_signal = intensity_detrended * windows.hamming(m)

    # Compute Real FFT and translate Hz to Beats Per Minute (BPM)
    fft_magnitudes = np.abs(np.fft.rfft(windowed_signal))
    frequencies_bpm = np.fft.rfftfreq(m, d=Ts) * 60.0

    # Restricting frequency range to valid physiological heart rates
    valid_idx = np.where((frequencies_bpm >= min_bpm) & (frequencies_bpm <= max_bpm))[0]
    bpm_range = frequencies_bpm[valid_idx]
    fft_range = fft_magnitudes[valid_idx]

    # Extracting dominant pulse frequency
    max_peak_idx = np.argmax(fft_range)
    estimated_heart_rate = bpm_range[max_peak_idx]

    print(f"Calculated Heart Rate: {estimated_heart_rate:.1f} BPM")

    # Plotting the full frequency spectrum and identifying the peak heart rate frequency.
    if graph:
        plt.figure(figsize=(12, 4))
        plt.plot(bpm_range, fft_range, color='royalblue', linewidth=1.5, label='Entire Frequency Spectrum')
        plt.axvline(x=estimated_heart_rate, color='crimson', linestyle='--', label=f'Peak: {estimated_heart_rate:.1f} BPM')
        plt.title('Pulse Spectrum Analysis')
        plt.xlabel('Beats Per Minute (BPM)')
        plt.ylabel('Magnitude')
        plt.xlim(min_bpm, max_bpm)
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.savefig('spectrumAnalysis.png')
        plt.show()

    return bpm_range, fft_range, estimated_heart_rate