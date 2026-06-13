import numpy as np
import utilities as ut

def main():

    # Video file path. Uncomment for testing various videos.
    # path = './pictures/clip-0.mov'
    # path = './pictures/clip-1.mov'
    path = './pictures/clip-2.mov'
    # path = './pictures/clip-3.mov'
    # path = './pictures/clip-4.mov'

    frames = ut.return_frames(path)

    # Calculating red color intensity of each frame.
    red_intensity = ut.calculate_intensity(frames)

    # Calculating fourier transform
    _, _, bpm = ut.calculate_fft(red_intensity, True)

    print(f"Your BPM is : {bpm:.2f}")
 

if __name__ == "__main__":
    main()