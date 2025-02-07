# Motion_Similairity
This project is implemented for the AR/VR design fabrication studio (CITE490K) course at POSTECH with [1MILLION](https://www.1milliondance.com).
#### Measure motion similarity between the two videos using [MediaPipe](https://chuoling.github.io/mediapipe/) pose detection AI and DTW(Dynamic Time Warping) algorithm for research on dance copyright.
By using the DTW algorithm, it is possible to accurately compare the similarity between two dances, even if their video lengths, dance speeds, or starting times differ.

<p align="center">
<img src="https://github.com/user-attachments/assets/60325726-1892-4be7-9f1e-2e3e23a966d7"  width="600" height="300">
</p>

After extracting the key 13 vector data through pose estimation, the DRW algorithm is applied using cosine similarity to compare the dance similarity as a percentage.

<p align="center">
<img src="https://github.com/user-attachments/assets/ae4acb9f-4a6e-461f-a5ae-2c82952f786c"  width="700" height="200">
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/cfbda1f3-05eb-44f4-a7ba-4dea7f552b38"  width="600" height="300">
</p>
