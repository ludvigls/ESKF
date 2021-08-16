# ESKF
In  this  project  the  Error  state  Kalman Filter  (ESKF)  was  implemented for a fixed wing UAV with IMU and GNSS sensors. It was tuned both for a simulated and a real dataset. The figure below shows the ground truth path in blue and the estimated path from the ESKF in red.

![3D](https://user-images.githubusercontent.com/36857118/129189182-f4a7f928-4c1c-48c1-86bc-2dfdea1923dc.png)
## How to run


The ESKF with a simulated dataset.
```
python3 run_simulated_INS.py
```
The ESKF with a real life dataset.
```
python3 run_real_INS.py
```
## Report
Check out report.pdf for more details.
