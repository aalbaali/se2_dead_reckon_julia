# In this repo
The repository includes an example of propagating poses through SE(2) process model and demonstrates how to compute a first-order uncertainty bound.

The script heavily follows [this MATLAB script](https://github.com/UMich-CURLY-teaching/UMich-ROB-530-public/blob/main/code-examples/MATLAB/matrix_groups/odometry_propagation_se2.m) by [UMich-CURLY-teaching](https://github.com/UMich-CURLY-teaching).

# Generated plots
## Umbrella plot without covariance
![scatterplot](images/umbrella_plot.png)

## Trajectory with covariance
![traj_with_cov](images/trajectory_with_cov.png)

## Long trajectory with confidence bounds
Note that for longer trajectories, the confidence bounds are no longer consistent.
This may be due to simplified Jacobian computations (e.g., the left/right Jacobians are *not* used).

![traj_with_inconsistent_bounds](images/long_traj_inconsistent_confidence_bounds.png)

## Confidence bounds with and without (right) Jacobians
As can be seen here, the effect is minimal for the presented example.
Note that the effects may be more substantion for different examples.

![conf_with_and_without_jac](images/conf_bound_with_and_without_jacs.png)