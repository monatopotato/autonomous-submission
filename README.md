# Ego-Trajectory & BEV Mapping (Traffic Light Reference)

## Approach (Part A)
We estimate ego trajectory using the traffic light as a fixed world reference.

1. For each frame, read the traffic light bounding box and compute its pixel center (u,v).
2. Load the corresponding XYZ point cloud (H,W,3) in camera coordinates and sample the 3D point at (u,v).
   - For robustness, we median-filter a small patch around (u,v) to reduce depth speckle and outliers.
3. Let p_cam(t) = [X,Y,Z] be the traffic light position in camera coordinates. We use its ground projection g(t)=[X,Y].
4. Define the world frame:
   - World origin is under the traffic light on the ground.
   - +Z is up.
   - At t=0, the light-to-car ground direction is aligned with +X.
5. Approximate planar ego position as:
   car_world_xy(t) = R0 * ( -g(t) ),
   where R0 is the 2D rotation that maps the initial light-to-car direction onto world +X.

This produces a BEV trajectory plot and an animated video over time.

## Outputs
- outputs/trajectory.png
- outputs/trajectory.mp4

## Notes / Assumptions
- We assume the main motion is planar and use only [X,Y] (ground projection).
- We keep a fixed initial world alignment (rotation at t=0). Large yaw changes would introduce drift, but the trajectory remains reasonable for short clips.

