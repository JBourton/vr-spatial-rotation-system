# This file presents my full submission for the VR coursework.
# Both sequences play in this file, one after the other, and in both there are falling bunnies at random intervals and positions that may or may not collide with the central rotating bunny
# As requested, the first sequence is just the fused gyroscope and accelerometer data, and the second is that combined with the magnetometer data too

# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
# IMPORTANT NOTE: To get the second sequence to play after the first one finishes, simply press any key to close off the current window and starup the next.
# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}


# firstly, all my basic imports needed for running these sequeneces
import numpy as np
import cv2
import pandas as pd
import math
import random
import time as _time



# next, all my global hyperparams. Each and every tweakable constant i use lives here, in one place
HYPERPARAMS = {
    # first up is video recording. 
    # If set to true, both sequences are rendered with video files with no live display (note: this takes ages to do and isn't necessary for marking, as I've already uploaded that in my submission)
    # If set to false as it is here, both sequences are shown sequentially in real time, and no video is saved.
    "RECORD_VIDEO": False,

    # this is a video recording hyperparm: if it's lower, the video is smoother but it takes even longer to render
    # I calculate fps like this: stride=4 + fps_divisor=256 => 64 fps, ~27s duration to match up with the real IMU recording
    "VIDEO_RENDER_STRIDE": 4,
    "VIDEO_FPS_DIVISOR": 256, # because video_fps = max(1, VIDEO_FPS_DIVISOR // VIDEO_RENDER_STRIDE)

    # contrarilly, this is for the real-time display
    # I calcualte render_stride= max(1, total_frames // REALTIME_RENDER_DIVISOR)
    # ~80 gives about 80 rendered frames per sequence which is fine for striking the balance between responiveness and speed
    "REALTIME_RENDER_DIVISOR": 80,


    # these are just setting my 2 output filenames for the two sequence videos
    "FILENAME_OF_FIRST_SEQ": "first_seq_gyro_accel.mp4",
    "FILENAME_OF_SECOND_SEQ": "second_seq_full_fusion.mp4",

    # next up is my sensor fusion filter alphas
    # firstly, I have blend weight for accelerometer tilt correction in the complementary filter (low alpha=trusting gyro more; high alpha=trusting accelerometer more)
    "ACCEL_ALPHA": 0.02,
    # then the blend weight for magnetometer yaw correction in my full fusion filter
    # i've deliberately kept that separate from ACCEL_ALPHA because magnetic fields are inherently noisier, so playing around with different alpha values for both is useful for my experimentation
    "MAG_ALPHA": 0.02,


    # these are my ema low-pass pre-filter values, not to be confused with the coursework alpha values I tuned above
    "ACCEL_EMA_SMOOTH_FACTOR": 0.1,
    "MAG_EMA_SMOOTH_FACTOR": 0.1,

    # here I've explicitly defined all the scene/camera vars. Most of them are just from the standard engine.
    "WIDTH": 1200,
    "HEIGHT": 1200,
    "FOV_DEG": 60,
    "NEAR": 0.1,
    "FAR": 100.0,
    # I decided for my final submission that I wanted the camera pulled back and slightly elevated so a clear and fully visible head-on view of the central
    # bunny is present that has enough vertical room to watch falling bunnies enter from the top.
    # I decided to ditch the floor though, and any bunny collisions with it, as it doesn't seem to add any marks to my submission and would add more complexity.
    "EYE": [0, 1.5, 6.0],
    "CENTER": [0, 0, 0],
    "UP": [0, 1, 0],

    # now is all the falling bunny spawning params
    # first up here is the spawn interval range in real IMU seconds.
    # that's just determines when a new bunny appears after a random duration, which I chose to appear uniformly between [SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX]
    "SPAWN_INTERVAL_MIN": 3.0,
    "SPAWN_INTERVAL_MAX": 5.0,
    # then y height (in world space) where new teeny falling bunnies are born. I set at that as it's just above the top of screen
    "Y_HEIGHT_TO_SPAWN_BUNNY": 5.0,
    # to ensure some randomness in falling but still enough that collisions are still pretty likely, I set this to determine the maximum x/z offset from the world origin when spawning.
    "SPAWN_RADIUS": 1.4,

    # Now i determine when bunnies are considered off-screen, so they can be removed from the simulation once their y position drops below this threshold
    "OFFSCREEN_Y": -5.0,

    # this is a uniform scale factor applied to each falling bunny, which I set relative to the normalised mesh
    "FALLING_BUNNY_SCALE": 0.6,
    # and this is the tumble spin rate of falling bunnies around their y-axis (set in rad/s)
    "SPIN_SPEED": 0.8,

    # here I set all the physics (i.e. gravity & air resistance)
    # I just set all parameters according to that NASA falling-object model paper given in the coursework spec
    "GRAVITY": 9.8,
    "MASS": 1.0,
    "DRAG_COEFFICIENT": 0.5, 
    "AIR_DENSITY": 1.3,
    "AREA": 0.2,
    "RESTITUTION": 0.6, # coefficient of restitution, which is just fraction of impact speed that bounces back upward (0-1) in real terms

    # this is relating to the physics integration, where sim_dt * engine_substeps gives sim time per rendered frame (s)
    "SIM_TIMESHIFT": 0.015,
    "ENGINE_SUBSTEPS": 4,

    # this is the sphere collision radii
    "TUMBLING_TEENY_BUNNY_RADIUS":  0.44,
    "BIG_BOSS_BUNNY_RADIUS": 0.56,

    # this is the cap I set on horizontal velocity impulse that's imparted to a falling bunny after collision.
    # it prevents fast-impact bunnies from completely flying off the screen, which would be a touch nausiating at full speed (like in the video)
    "HORIZ_KICKOFF_CAP": 1.2,
}



# First up are all my q1 answers, my matrix transformation functions
# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

# this first one generates a 4x4 scaling matrix
def matrix_scaler(sc_x_coord, sc_y_coord, sc_z_coord):
    return np.array([
        [sc_x_coord, 0,0,0],
        [ 0,sc_y_coord, 0,0],
        [ 0, 0,sc_z_coord, 0],
        [ 0,0,0,1]
    ], dtype=float)

# & this one does the same but for translation matrix
def matrix_translater(tr_x_coord, tr_y_coord, tr_z_coord):
    return np.array([
        [ 1,0, 0, tr_x_coord],
        [ 0, 1, 0, tr_y_coord],
        [ 0, 0, 1, tr_z_coord],
        [ 0,0,0,1]
    ], dtype=float)

# and the final one(s), rotating around each of the 3 axises
# they use the right-hand coordinate system
# and all defined using radians as its just easier as I can work w/ pi
# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

# x-axis rotation
def matrix_rotater_x(angle_to_rotate_by):
    # I'm 1st calculting the sine & cosine vals separate, which I'm using in the formula as-is
    cos_angle = np.cos(angle_to_rotate_by)
    sin_angle = np.sin(angle_to_rotate_by)

    # & now the matrix for x-axis rotation is constructed]
    return np.array([
        [1,0,0,0],
        [0, cos_angle, -sin_angle, 0],
        [0, sin_angle,  cos_angle,0],
        [0,0,0,1]
    ], dtype=float)


# y axis, and I'm using same base function logic
def matrix_rotater_y(angle_to_rotate_by):
    cos_angle = np.cos(angle_to_rotate_by)
    sin_angle = np.sin(angle_to_rotate_by)

    # formulas for all of these from helpsheet in coursework 'materials' folder
    return np.array([
        [cos_angle, 0, sin_angle, 0],
        [0, 1, 0, 0],
        [-sin_angle, 0, cos_angle, 0],
        [0, 0, 0, 1]
    ], dtype=float)


# last but not least, z-axis
def matrix_rotater_z(angle_to_rotate_by):
    cos_angle = np.cos(angle_to_rotate_by)
    sin_angle = np.sin(angle_to_rotate_by)

    return np.array([
        [cos_angle, -sin_angle, 0,0],
        [sin_angle,cos_angle, 0,0],
        [0,0,1,0], 
        [0,0,0,1]
    ], dtype=float)


# {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
# below are all of my pt2/pt3 functions I implemented
# {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{

# the following function is a custom reader for loading in the IMUData.csv file
def readin_imu_dataset(imu_data_filename="IMUData.csv"): # I added this as the deafult filepath but it can be flexibly changed with new imu csv data here
    imu_dataset = pd.read_csv(imu_data_filename, skipinitialspace=True) # (the file seemed to have a leading whitespace, so this operation was added to ditch it)
    imu_dataset.columns = imu_dataset.columns.str.strip()

    # as required by cousrework q2.1.i, this takes the rotational rates from deg/s to rad/s
    gyroscope_cols_in_rads_to_convert = ['gyroscope.X', 'gyroscope.Y', 'gyroscope.Z']
    imu_dataset[gyroscope_cols_in_rads_to_convert] = np.deg2rad(imu_dataset[gyroscope_cols_in_rads_to_convert])

    return imu_dataset


imu_data = readin_imu_dataset()

# now for all the manipualtion functions

# i. here I'm calculating quaternion from euler radians
def convert_euler_to_quaternion(roll_x, pitch_y, yaw_z):
    # for this I'm using the standard eq.:
    # w = cosroll*cospitch*cosyaw + sinroll*sinpitch*sinyaw ; x = sinroll*cospitch*cosyaw - cosroll*sinpitch*sinyaw
    # y = cosroll*sinpitch*cosyaw + sinroll*cospitch*sinyaw ; z = cosroll*cospitch*sinyaw - sinroll*sinpitch*cosyaw
    # & that gives me the output quaternion [w,x,y,z]

    # half-angles in rads are 1st precalculted
    half_roll= roll_x *0.5
    half_pitch= pitch_y * 0.5
    half_yaw= yaw_z *0.5

    # now i'm calcluating trigonmetric vals here
    cos_roll= math.cos(half_roll)
    sin_roll= math.sin(half_roll)
    cos_pitch= math.cos(half_pitch)
    sin_pitch= math.sin(half_pitch)
    cos_yaw= math.cos(half_yaw)
    sin_yaw= math.sin(half_yaw)

    # now I construct each of the w,x,y,z components of the quaternion here
    w= cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    x= sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    y= cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    z= cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    return w,x,y,z



# ii. now the opposite, I'm calculating euler radians from quaternion
def convert_quaternion_to_euler(w, x, y, z):
    # (note: I verified my equation at: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler
    # my function output is euler angles (roll_x, pitch_y, yaw_z) in rads

    # firstly, I'm calculating the roll (aka x-axis rotation)
    sin_roll_cos_pitch= 2 * (w * x + y * z)
    cos_roll_cos_pitch= 1 - 2 * (x * x + y * y)
    roll_x = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    # secondly, I'm calculating pitch (aka y-axis rotation)
    sin_pitch= 2* (w*y - z*x)
    
    # here I handle floating-point inaccuracies (i.e. gimbal lock) by clamping my sin val to a valid [-1.0,1.0] range
    if abs(sin_pitch) >= 1:
        pitch_y= math.copysign(math.pi / 2, sin_pitch)
    else:
        pitch_y= math.asin(sin_pitch)

    # finally, I'm calculating yaw (aka z-axis rotation)
    sin_yaw_cos_pitch= 2* (w*z + x*y)
    cos_yaw_cos_pitch= 1 - 2* (y*y + z*z)
    yaw_z = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return roll_x,pitch_y, yaw_z


# iii. next, I convert a quaternion to its conjugate (i.e. i'm inversely rotating it)
def convert_quaternion_to_conjugate(w, x, y, z):
    # eq. is nice and simple for this, just negating x,y & z: q*= [w,-x,-y,-z] (fetched from https://mathworld.wolfram.com/QuaternionConjugate.html)
    return w,-x,-y,-z


# iv. finally, this one is calcting the quaternion product of a w/ b
def calculate_quaternion_product(w1, x1, y1, z1, w2, x2, y2, z2):
    # it's just standard hamilton product formula here that i got from: https://www.tutis.ca/Rotate/7quaternions.htm#:~:text=i2%20=%20j2%20=%20k,%7Cq%7C2%20(7.4)
    combined_w= w1*w2 - x1*x2- y1*y2 -z1*z2
    combined_x= w1*x2 + x1 * w2 + y1 * z2 - z1 *y2
    combined_y= w1* y2 - x1*z2 + y1*w2 +z1*x2
    combined_z= w1*z2 + x1*y2- y1* x2 + z1*w2
    return combined_w,combined_x,combined_y,combined_z


# this is my attempt at implementing dead reckoning for Q3.1
# I took the maths for this from lecture 4's slides
def dead_reckoning_orientation_calc_implementation(all_imu_data_df):
    # 1st I need to evaluate sequential orientation over time w/ this code
    # q[0] per coursework spec is [1,0,0,0] (i.e. identity state)
    w_id_ref,x_id_ref,y_id_ref,z_id_ref=1.0, 0.0, 0.0, 0.0

    # i'm using an array to store tracked orientations history
    orientation_history_container = [(w_id_ref, x_id_ref, y_id_ref, z_id_ref)]

    # now for the time difference calcs
    # i'm iterating through dataset, but I start at idx1 as the timestamp at idx0 is 0s
    for i in range(1, len(all_imu_data_df)):
        # 1st, I calcualte the elapsed time betwween current and previous sensor reading
        elapsed_sensor_time_change= all_imu_data_df['time'][i] -all_imu_data_df['time'][i-1]

        # 2nd, i calculate angular displacment for each axis
        change_over_roll= all_imu_data_df['gyroscope.X'][i] * elapsed_sensor_time_change # (i.e. rotational speed (rad/s) * time elapsed (s))
        change_over_pitch= all_imu_data_df['gyroscope.Y'][i] * elapsed_sensor_time_change
        change_over_yaw = all_imu_data_df['gyroscope.Z'][i] * elapsed_sensor_time_change

        # 3rd I'm converting small euler rotation--> quaternion using my function I was asked to implmenet before in Q2
        w_change, x_change, y_change, z_change = convert_euler_to_quaternion(change_over_roll, change_over_pitch, change_over_yaw)

        # and last but not least, accumulating rotation w/ hamilton product 
        # per lecture slides, eq. is: q_current = delta_q_current * q_previous
        w_id_ref, x_id_ref, y_id_ref,z_id_ref = calculate_quaternion_product(
            w_change, x_change,y_change, z_change,  # quaternion a (==the new movement)
            w_id_ref, x_id_ref, y_id_ref, z_id_ref) # & quaternion b (==the previous state)

        # now it's done, so I append the new orientation calc to the tracking array & move onto the next
        orientation_history_container.append((w_id_ref, x_id_ref, y_id_ref, z_id_ref))
    return orientation_history_container


# next up, I'm implementing the code to transform acceleration measurements into the global frame
# I used the LaValle et al. paper for the theory in this one
def transform_vector_to_global_frame(quat_w, quat_x, quat_y, quat_z, accel_x, accel_y, accel_z): # quat_[] represents the orientation quaternion, i.e. how the headset is currently rotated in space
    # I need to rotate a local 3D vector back into world space
    # To do that, I 1st convert the 3D accelerometer reading -> pure quaternion (w/ scalar w==0)
    w_pure_local, x_pure_local, y_pure_local,z_pure_local= 0.0, accel_x,accel_y,accel_z # (these are the pure quaternion representations of the 3D vectors of the headset's current tilt)

    # 2nd, the inverse of the current orientation is used to shift measurement to world frame 
    # note that in my naming convention, these variables below are the inverse/conjugate of the orientation; they're doing the untwisting of local sensor data back into the world format
    conj_w,conj_x,conj_y, conj_z= convert_quaternion_to_conjugate(quat_w, quat_x, quat_y, quat_z)

    # 3rdly I'm calcaulting the 1st hamilton product, conj_q * pure_local_p
    temp_w, temp_x, temp_y, temp_z= calculate_quaternion_product(
        quat_w, quat_x, quat_y,quat_z,
        w_pure_local,x_pure_local, y_pure_local, z_pure_local
    )

    # finally I'm calcultating the 2nd hamilton product, (conj_q * pure_local_p) * q
    # noting that the conjugate of the inverse is simply our original q
    global_w, global_x, global_y, global_z = calculate_quaternion_product(
        temp_w, temp_x, temp_y, temp_z,
        conj_w, conj_x, conj_y, conj_z
    )

    # and now, resulting global vector is isolated by getting rid of scalar w part (the global_w var above)
    return global_x, global_y, global_z # these are all now measured relative to the virtual world, w/ global up-axis pointing towards virtual cieling



# next, I calculate the tilt axis & correction angle between accel-derived up and true world up
def calculate_tilt_axis_and_angle(global_accel_x, global_accel_y, global_accel_z):
    # here I find the world 'up' reference vector. As gravity points down, accelerometer is going to be reading up when still
    # note: the IMU dataset uses z-up convention according to coursework spec
    x_ref_to_up, y_ref_to_up, z_ref_to_up = 0.0, 0.0, 1.0

    # 1st I normalise the accelerometer-derived up vector so magnitude doesn't affect the angle calc
    magnitude_of_acceleromater = math.sqrt(global_accel_x**2 + global_accel_y**2 + global_accel_z**2)
    if magnitude_of_acceleromater<1e-8: # here I'm guarding against zero-vector edge case
        return 1.0, 0.0, 0.0, 0.0 # return identity-ish (no correction needed)
    normalised_x_comp= global_accel_x/magnitude_of_acceleromater
    normalised_y_comp= global_accel_y/magnitude_of_acceleromater
    normalised_z_comp= global_accel_z/magnitude_of_acceleromater

    # 2nd, I find the cross product of accel_normalised * up_ref gives the tilt axis
    # (i.e. axis perpendicular to both, & around that the error rotation lives)
    tilt_axis_x= normalised_y_comp*z_ref_to_up - normalised_z_comp*y_ref_to_up
    tilt_axis_y= normalised_z_comp*x_ref_to_up - normalised_x_comp*z_ref_to_up
    tilt_axis_z= normalised_x_comp*y_ref_to_up -normalised_y_comp*x_ref_to_up

    # 3rd, I normalise the tilt axis (cross product length varies with angle)
    axis_magnitude = math.sqrt(tilt_axis_x**2 + tilt_axis_y**2 + tilt_axis_z**2)
    if axis_magnitude < 1e-8: # vectors are already aligned--> no correction needed
        return 1.0, 0.0, 0.0, 0.0
    tilt_axis_x/= axis_magnitude
    tilt_axis_y/= axis_magnitude
    tilt_axis_z/= axis_magnitude

    # 4th, dot product of the two normalised vectors gives cos(phi)--> therefore arccos gives phi itself
    dot_product= normalised_x_comp*x_ref_to_up + normalised_y_comp*y_ref_to_up + normalised_z_comp*z_ref_to_up
    dot_product= max(-1.0, min(1.0, dot_product)) # clamping again to dodge floating point arccos errors
    phi_angle= math.acos(dot_product) # this is the angular error between accel up & world up
    return tilt_axis_x, tilt_axis_y, tilt_axis_z, phi_angle


# now the complementary filter, fusing gyroscope dead reckoning w/ accelerometer tilt correction
# this balances short term accuracy (but w/ drift) of gyroscope w/ noisey short-term accelerometer traits that's more stable longer term (thanks to https://guidenav.com/blog/how-gyroscopes-and-accelerometers-shape-imu-performance for more context on this)
# the alpha here is responible for controlling the blend of the 2(low alpha==trust gyro more & high alpha==trust more in the acceleratormer)
def complementary_filter_orientation_calc(imu_dataframe, alpha=HYPERPARAMS["ACCEL_ALPHA"]):
    # this is my initial id quaternion again (w/ same starting point)
    w_id_ref, x_id_ref, y_id_ref, z_id_ref = 1.0, 0.0, 0.0, 0.0

    # I have this tracking array just like in my dead reckoning code too
    orientation_history_container = [(w_id_ref, x_id_ref, y_id_ref, z_id_ref)]

    # low-pass filter state for accelerometer (exponential moving average, smoothing factor 0.1)
    # initialised to first sample so there's no cold-start spike on frame 1
    lp_accel_x = imu_dataframe['accelerometer.X'][0]
    lp_accel_y = imu_dataframe['accelerometer.Y'][0]
    lp_accel_z = imu_dataframe['accelerometer.Z'][0]
    lp_smoothing_factor = 0.1

    for i in range(1, len(imu_dataframe)):
        # 1st, elapsed time between readings
        elapsed_sensor_time_change= imu_dataframe['time'][i]- imu_dataframe['time'][i-1]

        # 2nd, gyroscope angular displacement for each axis (same as dead reckoning)
        change_over_roll= imu_dataframe['gyroscope.X'][i]*elapsed_sensor_time_change
        change_over_pitch= imu_dataframe['gyroscope.Y'][i]*elapsed_sensor_time_change
        change_over_yaw= imu_dataframe['gyroscope.Z'][i]*elapsed_sensor_time_change

        # 3rdly I convert small euler rotation--> quaternion & accumulate via hamilton product
        w_change, x_change, y_change, z_change= convert_euler_to_quaternion(change_over_roll, change_over_pitch, change_over_yaw)
        gyro_w, gyro_x, gyro_y, gyro_z = calculate_quaternion_product(
            w_change, x_change, y_change, z_change,
            w_id_ref, x_id_ref,y_id_ref, z_id_ref)

        # 4th I normalise the gyro estimate (for preventing floating-point drift, same as before)
        normed_gyro_estimate= math.sqrt(gyro_w**2 + gyro_x**2 + gyro_y**2 + gyro_z**2)
        gyro_w, gyro_x, gyro_y, gyro_z= gyro_w/normed_gyro_estimate, gyro_x/normed_gyro_estimate, gyro_y/normed_gyro_estimate, gyro_z/normed_gyro_estimate

        # 5th I read the raw accelerometer values for this timestep
        raw_accel_x= imu_dataframe['accelerometer.X'][i]
        raw_accel_y= imu_dataframe['accelerometer.Y'][i]
        raw_accel_z= imu_dataframe['accelerometer.Z'][i]

        # 5a. low-pass filter the raw accel values (exponential moving average) to attenuate
        # high-frequency noise before using them for tilt correction
        lp_accel_x = lp_smoothing_factor * raw_accel_x + (1.0 - lp_smoothing_factor) * lp_accel_x
        lp_accel_y = lp_smoothing_factor * raw_accel_y + (1.0 - lp_smoothing_factor) * lp_accel_y
        lp_accel_z = lp_smoothing_factor * raw_accel_z + (1.0 - lp_smoothing_factor) * lp_accel_z

        # 5b. magnitude gate: only trust the accelerometer when it's reading close to 1g
        # (i.e. the headset isn't undergoing significant linear acceleration that would corrupt
        # the gravity estimate, as per LaValle et al. 2014 a = ag + al)
        accel_magnitude = math.sqrt(lp_accel_x**2 + lp_accel_y**2 + lp_accel_z**2)
        accel_is_trustworthy = 0.85 <= accel_magnitude <= 1.15

        # 6th I rotate accel reading into global frame using my current gyro orientation estimate
        global_accel_x, global_accel_y, global_accel_z = transform_vector_to_global_frame(
            gyro_w, gyro_x, gyro_y, gyro_z,
            lp_accel_x, lp_accel_y, lp_accel_z)

        # 7th I calculate the tilt axis & correction angle phi from global accel vector
        tilt_axis_x, tilt_axis_y, tilt_axis_z, phi_angle = calculate_tilt_axis_and_angle(
            global_accel_x,global_accel_y, global_accel_z)

        # 8th I build the correction quaternion from the tilt axis & a scaled-down version of phi, representing a small rotation (alpha*phi) around tilt axis
        # (using the axis-angle-->quaternion formula: w=cos(theta/2), xyz=axis*sin(theta/2))
        # correction is only applied if the magnitude gate passed (accel_is_trustworthy)
        if accel_is_trustworthy:
            correction_angle= alpha*phi_angle
            correction_w= math.cos(correction_angle/2.0)
            correction_x= tilt_axis_x*math.sin(correction_angle/2.0)
            correction_y= tilt_axis_y*math.sin(correction_angle/2.0)
            correction_z= tilt_axis_z*math.sin(correction_angle/2.0)

            # 9th, I'm applying the correction by multiplying it onto the gyro estimate
            # (pre-multiplying in world space: correction_q*gyro_q)
            w_id_ref, x_id_ref, y_id_ref, z_id_ref = calculate_quaternion_product(
                correction_w, correction_x, correction_y, correction_z,
                gyro_w, gyro_x, gyro_y, gyro_z)
        else:
            # if this part is triggered, my magnitude gate failed. In that case I've set it up to just use the gyro estimate unchanged this step
            w_id_ref, x_id_ref, y_id_ref, z_id_ref = gyro_w, gyro_x, gyro_y, gyro_z

        # 10th I'm normalising the fused result before storing
        fused_norm = math.sqrt(w_id_ref**2 + x_id_ref**2 + y_id_ref**2 + z_id_ref**2)
        w_id_ref, x_id_ref, y_id_ref, z_id_ref = w_id_ref/fused_norm, x_id_ref/fused_norm, y_id_ref/fused_norm, z_id_ref/fused_norm

        # & now I'm done for this step, so I store & move on
        orientation_history_container.append((w_id_ref, x_id_ref, y_id_ref, z_id_ref))
    return orientation_history_container


# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
# end of all those functions (q2-3)

# now for q4:

# here I'm taking the initial magnetometer reading as a reference for yaw correction, which I capture at frame 0 before rotation occurs
# (i.e. it represents magnetic north in the sensor's starting orientation)
def capture_magnetometer_reference(imu_dataframe):
    # just grabbing the very first magnetometer reading from the dataset
    # this is my source of truth magnetic north from the get-go of the animation
    ref_x, ref_y, ref_z= imu_dataframe['magnetometer.X'][0], imu_dataframe['magnetometer.Y'][0], imu_dataframe['magnetometer.Z'][0]

    # i'm normalising so magnitude differences don't affect angle calc later
    ref_magnitude= math.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
    ref_x /= ref_magnitude
    ref_y /= ref_magnitude
    ref_z /= ref_magnitude

    return ref_x, ref_y, ref_z


# here I'm calculating the yaw correction axis & angle by comparing current mag reading to the reference one
# theory: mirrors calculate_tilt_axis_and_angle but projected onto the horizontal plane only
# the point being is that I'm only correcting yaw & not disturbing the roll/pitch already fixed by accelerometer
def calculate_yaw_axis_and_angle(global_mag_x, global_mag_y, global_mag_z, ref_mag_x, ref_mag_y, ref_mag_z):
    # 1st, I project both vectors onto the horizontal plane by zeroing the vertical (Z) component
    # this isolates the yaw component of any difference between them
    # (Z is up in this IMU's convention per the coursework spec)
    curr_x, curr_y, curr_z= global_mag_x, global_mag_y, 0.0
    ref_x_h, ref_y_h, ref_z_h= ref_mag_x, ref_mag_y, 0.0

    # 2nd, normalise both projected vectors
    curr_magnitude= math.sqrt(curr_x**2 + curr_y**2 + curr_z**2)
    ref_magnitude= math.sqrt(ref_x_h**2 + ref_y_h**2 + ref_z_h**2)

    # now I'm guarding against zero vectors (e.g. if the mag reading is purely vertical)
    if curr_magnitude < 1e-8 or ref_magnitude < 1e-8:
        return 0.0, 0.0, 1.0, 0.0  # no correction, axis pointing up
    curr_x /= curr_magnitude
    curr_y /= curr_magnitude
    ref_x_h /= ref_magnitude
    ref_y_h /= ref_magnitude

    # 3rd I take the cross product to get the yaw correction axis (it'll point along Z since both vectors are horizontal)
    yaw_axis_x= curr_y * ref_z_h - curr_z * ref_y_h
    yaw_axis_y= curr_z * ref_x_h - curr_x * ref_z_h
    yaw_axis_z= curr_x * ref_y_h - curr_y * ref_x_h

    # 4th I normalise the axis
    axis_magnitude= math.sqrt(yaw_axis_x**2 + yaw_axis_y**2 + yaw_axis_z**2)
    if axis_magnitude < 1e-8:  # vectors already aligned, no yaw correction needed
        return 0.0, 0.0, 1.0, 0.0
    yaw_axis_x /= axis_magnitude
    yaw_axis_y /= axis_magnitude
    yaw_axis_z /= axis_magnitude

    # 5th, dot product + arccos gives the yaw error angle phi, same as tilt angle calculation
    dot_product= curr_x * ref_x_h + curr_y * ref_y_h + curr_z * ref_z_h
    dot_product= max(-1.0, min(1.0, dot_product))  # clamp to dodge arccos float errors
    yaw_phi_angle= math.acos(dot_product)

    return yaw_axis_x, yaw_axis_y, yaw_axis_z, yaw_phi_angle


# Now for the bulk of this question. This is my full complementary filter that fuses gyroscope & accelerometer with magnometer too, so yaw drift can also be detected
# the LaValle 2020 coursework book gave me the additional context needed to build on this function
# for reference, mag_alpha_tuner controls the magnetometer blend, which I've intentionally made seperate from accel_alpha_tuner as magnetic fields are more precarious than acceleromater gravity-based corrections
def full_fusion_orientation_calc(imu_dataframe, accel_alpha_tuner=HYPERPARAMS["ACCEL_ALPHA"],mag_alpha_tuner=HYPERPARAMS["MAG_ALPHA"]):
    # here I'm fetching the val of reference magnetometer & reading at frame 0 prior to any of the rotation movemnents starting to occur
    ref_mag_x, ref_mag_y, ref_mag_z= capture_magnetometer_reference(imu_dataframe)

    # this is my initial identity quaternion, same as before
    id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z= 1.0, 0.0, 0.0, 0.0

    # & my tracking array, again same pattern as my previous filters
    orientation_history_container = [(id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z)]

    # here I'm initialising the smoothed accel & mag values at rest assumption
    smoothed_accel_x, smoothed_accel_y, smoothed_accel_z = 0.0, 0.0, 1.0
    smoothed_mag_x, smoothed_mag_y, smoothed_mag_z= ref_mag_x, ref_mag_y, ref_mag_z
    accel_smooth_factor_for_ema= HYPERPARAMS["ACCEL_EMA_SMOOTH_FACTOR"]
    mag_smooth_factor_for_ema= HYPERPARAMS["MAG_EMA_SMOOTH_FACTOR"]

    for i in range(1, len(imu_dataframe)):
        # 1st, elapsed time between readings (same as always)
        elapsed_sensor_time_change = imu_dataframe['time'][i] - imu_dataframe['time'][i-1]

        # 2ndly gyroscope angular displacement for each axis
        change_over_roll  = imu_dataframe['gyroscope.X'][i] * elapsed_sensor_time_change
        change_over_pitch = imu_dataframe['gyroscope.Y'][i] * elapsed_sensor_time_change
        change_over_yaw   = imu_dataframe['gyroscope.Z'][i] * elapsed_sensor_time_change

        # 3rdly I'm dead reckoning the gyroscope forward one step
        w_change, x_change, y_change, z_change= convert_euler_to_quaternion(change_over_roll, change_over_pitch, change_over_yaw)
        gyro_w, gyro_x, gyro_y, gyro_z= calculate_quaternion_product(
            w_change, x_change, y_change, z_change,
            id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z)

        # 4th, normalise gyro estimate
        normed_gyro_estimate = math.sqrt(gyro_w**2 + gyro_x**2 + gyro_y**2 + gyro_z**2)
        gyro_w, gyro_x, gyro_y, gyro_z = gyro_w/normed_gyro_estimate, gyro_x/normed_gyro_estimate, gyro_y/normed_gyro_estimate, gyro_z/normed_gyro_estimate

        # now I do the same as before, roll & pitch correction w/ my acceleromater
        # 5th, smooth the raw accelerometer to reduce noise before tilt correction
        raw_accel_x= imu_dataframe['accelerometer.X'][i]
        raw_accel_y= imu_dataframe['accelerometer.Y'][i]
        raw_accel_z= imu_dataframe['accelerometer.Z'][i]
        smoothed_accel_x= accel_smooth_factor_for_ema * raw_accel_x + (1 - accel_smooth_factor_for_ema) * smoothed_accel_x
        smoothed_accel_y= accel_smooth_factor_for_ema * raw_accel_y + (1 - accel_smooth_factor_for_ema) * smoothed_accel_y
        smoothed_accel_z= accel_smooth_factor_for_ema * raw_accel_z + (1 - accel_smooth_factor_for_ema) * smoothed_accel_z

        # 6th is the magnitude gate that I'm talking about in the report. I'm only applying accel correction when reading is close to 1g
        magnitude_of_acceleromater= math.sqrt(smoothed_accel_x**2 + smoothed_accel_y**2 + smoothed_accel_z**2)
        accel_is_reliable= 0.85 < magnitude_of_acceleromater < 1.15

        # 7th, I rotate accel into global frame & calculate tilt correction
        global_accel_x, global_accel_y, global_accel_z= transform_vector_to_global_frame(gyro_w, gyro_x, gyro_y, gyro_z, smoothed_accel_x, smoothed_accel_y, smoothed_accel_z)
        tilt_axis_x, tilt_axis_y, tilt_axis_z, phi_angle= calculate_tilt_axis_and_angle(global_accel_x, global_accel_y, global_accel_z)

        # 8th, I build and apply accelerometer correction quaternion (gated on reliability again)
        accel_correction_angle= (accel_alpha_tuner * phi_angle) if accel_is_reliable else 0.0
        accel_corr_w= math.cos(accel_correction_angle / 2.0)
        accel_corr_x= tilt_axis_x * math.sin(accel_correction_angle/2.0)
        accel_corr_y= tilt_axis_y * math.sin(accel_correction_angle/2.0)
        accel_corr_z= tilt_axis_z * math.sin(accel_correction_angle/2.0)

        accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z = calculate_quaternion_product(accel_corr_w, accel_corr_x, accel_corr_y, accel_corr_z,gyro_w, gyro_x, gyro_y, gyro_z)

        # 9th. and now for the magnometer correction (new stuff), that's responsible for correcting the yaw (which I didn't correct before)
        # here I'm smoothing the raw magnetometer reading to reduce noise, because the magnometer is noisy inherently
        raw_mag_x, raw_mag_y, raw_mag_z = imu_dataframe['magnetometer.X'][i], imu_dataframe['magnetometer.Y'][i], imu_dataframe['magnetometer.Z'][i]
        smoothed_mag_x= mag_smooth_factor_for_ema*raw_mag_x + (1-mag_smooth_factor_for_ema)*smoothed_mag_x # (dev note: I would normally put these all on one line but it's going to be a tad messy with this smoothing formula)
        smoothed_mag_y= mag_smooth_factor_for_ema*raw_mag_y + (1-mag_smooth_factor_for_ema)*smoothed_mag_y
        smoothed_mag_z= mag_smooth_factor_for_ema*raw_mag_z + (1-mag_smooth_factor_for_ema)*smoothed_mag_z

        # 10th I'm rotating magnetometer reading-->global frame using accel-fused estimate
        global_mag_x, global_mag_y, global_mag_z= transform_vector_to_global_frame(accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z, # (p.s. I'm using the accellerometer-fused estimate instead of just gyro alone because it's more accurate i.e. it's already been corrected for roll & pitch)
            smoothed_mag_x, smoothed_mag_y, smoothed_mag_z)

        # 11th I'm calculating yaw correction axis and angle vs reference
        yaw_axis_x, yaw_axis_y, yaw_axis_z, yaw_phi= calculate_yaw_axis_and_angle( # this is projecting both current mag vector & reference one onto horizontal plane so that vertical component of magnetic field doesn't contaminate yaw estimate
            global_mag_x, global_mag_y, global_mag_z, ref_mag_x, ref_mag_y, ref_mag_z)

        # 12th I'm building & applying magnetometer yaw correction quaternion (pretty much same as I did for acceleromter correction)
        mag_correction_angle= mag_alpha_tuner*yaw_phi # 12.i I'm scaling yaw error by the mag_alpha_tuner param so only a little bit of the total error is being nudged each frame, & therefore the correction doesn't instantly get snapped too instanteneously
        mag_corr_w= math.cos(mag_correction_angle/2.0)
        mag_corr_x= yaw_axis_x*math.sin(mag_correction_angle/2.0)
        mag_corr_y= yaw_axis_y*math.sin(mag_correction_angle/2.0)
        mag_corr_z= yaw_axis_z*math.sin(mag_correction_angle/2.0)

        id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z= calculate_quaternion_product( # 12ii. then I convert scaled angle-->quaternion w/ axis-angle formula  & pre-multiply onto accel-fused estimate
            mag_corr_w, mag_corr_x, mag_corr_y, mag_corr_z, accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z)

        # 13th, normalise the final fused result before storing
        fused_norm= math.sqrt(id_ref_for_w**2 + id_ref_for_x**2 + id_ref_for_y**2 + id_ref_for_z**2)
        id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z= id_ref_for_w/fused_norm, id_ref_for_x/fused_norm, id_ref_for_y/fused_norm, id_ref_for_z/fused_norm

        # & now I'm done for this step, so I store and & move on
        orientation_history_container.append((id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z))

    return orientation_history_container


# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
# q5) last up is all the physics stuff like gravity and distance-based collision

# note: I reffered to the coursework spec + the supplementary nasa physics paper: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/falling-object-with-air-resistance/ 
def calc_accel_on_bunnies(y_comp_of_velocity):  # y component because that's the perpendicular to earth's surface so has effect of gravity
    # I chose to put these in the hyperparams list so this could be experimented w/ and changed really easily later on
    bunny_weight = HYPERPARAMS["GRAVITY"] * HYPERPARAMS["MASS"]  # (w=mg)
    bunny_drag = (HYPERPARAMS["DRAG_COEFFICIENT"] *0.5 * HYPERPARAMS["AIR_DENSITY"] * (y_comp_of_velocity ** 2) * HYPERPARAMS["AREA"])  # (D=Cd*.5*v^2*A gives opposition to motion)
    bunny_net_force = bunny_weight - bunny_drag  # (F=W-D)
    bunny_accel = bunny_net_force / HYPERPARAMS["MASS"]  # (a=F/m)
    return bunny_accel


# I set this as a class as it these falling bunnies are going to be reusable, so this way all the logic can just be encapsulated easily
class TumblingBunny:
    def __init__(self, start_x, start_y, start_z):
        # position in world space (y was defined as up because of up= np.array([0, 1, 0]..) line which was passed into lookat function)
        self.x, self.y, self.z= start_x, start_y, start_z
        self.vel_y= 0.0# initial vel=0, then bunny gets pushed down by g
        self.vel_x= 0.0  # and here I have the horizontal v components too
        self.vel_z= 0.0

        # I added a spin feature which can be turned off
        self.spin= 0.0
        self.spin_speed= HYPERPARAMS["SPIN_SPEED"] # rad/s; sourced from HYPERPARAMS so it's tweakable centrally

        # the ground_y here is the control for the off-screen removal boundary: once a bunny falls past this y val it's considered gone from the scene and gets cleared from the active list
        self.ground_y= HYPERPARAMS["OFFSCREEN_Y"]
        self.landed= False # note that I define 'landed' here just to mean 'exited the visible scene through the bottom'
        self.radius = HYPERPARAMS["TUMBLING_TEENY_BUNNY_RADIUS"] #this is the radius for collision with the little falling bunny

    # the logic to update the bunny's movement
    def update(self, dt):
        if self.landed:
            return
        
        # here I calculate the downward accel of a bunny
        speed = abs(self.vel_y)
        accel_down = calc_accel_on_bunnies(speed)

        # now I'm integrating velocity w/ position update
        self.vel_y -= accel_down*dt #(downward is -y not +y, so I need to subtract here)
        self.y+= self.vel_y*dt
        self.x += self.vel_x * dt # + horizontal components integrated
        self.z += self.vel_z * dt

        # this spin is entirely optional, I just thought it looks pretty cool. Can comment out if needed
        self.spin += self.spin_speed*dt

        # now I'm checking if the bunny has fallen off the bottom of the visible scene
        # (it can't keep falling if so, so my flag is updated and it'll be removed from the active list)
        if self.y <= self.ground_y:
            self.y= self.ground_y
            self.vel_y= 0.0
            self.vel_x = 0.0# x & z motion also stops once off screen; bunny is about to be deleted anyway
            self.vel_z = 0.0
            self.landed= True

    def get_model_matrix(self): # here I'm taking the falling bunny & translating to position, scaling down & spinning around y-axis
        return (matrix_translater(self.x, self.y, self.z) @ matrix_rotater_y(self.spin)
            @ matrix_scaler(HYPERPARAMS["FALLING_BUNNY_SCALE"], HYPERPARAMS["FALLING_BUNNY_SCALE"], HYPERPARAMS["FALLING_BUNNY_SCALE"]))


# now I'm adding the sphere-radius collision system in
def examine_collision_radius(x1,y1,z1,r1,x2,y2,z2,r2):
    # I'm just using the theory that if 2 speheres are overlapping theres a collision, which happens when the distance between their centres < sum of their radia
    dist_between_sphere_centres= math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    overlapping_bunny_spheres = dist_between_sphere_centres < (r1 + r2), dist_between_sphere_centres
    return overlapping_bunny_spheres


# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
# END OF MY ANSWERS TO ALL THE COURSEWORK QUESTIONS
# from here on down it's just the standard engine w/ modifications for video & 2 sequential sequences for the coursework.
# }}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

def triangle(t, v0, v1, v2, intensity):
    global coords, image, zbuffer

    # Triangle bounding box (clipped to screen)
    xmin = int(max(0,               min(v0[0], v1[0], v2[0])))
    xmax = int(min(image.shape[1],  max(v0[0], v1[0], v2[0]) + 1))
    ymin = int(max(0,               min(v0[1], v1[1], v2[1])))
    ymax = int(min(image.shape[0],  max(v0[1], v1[1], v2[1]) + 1))

    # All pixel positions in the bounding box (vectorized)
    P = coords[:, xmin:xmax, ymin:ymax].reshape(2, -1)

    # Barycentric coordinates for each pixel
    B = np.dot(t, np.vstack((P, np.ones((1, P.shape[1]), dtype=int))))

    # Keep pixels inside the triangle (α,β,γ >= 0)
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y = P[0, I], P[1, I]
    Z = v0[2]*B[0, I] + v1[2]*B[1, I] + v2[2]*B[2, I]

    # clip to screen bounds before zbuffer access
    valid = (X >= 0) & (X < image.shape[1]) & (Y >= 0) & (Y < image.shape[0])
    X, Y, Z = X[valid], Y[valid], Z[valid]

    # Z-buffer test (keep closest)
    I = np.argwhere(zbuffer[Y, X] < Z)[:, 0]
    X, Y, Z = X[I], Y[I], Z[I]
    zbuffer[Y, X] = Z
    image[Y, X] = intensity, intensity, intensity, 255

    # Outline (black) using line rasterization + depth check
    Pedge = []
    Pedge.extend(line(v0, v1))
    Pedge.extend(line(v1, v2))
    Pedge.extend(line(v2, v0))
    Pedge = np.array(Pedge).T  # shape (2, n)

    # clip edge points to screen bounds before zbuffer access
    valid = (
        (Pedge[0] >= 0) & (Pedge[0] < image.shape[1]) &
        (Pedge[1] >= 0) & (Pedge[1] < image.shape[0])
    )
    Pedge = Pedge[:, valid]
    if Pedge.shape[1] == 0:
        return

    B = np.dot(t, np.vstack((Pedge, np.ones((1, Pedge.shape[1]), dtype=int))))
    I = np.argwhere(np.all(B >= 0, axis=0))
    X, Y = Pedge[0, I], Pedge[1, I]
    Z = v0[2]*B[0, I] + v1[2]*B[1, I] + v2[2]*B[2, I]

    I = np.argwhere(zbuffer[Y, X] <= Z)[:, 0]
    X, Y = X[I], Y[I]
    image[Y, X] = 0, 0, 0, 255


def line(A, B):
    # Bresenham-ish integer line rasterizer; returns list[(x,y)]
    (x0, y0, _), (x1, y1, _) = np.array(A).astype(int), np.array(B).astype(int)
    P = []
    steep = False

    if abs(x0 - x1) < abs(y0 - y1):
        steep, x0, y0, x1, y1 = True, y0, x0, y1, x1

    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    dx, dy = x1 - x0, y1 - y0
    y, error2, derror2 = y0, 0, abs(dy) * 2

    for x in range(x0, x1 + 1):
        P.append((y, x) if steep else (x, y))
        error2 += derror2
        if error2 > dx:
            y += 1 if y1 > y0 else -1
            error2 -= dx * 2
    return P


def obj_load(filename):
    # Minimal OBJ loader: supports only 'v' and triangular 'f'
    V, Vi = [], []
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                # NOTE: expects faces like "f 1 2 3" (no slashes)
                Vi.append([int(x) for x in values[1:4]])
    return np.array(V), np.array(Vi) - 1


def lookat(eye, center, up):
    """
    Standard right-handed LookAt view matrix.
    Transforms world -> camera space.
    """
    def normalize(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    f = normalize(center - eye)         # forward
    s = normalize(np.cross(f, up))      # right
    u = np.cross(s, f)                  # corrected up

    M = np.eye(4, dtype=float)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f                       # camera looks down -Z
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] =  np.dot(f, eye)
    return M


def perspective(fov_deg, aspect, near, far):
    """
    Right-handed perspective projection matrix (OpenGL-style clip space).
    After multiplying: ndc = clip.xyz / clip.w
    """
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=float)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def viewport(x, y, w, h, d):
    # Maps NDC [-1..1] to screen coordinates
    return np.array([
        [w/2,   0,    0, x + w/2],
        [0,   h/2,    0, y + h/2],
        [0,     0,  d/2,     d/2],
        [0,     0,    0,       1]
    ], dtype=float)



# I wrapped up engine render logic into this helper so I can reuse it w/ each scene
# it's just all the stuf needed for projecting a mesh with a given model matrix into the global framebuffer 
def render_mesh_to_scene(V_inst, Vi_inst, model_matrix, view, proj, vp_mat, light):
    # Object vertices to homogeneous
    Vh = np.c_[V_inst, np.ones(len(V_inst))]  # (N,4)

    # now object space is converted to world space using the model matrix defined up above, for this particiarly version of the bunny
    V_world_h = Vh @ model_matrix.T # (transposing)

    # World -> camera
    V_cam_h = V_world_h @ view.T           # (N,4)
    V_cam   = V_cam_h[:, :3]               # keep for normals/lighting if desired

    # Camera -> clip
    clip = V_cam_h @ proj.T                # (N,4)

    # Perspective divide: clip -> NDC
    ndc = clip[:, :3] / clip[:, 3:4]

    # NDC -> screen
    ndc_h = np.c_[ndc, np.ones(len(ndc))]
    Vs_h  = ndc_h @ vp_mat.T
    Vs    = Vs_h[:, :3]                    # (N,3) screen x,y,z

    # Expand indexed faces (triangles)
    V_tri  = V_cam[Vi_inst]                # (F,3,3) for normals
    Vs_tri = Vs[Vi_inst]                   # (F,3,3) for rasterization

    # Pre-compute barycentric transforms per triangle (screen space)
    T = np.transpose(Vs_tri, axes=[0, 2, 1]).copy()  # (F,3,3) but columns are vertices
    T[:, 2, :] = 1                                   # replace z row with ones for barycentrics
    try:
        T = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return

    # Face normals (camera space) and flat intensity
    N = np.cross(V_tri[:, 2] - V_tri[:, 0], V_tri[:, 1] - V_tri[:, 0])
    N = N / (np.linalg.norm(N, axis=1).reshape(len(N), 1) + 1e-8) # just blocks /0 error

    # Light dot normal -> intensity
    I = np.dot(N, light) * 255

    # rasterisation time: now the valid faces get rendered
    for i in np.argwhere(I >= 0)[:, 0]:
        vs0, vs1, vs2 = Vs_tri[i]
        triangle(T[i], vs0, vs1, vs2, I[i])


# The following is now the sequence runner.
# this plays a sequence starting from a clean scene in the way I described at the top of the file.
def commence_full_sequence_execution(sequence_label, oreintations_set, imu_timestamps, local_V, Vi, view, proj, vp_mat, light, render_stride, video_filename=None):
    global image, zbuffer

    width, height, record= HYPERPARAMS["WIDTH"],  HYPERPARAMS["HEIGHT"], HYPERPARAMS["RECORD_VIDEO"]

    # here I have all the video writer setup stuff (but it's only actually used when RECORD_VIDEO is set to True)
    mp4_seq_output= None
    if record:
        # here I'm getting the by which I want to record (I explained full logic above in hyperparams list)
        output_vids_fps= max(1, HYPERPARAMS["VIDEO_FPS_DIVISOR"] // render_stride)
        mp4_seq_output= cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*"mp4v"), output_vids_fps, (width, height))
        print(f"Ok, show time! The recording of '{sequence_label}' to {video_filename} @ {output_vids_fps} fps is now commencing!!")

    # this is the falling bunny state tracker, which I specifically ensure is a fresh list per new sequence
    active_bunnies= []   # active_bunnies is holding all TumblingBunny instances currently in the scene, and is updated as new ones spawn and old ones fall out of it

    # here I'm picking the first spawn time as a random offset from the sequence start (in real IMU secs)
    seq_start_time= imu_timestamps[0]
    next_spawn_at= seq_start_time + random.uniform(HYPERPARAMS["SPAWN_INTERVAL_MIN"], HYPERPARAMS["SPAWN_INTERVAL_MAX"])

    # physics timestep and substep count (from HYPERPARAMS so they're tweakable)
    sim_timeshift= HYPERPARAMS["SIM_TIMESHIFT"]
    engine_substeps= HYPERPARAMS["ENGINE_SUBSTEPS"]

    print(f"\n{sequence_label}")
    print("")

    for frame_idx in range(0, len(oreintations_set), render_stride):

        # this is just the real elapsed time for this rendered frame, which is gotten directly off of IMU timestamps
        current_time= imu_timestamps[frame_idx]

        # here a new falling bunny is spawed whenever the random interval elapses
        if current_time >= next_spawn_at:
            # the random x/z offset is within SPAWN_RADIUS (one of my hyperparams) of the central bunny at the origin.
            # z-range here is a bit compressed by design so that bunnies should stay roughly in front of the camera
            offset_x= random.uniform(-HYPERPARAMS["SPAWN_RADIUS"], HYPERPARAMS["SPAWN_RADIUS"])
            offset_z= random.uniform(-HYPERPARAMS["SPAWN_RADIUS"] * 0.5, HYPERPARAMS["SPAWN_RADIUS"] *0.5)
            active_bunnies.append(TumblingBunny(offset_x, HYPERPARAMS["Y_HEIGHT_TO_SPAWN_BUNNY"], offset_z))
            print(f"Update!! A new bunny was spawned at time [t={current_time:.1f}s] and position: x={offset_x:.2f} z={offset_z:.2f}")

            # now I schedule the next spawn (which is another random interval after this one)
            next_spawn_at = current_time + random.uniform(HYPERPARAMS["SPAWN_INTERVAL_MIN"], HYPERPARAMS["SPAWN_INTERVAL_MAX"])

        # now for all the physics. I update every active falling bunny, then check collisions
        for falling_little_bunny in active_bunnies:
            for _ in range(engine_substeps): # param controlling their fall
                falling_little_bunny.update(sim_timeshift)# timestep's updated.

                # don't bother with collision checks if the bunny has already left the scene so as not to waste compute (even though really its probably neglible)
                if falling_little_bunny.landed:
                    continue

                # the requested sphere collision radius test commences, and that's between this falling bunny & the central bunny
                colliding, dist = examine_collision_radius(falling_little_bunny.x, falling_little_bunny.y, falling_little_bunny.z, falling_little_bunny.radius,
                    0.0, 0.0, 0.0, HYPERPARAMS['BIG_BOSS_BUNNY_RADIUS'])  #I'm using the fact that the central big boss bunny is always at the origin, so I can just use origin coords

                # if this triggers, a collision has occured! I deal with it here.
                if colliding:
                    # computing impact speed before modifying anything
                    impact_speed = abs(falling_little_bunny.vel_y)

                    # a bounce upward would naturally lose a bit of energy (and coefficient of restitution is something I set in my params to account for this)
                    falling_little_bunny.vel_y = impact_speed * HYPERPARAMS["RESTITUTION"]

                    # push horizontally away from centre as a proper velocity impulse, conserving momentum
                    horiz_dist = math.sqrt(falling_little_bunny.x**2 + falling_little_bunny.z**2)
                    horiz_kickoff = min(impact_speed * 0.4, HYPERPARAMS["HORIZ_KICKOFF_CAP"])  # I've capped it here so fast bunnies don't fly off screen
                    if horiz_dist > 0.01:
                        falling_little_bunny.vel_x= (falling_little_bunny.x / horiz_dist) * horiz_kickoff
                        falling_little_bunny.vel_z= (falling_little_bunny.z / horiz_dist) * horiz_kickoff
                    else:
                        # if this triggers there's been an impressivly dead centre hit, so just kick sideways arbitrarily
                        falling_little_bunny.vel_x= impact_speed * 0.5
                        falling_little_bunny.vel_z= 0.0

                    # here I'm separating the spheres so they don't re-collide next frame
                    combined_radius= falling_little_bunny.radius + HYPERPARAMS['BIG_BOSS_BUNNY_RADIUS']
                    if horiz_dist > 0.01:
                        overlap = combined_radius - dist
                        falling_little_bunny.x += (falling_little_bunny.x / horiz_dist) * overlap
                        falling_little_bunny.z += (falling_little_bunny.z / horiz_dist) * overlap

                    print(f"look out! a little bunny has hit the big one!! collision pos was ({falling_little_bunny.x:.2f},{falling_little_bunny.y:.2f},{falling_little_bunny.z:.2f}), dist= {dist:.2f}")
                    break  # one collision response per substep is quite enough, so I move on (can deal with more in subsequent substeps)

        # here I remove bunnies that have fallen past the off-screen threshold (their landed flag was set in update())
        active_bunnies = [falling_little_bunny for falling_little_bunny in active_bunnies if not falling_little_bunny.landed]

        # now i clear framebuffer and zbuffer for this frame
        image[:], zbuffer[:]= 0, -1e18

        # & get fused orientation for this frame and convert to euler angles for existing rotation matrices
        quat_w, quat_x, quat_y, quat_z= oreintations_set[frame_idx]
        roll_val, pitch_val, yaw_val= convert_quaternion_to_euler(quat_w, quat_x, quat_y, quat_z)

        # and now for building scene. 1st central rotating bunny then all falling bunnies 
        scene_variations = [
            # central bunny is always origin and executing that IMU-driven rotation (the key orientation I'm being asked to demonstrate)
            {"mesh": local_V, "faces": Vi,
             "model_matrix": matrix_rotater_z(yaw_val) @ matrix_rotater_y(pitch_val) @ matrix_rotater_x(roll_val)},
        ]
        # I add each active falling bunny to scene here (each has its own world-space position & spin state)
        for falling_little_bunny in active_bunnies:
            scene_variations.append({
                "mesh": local_V, "faces": Vi,
                "model_matrix": falling_little_bunny.get_model_matrix()
            })

        # timer gets fired up to determine when collisions occur
        start = _time.time()

        for scene in scene_variations:
            render_mesh_to_scene(
                scene["mesh"], scene["faces"], scene["model_matrix"],
                view, proj, vp_mat, light)

        end = _time.time()
        print(f"Rendering time: {end - start:.3f}s  (frame {frame_idx}/{len(oreintations_set)})", end='\r')

        # this is to enable the real-time rendering of frame buffer on screen for coursework
        frame = image[::-1, :, :3] # here I normalise channels, droping alpha for opencv for just plain rgb
        # vertical inversion corrects the coordinate system mismatch between rasteriser & standard screen coord system

        if record:
            # in video mode: write each rendered frame directly into the VideoWriter; no window opened
            mp4_seq_output.write(frame)
        else:
            cv2.imshow(sequence_label, frame) # flash up the bunny from frame buffer
            if cv2.waitKey(1) & 0xFF == ord('q'): # advance frames automatically; q to quit early
                break

    # sequence is done. video gets saved.
    if record:
        mp4_seq_output.release()
        print(f"\n  Done - video saved as {video_filename}")
    else:
        print(f"\nAll done, press anything now to close off the bunny.")
        cv2.waitKey(0) # I keep it up on screen until user interrupt
        cv2.destroyAllWindows()


# main loop from here down
if __name__ == '__main__':

    width  = HYPERPARAMS["WIDTH"]
    height = HYPERPARAMS["HEIGHT"]

    # (I've modified these from the engine default with my variable params technique so I can tune them for a nicer angle)
    light  = np.array([0, 0, -1], dtype=float)  # directional light
    eye    = np.array(HYPERPARAMS["EYE"],    dtype=float)
    center = np.array(HYPERPARAMS["CENTER"], dtype=float)
    up     = np.array(HYPERPARAMS["UP"],     dtype=float)

    image   = np.zeros((height, width, 4), dtype=np.uint8)
    zbuffer = -1e18 * np.ones((height, width), dtype=float)  # very far away

    # coords[0] = x grid, coords[1] = y grid
    coords = np.mgrid[0:width, 0:height].astype(int)

    V, Vi = obj_load("bunny.obj")

    # Centering and scaling the model roughly into view (this is still object space)
    vmin, vmax = V.min(), V.max()
    local_V = (2 * (V - vmin) / (vmax - vmin) - 1) * 1.25
    local_V[:, 0] -= (local_V[:, 0].min() + local_V[:, 0].max()) / 2
    local_V[:, 1] -= (local_V[:, 1].min() + local_V[:, 1].max()) / 2

    # Matrices
    vp_mat = viewport(32, 32, width - 64, height - 64, 1000)
    view   = lookat(eye, center, up)
    proj= perspective(fov_deg=HYPERPARAMS["FOV_DEG"], aspect=width / height, near=HYPERPARAMS["NEAR"], far=HYPERPARAMS["FAR"])

    # I choose render stride based on mode (realtime or video)
    # video mode: fixed fine stride for smooth output; real-time: coarser for display responsiveness
    if HYPERPARAMS["RECORD_VIDEO"]:
        render_stride = HYPERPARAMS["VIDEO_RENDER_STRIDE"]
    else:
        # stride through frames so rendering stays manageable on a software renderer
        render_stride = max(1, len(imu_data) // HYPERPARAMS["REALTIME_RENDER_DIVISOR"]) #note: I explain how this works at the begging of the file in the params list

    # extract timestamps once up front so the sequence runner can key spawns off them
    imu_timestamps = list(imu_data['time'])

    # 1st requested sequence: gyro + accelerometer complementary filter
    # central bunny orientation: dead reckoning corrected w/ accel tilt only (but no mag)
    print("First part of the coursework! Complementary filter (gyro + accel) firing up...")
    fused_orientations = complementary_filter_orientation_calc(imu_data, alpha=HYPERPARAMS["ACCEL_ALPHA"])
    print(f"Fetched a total of {len(fused_orientations)} orientation frames from IMU data")

    commence_full_sequence_execution(
        sequence_label= "1st sequence: Gyro + Accel complementary filter",
        oreintations_set= fused_orientations, imu_timestamps= imu_timestamps,
        local_V= local_V, Vi= Vi, view= view, proj= proj, vp_mat= vp_mat, light= light,
        render_stride= render_stride,
        video_filename= HYPERPARAMS["FILENAME_OF_FIRST_SEQ"],
    )

    # 2nd requested sequence: full fusion (gyro + accelerometer + magnetometer)
    # central bunny orientation: gyro + accel tilt correction + magnetometer yaw 'correction'
    print("\n& 2nd part of coursework: the triple correction (gyro, accel, magnometer) complementary filter firing up...")
    full_fused_orientations = full_fusion_orientation_calc(imu_data, accel_alpha_tuner=HYPERPARAMS["ACCEL_ALPHA"], mag_alpha_tuner=HYPERPARAMS["MAG_ALPHA"])
    print(f"Fetched a total of {len(full_fused_orientations)} orientation frames")

    commence_full_sequence_execution(
        sequence_label= "2nd sequence: Full fusion (gyro + accel + magnetometer)",
        oreintations_set= full_fused_orientations, imu_timestamps= imu_timestamps,
        local_V= local_V, Vi= Vi, view= view, proj= proj,
        vp_mat= vp_mat, light= light, render_stride= render_stride,
        video_filename= HYPERPARAMS["FILENAME_OF_SECOND_SEQ"],
    )

    print("\nAnndd that's a wrap! Both sequences are now complete.")
