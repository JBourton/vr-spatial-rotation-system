# Tiny 3D Renderer (with outlines) augmented with perspective projection
import numpy as np
import cv2
import pandas as pd
import math

# firstly, I define the functions for generating matrices for scaling & translating operations
# these are all my Q1 answers
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
    imu_dataset=  pd.read_csv(imu_data_filename, skipinitialspace=True) # (the file seemed to have a leading whitespace, so this operation was added to ditch it)
    imu_dataset.columns= imu_dataset.columns.str.strip()
    
    # as required by cousrework q2.1.i, this takes the rotational rates from deg/s to rad/s
    gyroscope_cols_in_rads_to_convert= ['gyroscope.X', 'gyroscope.Y', 'gyroscope.Z']
    imu_dataset[gyroscope_cols_in_rads_to_convert]= np.deg2rad(imu_dataset[gyroscope_cols_in_rads_to_convert])

    return imu_dataset


imu_data = readin_imu_dataset()
print(imu_data.head())

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
def complementary_filter_orientation_calc(imu_dataframe, alpha=0.02):
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
            # magnitude gate failed - just use the gyro estimate unchanged this step
            w_id_ref, x_id_ref, y_id_ref, z_id_ref = gyro_w, gyro_x, gyro_y, gyro_z

        # 10th I normalise fused result before storing
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
    curr_x, curr_y, curr_z = global_mag_x, global_mag_y, 0.0
    ref_x_h, ref_y_h, ref_z_h = ref_mag_x, ref_mag_y, 0.0

    # 2nd, normalise both projected vectors
    curr_magnitude = math.sqrt(curr_x**2 + curr_y**2 + curr_z**2)
    ref_magnitude  = math.sqrt(ref_x_h**2 + ref_y_h**2 + ref_z_h**2)

    # guard against zero vectors (e.g. if mag reading is purely vertical)
    if curr_magnitude < 1e-8 or ref_magnitude < 1e-8:
        return 0.0, 0.0, 1.0, 0.0  # no correction, axis pointing up
    curr_x /= curr_magnitude
    curr_y /= curr_magnitude
    ref_x_h /= ref_magnitude
    ref_y_h /= ref_magnitude

    # 3rd, cross product gives the yaw correction axis (will point along Z since both vectors are horizontal)
    yaw_axis_x = curr_y * ref_z_h - curr_z * ref_y_h
    yaw_axis_y = curr_z * ref_x_h - curr_x * ref_z_h
    yaw_axis_z = curr_x * ref_y_h - curr_y * ref_x_h

    # 4th, normalise the axis
    axis_magnitude = math.sqrt(yaw_axis_x**2 + yaw_axis_y**2 + yaw_axis_z**2)
    if axis_magnitude < 1e-8:  # vectors already aligned, no yaw correction needed
        return 0.0, 0.0, 1.0, 0.0
    yaw_axis_x /= axis_magnitude
    yaw_axis_y /= axis_magnitude
    yaw_axis_z /= axis_magnitude

    # 5th, dot product + arccos gives the yaw error angle phi, same as tilt angle calculation
    dot_product = curr_x * ref_x_h + curr_y * ref_y_h + curr_z * ref_z_h
    dot_product = max(-1.0, min(1.0, dot_product))  # clamp to dodge arccos float errors
    yaw_phi_angle = math.acos(dot_product)

    return yaw_axis_x, yaw_axis_y, yaw_axis_z, yaw_phi_angle


# this is my full complementary filter that fuses gyroscope & accelerometer with magnometer too, so yaw drift can also be detected
# the LaValle coursework book gave me the additional context needed to build on this function
# mag_alpha_tuner controls the magnetometer blend, which I've intentionally made seperate from accel_alpha_tuner as magnetic fields are more precarious than acceleromater gravity-based corrections
def full_fusion_orientation_calc(imu_dataframe, accel_alpha_tuner=0.02, mag_alpha_tuner=0.02):
    # here I'm fetching the val of reference magnetometer & reading at frame 0 prior to any of the rotation movemnents starting to occur
    ref_mag_x, ref_mag_y, ref_mag_z= capture_magnetometer_reference(imu_dataframe)

    # this is my initial identity quaternion, same as before
    w_id_ref, x_id_ref, y_id_ref, z_id_ref= 1.0, 0.0, 0.0, 0.0

    # tracking array, again same pattern as my previous filters
    orientation_history_container = [(w_id_ref, x_id_ref, y_id_ref, z_id_ref)]

    # here I'm initialising the smoothed accel & mag values at rest assumption
    smoothed_accel_x, smoothed_accel_y, smoothed_accel_z = 0.0, 0.0, 1.0
    smoothed_mag_x, smoothed_mag_y, smoothed_mag_z= ref_mag_x, ref_mag_y, ref_mag_z
    accel_smooth_factor= 0.1
    mag_smooth_factor= 0.1

    for i in range(1, len(imu_dataframe)):
        # 1st, elapsed time between readings (same as always)
        elapsed_sensor_time_change = imu_dataframe['time'][i] - imu_dataframe['time'][i-1]

        # 2ndly gyroscope angular displacement for each axis
        change_over_roll  = imu_dataframe['gyroscope.X'][i] * elapsed_sensor_time_change
        change_over_pitch = imu_dataframe['gyroscope.Y'][i] * elapsed_sensor_time_change
        change_over_yaw   = imu_dataframe['gyroscope.Z'][i] * elapsed_sensor_time_change

        # 3rdly I'm dead reckoning the gyroscope forward one step
        w_change, x_change, y_change, z_change = convert_euler_to_quaternion(change_over_roll, change_over_pitch, change_over_yaw)
        gyro_w, gyro_x, gyro_y, gyro_z = calculate_quaternion_product(
            w_change, x_change, y_change, z_change,
            w_id_ref, x_id_ref, y_id_ref, z_id_ref)

        # 4th, normalise gyro estimate
        normed_gyro_estimate = math.sqrt(gyro_w**2 + gyro_x**2 + gyro_y**2 + gyro_z**2)
        gyro_w, gyro_x, gyro_y, gyro_z = gyro_w/normed_gyro_estimate, gyro_x/normed_gyro_estimate, gyro_y/normed_gyro_estimate, gyro_z/normed_gyro_estimate

        # now I do the same as before, roll & pitch correction w/ my acceleromater
        # 5th, smooth the raw accelerometer to reduce noise before tilt correction
        raw_accel_x = imu_dataframe['accelerometer.X'][i]
        raw_accel_y = imu_dataframe['accelerometer.Y'][i]
        raw_accel_z = imu_dataframe['accelerometer.Z'][i]
        smoothed_accel_x = accel_smooth_factor * raw_accel_x + (1 - accel_smooth_factor) * smoothed_accel_x
        smoothed_accel_y = accel_smooth_factor * raw_accel_y + (1 - accel_smooth_factor) * smoothed_accel_y
        smoothed_accel_z = accel_smooth_factor * raw_accel_z + (1 - accel_smooth_factor) * smoothed_accel_z

        # 6th is the magnitude gate that I'm talking about in the report. I'm only applying accel correction when reading is close to 1g
        magnitude_of_acceleromater= math.sqrt(smoothed_accel_x**2 + smoothed_accel_y**2 + smoothed_accel_z**2)
        accel_is_reliable= 0.85 < magnitude_of_acceleromater < 1.15

        # 7th, I rotate accel into global frame & calculate tilt correction
        global_accel_x, global_accel_y, global_accel_z= transform_vector_to_global_frame(
            gyro_w, gyro_x, gyro_y, gyro_z,
            smoothed_accel_x, smoothed_accel_y, smoothed_accel_z)
        
        tilt_axis_x, tilt_axis_y, tilt_axis_z, phi_angle= calculate_tilt_axis_and_angle(
            global_accel_x, global_accel_y, global_accel_z)

        # 8th, I build and apply accelerometer correction quaternion (gated on reliability again)
        accel_correction_angle= (accel_alpha_tuner * phi_angle) if accel_is_reliable else 0.0
        accel_corr_w= math.cos(accel_correction_angle / 2.0)
        accel_corr_x= tilt_axis_x * math.sin(accel_correction_angle/2.0)
        accel_corr_y= tilt_axis_y * math.sin(accel_correction_angle/2.0)
        accel_corr_z= tilt_axis_z * math.sin(accel_correction_angle/2.0)

        accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z = calculate_quaternion_product(
            accel_corr_w, accel_corr_x, accel_corr_y, accel_corr_z,
            gyro_w, gyro_x, gyro_y, gyro_z)

        # 9th. and now for the magnometer correction (new stuff), that's responsible for correcting the yaw (which I didn't correct before)
        # here I'm smoothing the raw magnetometer reading to reduce noise, because the magnometer is noisy inherently
        raw_mag_x, raw_mag_y, raw_mag_z = imu_dataframe['magnetometer.X'][i], imu_dataframe['magnetometer.Y'][i], imu_dataframe['magnetometer.Z'][i]
        smoothed_mag_x= mag_smooth_factor*raw_mag_x + (1-mag_smooth_factor)*smoothed_mag_x # (dev note: I would put these all on one line but it's going to be a tad messy with this smoothing formula)
        smoothed_mag_y= mag_smooth_factor*raw_mag_y + (1-mag_smooth_factor)*smoothed_mag_y
        smoothed_mag_z= mag_smooth_factor*raw_mag_z + (1-mag_smooth_factor)*smoothed_mag_z

        # 10th I'm rotating magnetometer reading-->global frame using accel-fused estimate
        global_mag_x, global_mag_y, global_mag_z= transform_vector_to_global_frame(
            accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z, # (p.s. I'm using the accellerometer-fused estimate because it's more accurate i.e. it's already been corrected for roll & pitch)
            smoothed_mag_x, smoothed_mag_y, smoothed_mag_z)

        # 11th I'm calculating yaw correction axis and angle vs reference
        yaw_axis_x, yaw_axis_y, yaw_axis_z, yaw_phi= calculate_yaw_axis_and_angle( # this is projecting both current mag vector & reference one onto horizontal plane so that vertical component of magnetic field doesn't contaminate yaw estimate
            global_mag_x, global_mag_y, global_mag_z,
            ref_mag_x, ref_mag_y, ref_mag_z)

        # 12th I'm building & applying magnetometer yaw correction quaternion (pretty much same as I did for acceleromter correction)
        mag_correction_angle= mag_alpha_tuner*yaw_phi # 12.i I'm scaling yaw error by the mag_alpha_tuner param so only a little bit of the total error is being nudged each frame, & therefore the correction doesn't instantly get snapped too instanteneously
        mag_corr_w= math.cos(mag_correction_angle/2.0)
        mag_corr_x= yaw_axis_x*math.sin(mag_correction_angle/2.0)
        mag_corr_y= yaw_axis_y*math.sin(mag_correction_angle/2.0)
        mag_corr_z= yaw_axis_z*math.sin(mag_correction_angle/2.0)

        w_id_ref, x_id_ref, y_id_ref, z_id_ref= calculate_quaternion_product( # 12ii. then I convert scaled angle-->quaternion w/ axis-angle formula  & pre-multiply onto accel-fused estimate
            mag_corr_w, mag_corr_x, mag_corr_y, mag_corr_z,
            accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z)

        # 13th, normalise the final fused result before storing
        fused_norm= math.sqrt(w_id_ref**2 + x_id_ref**2 + y_id_ref**2 + z_id_ref**2)
        w_id_ref, x_id_ref, y_id_ref, z_id_ref= w_id_ref/fused_norm, x_id_ref/fused_norm, y_id_ref/fused_norm, z_id_ref/fused_norm

        # done for this step, store and move on
        orientation_history_container.append((w_id_ref, x_id_ref, y_id_ref, z_id_ref))

    return orientation_history_container




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


if __name__ == '__main__':
    import time
    import PIL.Image

    width, height = 1200, 1200

    light  = np.array([0, 0, -1], dtype=float)  # directional light
    eye    = np.array([-1.5, 1, 2], dtype=float)
    center = np.array([0, 0, 0], dtype=float)
    up     = np.array([0, 1, 0], dtype=float)

    image = np.zeros((height, width, 4), dtype=np.uint8)
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
    vp_mat  = viewport(32, 32, width - 64, height - 64, 1000)
    view    = lookat(eye, center, up)
    proj    = perspective(fov_deg=60, aspect=width/height, near=0.1, far=100.0)

    # now I run the complementary filter so the IMU_driven orientations can be fetched
    print("Running complementary filter...")
    fused_orientations = complementary_filter_orientation_calc(imu_data, alpha=0.02)
    print(f"Got {len(fused_orientations)} orientation frames from IMU data")

    # this is my experiment of different alpha values for the report
    complementary_filter_alpha_test_vals = [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    comp_filter_alpha_testing_result_log = {}

    for instance_of_alpha in complementary_filter_alpha_test_vals:
        print(f"executing 1x complementary filter test w/ alpha={instance_of_alpha}...")
        comp_filter_alpha_testing_result_log[instance_of_alpha]= complementary_filter_orientation_calc(imu_data,alpha=instance_of_alpha)

    # now the comparison at some of the key frames is printed
    key_frames_to_display_rpy_vals = [0, 1085, 1812, 2318, 2969, 4019, 4637, 4881, 5689, 6958]#[0, 860, 1720, 2580, 3440, 4300, 5160, 6020, 6880]
    print(f"\n{'frame':>6} | {'alpha':>6} | {'roll':>8} | {'pitch':>8} | {'yaw':>8}")
    print("-" * 55)
    for frame_idx in key_frames_to_display_rpy_vals:
        for instance_of_alpha, orientations in comp_filter_alpha_testing_result_log.items():
            q_w, q_x, q_y, q_z = orientations[frame_idx]
            roll, pitch, yaw = convert_quaternion_to_euler(q_w, q_x, q_y, q_z)
            print(f"{frame_idx:>6} | {instance_of_alpha:>6} | {math.degrees(roll):>+8.2f}° | {math.degrees(pitch):>+8.2f}° | {math.degrees(yaw):>+8.2f}°")
        print()

    # stride through frames so rendering stays manageable on a software renderer
    render_stride = max(1, len(fused_orientations) // 80)

    # # pt1 is the gyro + accelerameter complementary filter
    # for frame_idx in range(0, len(fused_orientations), render_stride):

    #     # clear framebuffer and zbuffer for this frame
    #     image[:] = 0
    #     zbuffer[:] = -1e18

    #     # get fused orientation for this frame and convert to euler angles for our existing rotation matrices
    #     q_w, q_x, q_y, q_z = fused_orientations[frame_idx]
    #     roll, pitch, yaw = convert_quaternion_to_euler(q_w, q_x, q_y, q_z)

    #     # [DEBUGGING TO VERIFY ACCURACY IN ANOTHER WAY]
    #     if frame_idx % (render_stride * 10) == 0:
    #         print(f"frame {frame_idx:4d} | roll={math.degrees(roll):+7.2f}° pitch={math.degrees(pitch):+7.2f}° yaw={math.degrees(yaw):+7.2f}°")

    #     # & now a couple of scene variations using that id matrix are defined
    #     scene_variations = [
    #         {"mesh": local_V, "faces": Vi, "model_matrix": matrix_rotater_z(yaw) @ matrix_rotater_y(pitch) @ matrix_rotater_x(roll)}, # IMU-driven rotation for sequence 1
    #         #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(0.8, 0, 0) @ matrix_scaler(1.8, 1.1, 1.2)}, # i added a bit of distortion so it's more obvious when they'e side-by-side
    #         # these are all little so as not to overcrowd the output
    #         #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(-0.5, 0.5, 0) @ matrix_scaler(0.4, 0.4, 0.4) @ matrix_rotater_x(np.pi/4)}, # 45deg rotation, left shift
    #         #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(0, 0.8, -0.5) @ matrix_scaler(0.4, 0.4, 0.4) @ matrix_rotater_y(np.pi/2)}, #90deg around y, up shift
    #         #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(0, -0.8, 0) @ matrix_scaler(0.4, 0.4, 0.8) @ matrix_rotater_z(np.pi/6)} # 30 deg around z, downshift
    #     ]

    #     start = time.time()

    #     for scene in scene_variations:
    #         V_inst = scene["mesh"]
    #         Vi_inst = scene["faces"]
    #         model_matrix = scene["model_matrix"]

    #         # Object vertices to homogeneous
    #         Vh = np.c_[V_inst, np.ones(len(V_inst))]  # (N,4)

    #         # now object space is converted to world space using the model matrix defined up above, for this particiarly version of the bunny
    #         V_world_h = Vh @ model_matrix.T # (transposing)

    #         # World -> camera
    #         V_cam_h = V_world_h @ view.T           # (N,4)
    #         V_cam   = V_cam_h[:, :3]               # keep for normals/lighting if desired

    #         # Camera -> clip
    #         clip = V_cam_h @ proj.T                # (N,4)

    #         # Perspective divide: clip -> NDC
    #         ndc = clip[:, :3] / clip[:, 3:4]

    #         # NDC -> screen
    #         ndc_h = np.c_[ndc, np.ones(len(ndc))]
    #         Vs_h  = ndc_h @ vp_mat.T
    #         Vs    = Vs_h[:, :3]                    # (N,3) screen x,y,z

    #         # Expand indexed faces (triangles)
    #         V_tri  = V_cam[Vi_inst]                # (F,3,3) for normals
    #         Vs_tri = Vs[Vi_inst]                   # (F,3,3) for rasterization

    #         # Pre-compute barycentric transforms per triangle (screen space)
    #         T = np.transpose(Vs_tri, axes=[0, 2, 1]).copy()  # (F,3,3) but columns are vertices
    #         T[:, 2, :] = 1                                   # replace z row with ones for barycentrics
    #         try:
    #             T = np.linalg.inv(T)
    #         except np.linalg.LinAlgError:
    #             continue

    #         # Face normals (camera space) and flat intensity
    #         N = np.cross(V_tri[:, 2] - V_tri[:, 0], V_tri[:, 1] - V_tri[:, 0])
    #         N = N / (np.linalg.norm(N, axis=1).reshape(len(N), 1) + 1e-8) # just blocks /0 error

    #         # Light dot normal -> intensity
    #         I = np.dot(N, light) * 255

    #         # rasterisation time: now the valid faces get rendered
    #         for i in np.argwhere(I >= 0)[:, 0]:
    #             vs0, vs1, vs2 = Vs_tri[i]
    #             triangle(T[i], vs0, vs1, vs2, I[i])

    #     end = time.time()

    #     print(f"Rendering time: {end - start:.3f}s  (frame {frame_idx}/{len(fused_orientations)})", end='\r')

    #     #PIL.Image.fromarray(image[::-1, :, :]).save("bunny.png")

    #     # this is to enable the real-time rendering of frame buffer on screen for coursework
    #     frame = image[::-1, :, :3] # here I normalise channels, droping alpha for opencv for just plain rgb
    #     # vertical inversion corrects the coordinate system mismatch between rasteriser & standard screen coord system

    #     cv2.imshow("Renderer", frame) # flash up the bunny from frame buffer
    #     if cv2.waitKey(1) & 0xFF == ord('q'): # advance frames automatically; q to quit early
    #         break

    # print("\nall done, just press any key now to close it all off.")
    # cv2.waitKey(0) # now keep it up on screen until user interrupt

    # here I'm now applying the full fusion combo of gyro + accelerometer + magneometer
    print("Triple correction (gyro, accel, magnometer) complementary filter firing up...")
    full_fused_orientations = full_fusion_orientation_calc(imu_data, accel_alpha_tuner=0.02, mag_alpha_tuner=0.02)
    print(f"Got {len(full_fused_orientations)} orientation frames")

    for frame_idx in range(0, len(full_fused_orientations), render_stride):

        # clear framebuffer and zbuffer for this frame
        image[:] = 0
        zbuffer[:] = -1e18

        q_w, q_x, q_y, q_z = full_fused_orientations[frame_idx]
        roll, pitch, yaw = convert_quaternion_to_euler(q_w, q_x, q_y, q_z)

        # [DEBUGGING TO VERIFY ACCURACY IN ANOTHER WAY]
        if frame_idx % (render_stride * 10) == 0:
            print(f"frame {frame_idx:4d} | roll={math.degrees(roll):+7.2f}° pitch={math.degrees(pitch):+7.2f}° yaw={math.degrees(yaw):+7.2f}°")

        # & now a couple of scene variations using that id matrix are defined
        scene_variations = [
            {"mesh": local_V, "faces": Vi, "model_matrix": matrix_rotater_z(yaw) @ matrix_rotater_y(pitch) @ matrix_rotater_x(roll)}, # IMU-driven rotation for sequence 2
            #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(0.8, 0, 0) @ matrix_scaler(1.8, 1.1, 1.2)},
            #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(-0.5, 0.5, 0) @ matrix_scaler(0.4, 0.4, 0.4) @ matrix_rotater_x(np.pi/4)},
            #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(0, 0.8, -0.5) @ matrix_scaler(0.4, 0.4, 0.4) @ matrix_rotater_y(np.pi/2)},
            #{"mesh": local_V, "faces": Vi, "model_matrix": matrix_translater(0, -0.8, 0) @ matrix_scaler(0.4, 0.4, 0.8) @ matrix_rotater_z(np.pi/6)}
        ]

        start = time.time()

        for scene in scene_variations:
            V_inst = scene["mesh"]
            Vi_inst = scene["faces"]
            model_matrix = scene["model_matrix"]

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
                continue

            # Face normals (camera space) and flat intensity
            N = np.cross(V_tri[:, 2] - V_tri[:, 0], V_tri[:, 1] - V_tri[:, 0])
            N = N / (np.linalg.norm(N, axis=1).reshape(len(N), 1) + 1e-8) # just blocks /0 error

            # Light dot normal -> intensity
            I = np.dot(N, light) * 255

            # rasterisation time: now the valid faces get rendered
            for i in np.argwhere(I >= 0)[:, 0]:
                vs0, vs1, vs2 = Vs_tri[i]
                triangle(T[i], vs0, vs1, vs2, I[i])

        end = time.time()

        print(f"Rendering time: {end - start:.3f}s  (frame {frame_idx}/{len(full_fused_orientations)})", end='\r')

        #PIL.Image.fromarray(image[::-1, :, :]).save("bunny.png")

        # this is to enable the real-time rendering of frame buffer on screen for coursework
        frame = image[::-1, :, :3] # here I normalise channels, droping alpha for opencv for just plain rgb
        # vertical inversion corrects the coordinate system mismatch between rasteriser & standard screen coord system

        cv2.imshow("Complementary filter: gyro, accel & magnetometer", frame) # flash up the bunny from frame buffer
        if cv2.waitKey(1) & 0xFF == ord('q'): # advance frames automatically; q to quit early
            break

    print("\nAll done, press anything now to close off the bunny.")
    cv2.waitKey(0) # now keep it up on screen until user interrupt
    cv2.destroyAllWindows()