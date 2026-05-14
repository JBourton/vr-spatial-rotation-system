# Note to examiner about this file:
# This file calculates and prints each and every metric cited in the report in the order they appear.
# I created this because I felt that adding these to the main render.py was too hectic and complicated my submission, so to seperate the report analysis complexity from the actual engine I decided this was necessary.
# If you want to run it, it needs to be ran from the coursework engine directory, ensuring IMUData.csv and bunny.obj and in the same folder.


# Firstly, all the imports I need to run this
import math
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


# Now all of the functions I'm just using from render.py, but placing in here as it's just simpler than exporting and importing them.
# All I've done is copy-pasted the functions I wrote from the engine, which is the main reason why this file is so big; 
# if you scroll straight down to line ~530 you can see where I've started my actual report code



# the following function is a custom reader for loading in the IMUData.csv file
def readin_imu_dataset(imu_data_filename="IMUData.csv"): # I added this as the deafult filepath but it can be flexibly changed with new imu csv data here
    imu_dataset=  pd.read_csv(imu_data_filename, skipinitialspace=True) # (the file seemed to have a leading whitespace, so this operation was added to ditch it)
    imu_dataset.columns= imu_dataset.columns.str.strip()
    
    # as required by cousrework q2.1.i, this takes the rotational rates from deg/s to rad/s
    gyroscope_cols_in_rads_to_convert= ['gyroscope.X', 'gyroscope.Y', 'gyroscope.Z']
    imu_dataset[gyroscope_cols_in_rads_to_convert]= np.deg2rad(imu_dataset[gyroscope_cols_in_rads_to_convert])

    return imu_dataset


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
    w= cos_roll*cos_pitch*cos_yaw + sin_roll*sin_pitch*sin_yaw
    x= sin_roll*cos_pitch*cos_yaw - cos_roll*sin_pitch*sin_yaw
    y= cos_roll*sin_pitch*cos_yaw + sin_roll*cos_pitch*sin_yaw
    z= cos_roll*cos_pitch*sin_yaw - sin_roll*sin_pitch*cos_yaw
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

# here I'm taking the initial magnetometer reading as a reference for yaw correction, which I capture at frame 0 before rotation occurs
# (i.e. it represents magnetic north in the sensor's starting orientation)
def capture_magnetometer_reference(imu_dataframe, hard_iron_x=0.0, hard_iron_y=0.0, hard_iron_z=0.0, soft_iron_scale_x=1.0, soft_iron_scale_y=1.0, soft_iron_scale_z=1.0): # these defaults are getting overwritten anyway at function call, I just set them as this value initialy
    # just grabbing the very first magnetometer reading from the dataset
    # this is my source of truth magnetic north from the get-go of the animation
    x_ref, y_ref, z_ref= imu_dataframe['magnetometer.X'][0], imu_dataframe['magnetometer.Y'][0], imu_dataframe['magnetometer.Z'][0]

    # I'm applying the same hard-iron & soft-iron corrections as are applied to every subsequent reading
    # so the reference lives in the same calibrated space as the readings it's compared against
    x_ref= (x_ref-hard_iron_x) * soft_iron_scale_x
    y_ref= (y_ref-hard_iron_y) * soft_iron_scale_y
    z_ref= (z_ref-hard_iron_z) * soft_iron_scale_z

    # i'm normalising so magnitude differences don't affect angle calc later
    ref_magnitude= math.sqrt(x_ref**2 + y_ref**2 + z_ref**2)
    x_ref /= ref_magnitude
    y_ref /= ref_magnitude
    z_ref /= ref_magnitude

    return x_ref, y_ref, z_ref


# here I'm computing hard-iron & soft-iron calibration offsets from full dataset.
# my understanding of hard-iron bias is that it's a constant additive offset caused by ferromagnetic materials physically attached to sensor (e.g. the circuit board or battery) 
# and that bias adds a fixed vector to every reading regardless of orientation.
# The coursework LaValle book points that out as "materials that serve as a source of magnetic fields are called hard iron"
# my soft-iron bias understanding is that nearby materials distort earth's field as it passes through them which stretches the ideal sphere of readings into an elipsis-like shape w/ unequal axis lengths.
# LaValle also talks about this, saying "other materials distort magnetic fields that pass through them; these are called soft iron"
# thus my fix is created from the imu data given: over the full 27s sequence the headset rotates through all 3 axes which gives a pretty good spread spread of field directions to fit against
def calibrate_magnetometer(imu_dataframe):
    # I find my hard-iron correction by taking the midpoint of the observed range to find the bias per axis
    # (i.e. the centre of the bounding box of all readings - a standard first-order approximation)
    x_axis_hard_iron= (imu_dataframe['magnetometer.X'].max() + imu_dataframe['magnetometer.X'].min())/2.0
    y_axis_hard_iron= (imu_dataframe['magnetometer.Y'].max() + imu_dataframe['magnetometer.Y'].min())/2.0
    z_axis_hard_iron= (imu_dataframe['magnetometer.Z'].max() + imu_dataframe['magnetometer.Z'].min())/2.0

    # and then my soft-iron correction, which after subtracting hard-iron bias, the 1/2 range per axis should be equal
    # & if it isn't, the field sphere has been squished down into an ellipsis-like shape by soft-iron distortion
    # Here I'm normalising each axis by its own half-range so they all end up with the same unit-sphere scale
    x_axis_half_range= (imu_dataframe['magnetometer.X'].max() - imu_dataframe['magnetometer.X'].min())/2.0
    y_axis_half_range= (imu_dataframe['magnetometer.Y'].max() - imu_dataframe['magnetometer.Y'].min())/2.0
    z_axis_half_range= (imu_dataframe['magnetometer.Z'].max() - imu_dataframe['magnetometer.Z'].min())/2.0

    # I compute the average half-range so the scale keeps pretty comparable to field's original strength
    avg_half_range= (x_axis_half_range + y_axis_half_range + z_axis_half_range) / 3.0

    # to get the soft-iron scale factors I divide by 1/2 range and multipl by avg_half_range, which then rescales each axis
    # to match avg, making the ellipsis-like shape spherical without changing the overall magnitude much
    # + I have a guard against zero half-range (in case an axis doesn't actually have any variation at all)
    soft_iron_scale_x= avg_half_range/x_axis_half_range if x_axis_half_range > 0.00000001 else 1.0
    soft_iron_scale_y= avg_half_range/y_axis_half_range if y_axis_half_range > 0.00000001 else 1.0
    soft_iron_scale_z= avg_half_range/z_axis_half_range if z_axis_half_range > 0.00000001 else 1.0

    return x_axis_hard_iron, y_axis_hard_iron, z_axis_hard_iron, soft_iron_scale_x, soft_iron_scale_y, soft_iron_scale_z


# here I'm calculating the yaw correction axis & angle by comparing current mag reading to the reference one
# theory: mirrors calculate_tilt_axis_and_angle but projected onto the horizontal plane only
# the point being is that I'm only correcting yaw & not disturbing the roll/pitch already fixed by accelerometer
#
# note: this function is recieving the mag vector ALREADY rotated to global frame.
# I concluded this was the the correct approach because when the quaternion estimate is accurate, the global-frame mag vector==the world-frame field direction
# whatever sensor orientation actually is, so real rotation gives 0 yaw error and only gyroscope drift produces a nonzero signal. 
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
def full_fusion_orientation_calc(imu_dataframe, accel_alpha_tuner=0.02,mag_alpha_tuner=0.02, hard_iron_x=0.0, hard_iron_y=0.0, hard_iron_z=0.0, soft_iron_scale_x=1.0, soft_iron_scale_y=1.0, soft_iron_scale_z=1.0): # (note again, these hard & soft iron values are just placeholder that actually get overwritten)
    # here I'm fetching the val of reference magnetometer & reading at frame 0 prior to any of the rotation movemnents starting to occur
    # note: the reference is captured *after* applying calibration so it's in the same corrected space as all subsequent readings
    ref_mag_x, ref_mag_y, ref_mag_z= capture_magnetometer_reference(
        imu_dataframe, hard_iron_x, hard_iron_y, hard_iron_z,
        soft_iron_scale_x, soft_iron_scale_y, soft_iron_scale_z)

    # this is my initial identity quaternion, same as before
    id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z= 1.0, 0.0, 0.0, 0.0

    # tracking array, again same pattern as my previous filters
    orientation_history_container = [(id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z)]

    # here I'm initialising the smoothed accel & mag values at rest assumption
    smoothed_accel_x, smoothed_accel_y, smoothed_accel_z = 0.0, 0.0, 1.0
    smoothed_mag_x, smoothed_mag_y, smoothed_mag_z= ref_mag_x, ref_mag_y, ref_mag_z
    accel_smooth_factor_for_ema= 0.1
    mag_smooth_factor_for_ema= 0.1

    for i in range(1, len(imu_dataframe)):
        # 1st, elapsed time between readings (same as always)
        elapsed_sensor_time_change= imu_dataframe['time'][i] - imu_dataframe['time'][i-1]

        # 2ndly gyroscope angular displacement for each axis
        change_over_roll= imu_dataframe['gyroscope.X'][i] * elapsed_sensor_time_change
        change_over_pitch= imu_dataframe['gyroscope.Y'][i] * elapsed_sensor_time_change
        change_over_yaw= imu_dataframe['gyroscope.Z'][i] * elapsed_sensor_time_change

        # 3rdly I'm dead reckoning the gyroscope forward one step
        change_over_w, change_over_x, change_over_y, change_over_z= convert_euler_to_quaternion(change_over_roll, change_over_pitch, change_over_yaw)
        gyro_w, gyro_x, gyro_y, gyro_z= calculate_quaternion_product(change_over_w, change_over_x, change_over_y, change_over_z, id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z)

        # 4th, normalise gyro estimate
        normed_gyro_estimate= math.sqrt(gyro_w**2 + gyro_x**2 + gyro_y**2 + gyro_z**2)
        gyro_w, gyro_x, gyro_y, gyro_z= gyro_w/normed_gyro_estimate, gyro_x/normed_gyro_estimate, gyro_y/normed_gyro_estimate, gyro_z/normed_gyro_estimate

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

        accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z= calculate_quaternion_product(accel_corr_w, accel_corr_x, accel_corr_y, accel_corr_z,gyro_w, gyro_x, gyro_y, gyro_z)

        # 9th. and now for the magnometer correction (new stuff), that's responsible for correcting the yaw (which I didn't correct before)
        # here I'm smoothing the raw magnetometer reading to reduce noise, because the magnometer is noisy inherently
        raw_mag_val_x, raw_mag_val_y, raw_mag_val_z= imu_dataframe['magnetometer.X'][i], imu_dataframe['magnetometer.Y'][i], imu_dataframe['magnetometer.Z'][i]

        # 9a. per one of my coursework 'extra' fixes, I apply hard-iron correction. Here I'm subtracting the constant bias offset I calculated from the full dataset
        # which removes that fixed vector contribution from ferromagnetic materials attached to the sensor I described before.
        # 9b. and another one them, soft-iron correction, which as described is rescaling each axis by its scale factor to turn distorted ellipsoid back into sphere
        # I apply both corrections before EMA smoothing so the filter never sees the raw biased signal
        raw_mag_val_x= (raw_mag_val_x-hard_iron_x) * soft_iron_scale_x
        raw_mag_val_y= (raw_mag_val_y-hard_iron_y) * soft_iron_scale_y
        raw_mag_val_z= (raw_mag_val_z-hard_iron_z) * soft_iron_scale_z

        smoothed_mag_x= mag_smooth_factor_for_ema*raw_mag_val_x + (1-mag_smooth_factor_for_ema)*smoothed_mag_x # (dev note: I would normally put these all on one line but it's going to be a tad messy with this smoothing formula)
        smoothed_mag_y= mag_smooth_factor_for_ema*raw_mag_val_y + (1-mag_smooth_factor_for_ema)*smoothed_mag_y
        smoothed_mag_z= mag_smooth_factor_for_ema*raw_mag_val_z + (1-mag_smooth_factor_for_ema)*smoothed_mag_z

        # 10th I'm rotating magnetometer reading-->global frame using accel-fused estimate
        global_mag_x, global_mag_y, global_mag_z= transform_vector_to_global_frame(accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z, # (p.s. I'm using the accellerometer-fused estimate instead of just gyro alone because it's more accurate i.e. it's already been corrected for roll & pitch)
            smoothed_mag_x, smoothed_mag_y, smoothed_mag_z)

        # 11th I'm calculating yaw correction axis and angle vs reference
        yaw_axis_x, yaw_axis_y, yaw_axis_z, yaw_phi= calculate_yaw_axis_and_angle( # this is projecting both current mag vector & reference one onto horizontal plane so that vertical component of magnetic field doesn't contaminate yaw estimate
            global_mag_x, global_mag_y, global_mag_z, ref_mag_x, ref_mag_y, ref_mag_z)

        # 12th as described in my coursework writeup, this is my dual reliability gate before I apply any magnetometer correction.
        
        # The 1st gate is the gyroscope rate gate for suppressing correction during active rotation.
        # The idea behind this is, during fast rotation the EMA-smoothed mag gets caught up lagging behind the real field direction so any quaternion drift that's currently present corrupts the
        # global-frame projection. The reasoning behind why I chose the current value can be found in my hyperparmas desription list above.
        total_gyro_rate= math.sqrt(imu_dataframe['gyroscope.X'][i]**2 + imu_dataframe['gyroscope.Y'][i]**2 + imu_dataframe['gyroscope.Z'][i]**2)
        is_gyro_currently_too_slow= total_gyro_rate < 0.05

        # The 2nd gate is the normal accelerometer tilt gate, as before.
        angle_for_accel_tilt_gate = math.acos(max(-1.0, min(1.0,smoothed_accel_z / max(1e-8, math.sqrt(smoothed_accel_x**2 + smoothed_accel_y**2 + smoothed_accel_z**2)))))
        sensor_is_level = angle_for_accel_tilt_gate < math.radians(20)  # within 20 degrees of upright as learnt from textbook
        
        # 13th I'm building & applying magnetometer yaw correction quaternion (pretty much same as I did for acceleromter correction)
        mag_is_reliable = is_gyro_currently_too_slow and sensor_is_level
        mag_correction_angle= (mag_alpha_tuner*yaw_phi) if mag_is_reliable else 0.0 # 12.i I'm scaling yaw error by the mag_alpha_tuner param so only a little bit of the total error is being nudged each frame, & therefore the correction doesn't instantly get snapped too instanteneously
        correction_for_mag_w= math.cos(mag_correction_angle/2.0)
        correction_for_mag_x= yaw_axis_x*math.sin(mag_correction_angle/2.0)
        correction_for_mag_y= yaw_axis_y*math.sin(mag_correction_angle/2.0)
        correction_for_mag_z= yaw_axis_z*math.sin(mag_correction_angle/2.0)

        id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z= calculate_quaternion_product( # 12ii. then I convert scaled angle-->quaternion w/ axis-angle formula  & pre-multiply onto accel-fused estimate
            correction_for_mag_w, correction_for_mag_x, correction_for_mag_y, correction_for_mag_z, accel_fused_w, accel_fused_x, accel_fused_y, accel_fused_z)

        # 14th, I normalise the final fused result before storing
        fused_norm= math.sqrt(id_ref_for_w**2 + id_ref_for_x**2 + id_ref_for_y**2 + id_ref_for_z**2)
        id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z= id_ref_for_w/fused_norm, id_ref_for_x/fused_norm, id_ref_for_y/fused_norm, id_ref_for_z/fused_norm

        # & now I'm done for this step, so I store and & move on
        orientation_history_container.append((id_ref_for_w, id_ref_for_x, id_ref_for_y, id_ref_for_z))

    return orientation_history_container







# Now that I've gone and copy-pasted all my old functions, I'm next reading in the IMU data
print("reading in all the imu data...")
imu_data= readin_imu_dataset()
all_timestamps_lst= list(imu_data['time'])
total_samples= len(imu_data)
given_imu_smple_rate_hz= 256 # the coursework report stated as such
time_budget_per_sample_us= (1.0 / given_imu_smple_rate_hz) * 1000000
print(f"samples overall are: {total_samples}, and sample rate is: {given_imu_smple_rate_hz} Hz")



# I've decided to add the alpha investigation charts to this file as they are shown in the report, so I want to ensure full transparency is presnt on how I create them.

# First up is the accelerometer-fused corections, which is shows the effect of alpha on roll and pitch drift correction
# Here I'm running the complementary filter at several alpha values and dead reckoning as a baseline, then plotting roll and pitch over time
# Note that I save the results to static_renders/alpha_investigation.png (for if you want to run this and find it yourself to verify alignment w/ my paper)
comp_filt_alphas_for_testing= [0.0005, 0.001, 0.01, 0.02, 0.1]
results_of_different_alphas= {}
for test_alpha in comp_filt_alphas_for_testing:
    print(f"Executing the complementary filter w/ alpha={test_alpha}...")
    results_of_different_alphas[test_alpha]= complementary_filter_orientation_calc(imu_data, alpha=test_alpha)

# & also run dead reckoning once for my baseline drift comparison
print("Now executing my dead reckoning for baseline...")
dead_reckoning_results= dead_reckoning_orientation_calc_implementation(imu_data)


# now I go ahead and produce my visualisation of roll & pitch over imu seq. for each alpha
# (yaw excluded - accelerometer cannot correct yaw, so alpha has no meaningful effect on it)
all_the_timestamps= list(imu_data['time'])
comp_filt_fig, comp_filt_axes= plt.subplots(2, 1, figsize=(14, 8), sharex=True)
colours = ["#a34848", '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'] # I'm going for the rainbow order as it gives some semblance of pattern amongst the increasing alphas

for comp_filt_axis, idx_of_angle, name_of_angle in [(comp_filt_axes[0], 0, 'Roll (deg)'), (comp_filt_axes[1], 1, 'Pitch (deg)')]:
    # 1st to plot is my dead reckoning baseline (dashed black)
    dead_reck_plotting_angles = [math.degrees(convert_quaternion_to_euler(*dead_reckoning_results[dead_reck_res])[idx_of_angle]) for dead_reck_res in range(len(dead_reckoning_results))]
    comp_filt_axis.plot(all_the_timestamps, dead_reck_plotting_angles, color='black', linewidth=1.2, linestyle='--', label='dead reckoning (no correction)', alpha=0.7)

    # 1 line per trial comp filt. alpha val
    for colour, (test_alpha, orientations) in zip(colours, results_of_different_alphas.items()):
        angles = [math.degrees(convert_quaternion_to_euler(*orientations[orient])[idx_of_angle]) for orient in range(len(orientations))]
        comp_filt_axis.plot(all_the_timestamps, angles, color=colour, linewidth=1.0,label=f'alpha={test_alpha}', alpha=0.85)

    comp_filt_axis.set_ylabel(name_of_angle, fontsize=10)
    comp_filt_axis.grid(True, alpha=0.3)
    comp_filt_axis.legend(loc='upper right', fontsize=12, ncol=2)
comp_filt_axes[1].set_xlabel("Time (seconds)", fontsize=10)

plt.tight_layout()
os.makedirs("static_renders", exist_ok=True) # I'm just putting them in the same folder as my static renders, serving as a kind of master directory for analysis files
plt.savefig("static_renders/alpha_investigation.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved alpha investigation plot to static_renders/alpha_investigation.png")



# Now I do the magnetometer alpha investigation (which I think should be called beta but for consistency of the report I'm sticking with alpha)
# this is exactly the same structure as the accel alpha graph above but for yaw.
# Here I'm running the full fusion filter at several of the same mag_alpha values with accel_alpha fixed at 0.02, & using the accel-only complementary filter as the baseline 
# (as that's what adding the magnetometer is supposed to improve on). Yaw only is plotted as that's what the mag is correcting.
# I use the same output files structure, saving to static_renders/mag_alpha_investigation.png

# now I compute magnetometer calibration 1x from the full dataset before any filter runs
# I only need to do that once as it's coming from the static data stats
print("Now computing magnetometer hard-iron & soft-iron calibration from dataset...")
hardiron_x_calib, hardiron_y_calib, hardiron_z_calib, softiron_x_calib, softiron_y_calib, softiron_z_calib= calibrate_magnetometer(imu_data)

# now my experiment of different mag_alpha values for the report (q4)
# alpha values are consistent w/ the accelerometer investigation (q3) for direct comparison
mag_alpha_test_vals = [0.0005, 0.001, 0.01, 0.02, 0.1]
mag_alpha_testing_result_log = {}

# accel-only comp filter as baseline (gyro + accel, no magn)
# I'm checking whether adding the magnetometer improves on the already-tilt-corrected estimate, not just on raw dead reckoning as I don't really infer that's required.
print("Next, computing accel-only baseline for mag alpha graph...")
accel_only_baseline = complementary_filter_orientation_calc(imu_data, alpha=0.02)

for instance_of_mag_alpha in mag_alpha_test_vals:
    print(f"executing 1x full fusion test w/ mag_alpha={instance_of_mag_alpha}...")
    mag_alpha_testing_result_log[instance_of_mag_alpha] = full_fusion_orientation_calc(
        imu_data, accel_alpha_tuner=0.02, mag_alpha_tuner=instance_of_mag_alpha,
        hard_iron_x=hardiron_x_calib, hard_iron_y=hardiron_y_calib, hard_iron_z=hardiron_z_calib,
        soft_iron_scale_x=softiron_x_calib, soft_iron_scale_y=softiron_y_calib, soft_iron_scale_z=softiron_z_calib)

# now I build the yaw investigation plot (yaw only, since that's what the magnetometer corrects), mirroring q3's roll/pitch plot but for yaw
colours = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # rainbow-y red, orange, green, purple, blue (same as q3)

# extract yaw from accel-only baseline
accel_only_yaw_vals= []
for (w_quat, x_quat, y_quat, z_quat) in accel_only_baseline:
    _, _, yaw_val= convert_quaternion_to_euler(w_quat, x_quat, y_quat, z_quat) # (only yaw)
    accel_only_yaw_vals.append(math.degrees(yaw_val))

comp_filt_fig, comp_filt_axis= plt.subplots(figsize=(14, 5))

# accel-only is my new dashed grey baseline (same styling as q3 graph)
comp_filt_axis.plot(all_the_timestamps, accel_only_yaw_vals, color='#7f7f7f', linestyle='--', linewidth=1.2, label='accel-only complementary filter (no mag)', zorder=5)

# now i plot each mag_alpha variant
for colour, instance_of_mag_alpha in zip(colours, mag_alpha_test_vals):
    orientations= mag_alpha_testing_result_log[instance_of_mag_alpha]
    vals_of_magcorrected_yaw= []
    for (w_quat, x_quat, y_quat, z_quat) in orientations:
        _, _, yaw_val = convert_quaternion_to_euler(w_quat, x_quat, y_quat, z_quat)
        vals_of_magcorrected_yaw.append(math.degrees(yaw_val))
    comp_filt_axis.plot(all_the_timestamps, vals_of_magcorrected_yaw, color=colour, linewidth=0.9, label=f'mag_alpha={instance_of_mag_alpha}')

# then all the labels and that's done.
comp_filt_axis.set_xlabel('Time (seconds)')
comp_filt_axis.set_ylabel('Yaw (deg)')
comp_filt_axis.legend(loc='upper right', fontsize=10)
comp_filt_axis.grid(True, alpha=0.3)

comp_filt_fig.tight_layout()
comp_filt_fig.savefig('static_renders/mag_alpha_investigation.png', dpi=150)
plt.close(comp_filt_fig)
print("mag alpha investigation graph saved to static_renders/mag_alpha_investigation.png") # where you can find this file if you want to check it out



# Now I'm running both filters & converting output to euler degrees
print("first up, running dead reckoning filter is getting ran...")
dead_reckoning_orientations = dead_reckoning_orientation_calc_implementation(imu_data)
print("next, it's the complementary filter (at alpha=0.02) that gets ran...")
complementary_filter_orientations = complementary_filter_orientation_calc(imu_data, alpha=0.02)

# here I'm converting each quaternion to euler angles in degrees for rms calculation below
dead_reckoning_euler_deg = [tuple(math.degrees(angle) for angle in convert_quaternion_to_euler(quat_w, quat_x, quat_y, quat_z))
    for (quat_w, quat_x, quat_y, quat_z) in dead_reckoning_orientations]
complementary_filter_euler_deg = [tuple(math.degrees(angle) for angle in convert_quaternion_to_euler(quat_w, quat_x, quat_y, quat_z))
    for (quat_w, quat_x, quat_y, quat_z) in complementary_filter_orientations]

# this part is the rms error during the final rest period, which I chose as I thought it was the most revealing indicator of the effect of accelerometer-based correction comapred to gyro alone
# in the report this is the table 1 data.
# in theory, both roll and pitch should be 0 degrees here the rms error is relative to that

# there are the final rest period boundaries that I defined in the report
BEGGINING_OF_REST_PERIOD_s= 23.0 #(approx times)
END_OF_REST_PERIOD_s= 27.5
indices_of_rest_frames= [rest_i for rest_i, time in enumerate(all_timestamps_lst) if BEGGINING_OF_REST_PERIOD_s <= time <= END_OF_REST_PERIOD_s]

def calculate_the_rms_from_zero_point(euler_series_deg, axis_index, frame_indices):
    # logic is: rms of angle values over a window where target is 0 deg so error === the actual value itself
    # axis_index is the usual I've used in this coursework: 0=roll, 1=pitch, 2=yaw
    squared_sum= sum(euler_series_deg[indicy][axis_index]**2 for indicy in frame_indices)
    return math.sqrt(squared_sum / len(frame_indices))

# these are all the values I show in the table in my report
dead_reckoning_rms_deg= calculate_the_rms_from_zero_point(dead_reckoning_euler_deg, 0, indices_of_rest_frames)
dead_reckoning_rms_deg= calculate_the_rms_from_zero_point(dead_reckoning_euler_deg, 1, indices_of_rest_frames)
complementary_filter_rms_deg= calculate_the_rms_from_zero_point(complementary_filter_euler_deg, 0, indices_of_rest_frames)
complementary_filter_pitch_rms_deg= calculate_the_rms_from_zero_point(complementary_filter_euler_deg, 1, indices_of_rest_frames)

print()
print("Data for table 1's report: rms error from 0 deg (for final rest period ~23-27.5s)")
print()
print(f"{'Filter':<37} {'Roll RMS':>9} {'Pitch RMS':>9}") # this is my table's heading row, off set to show the values comfortably
print(f"{'Dead reckoning (just gyroscope)':<37} {dead_reckoning_rms_deg:>8.1f}  {dead_reckoning_rms_deg:>8.1f}")
print(f"{'And complementary filter (alpha=0.02)':<37} {complementary_filter_rms_deg:>8.1f}  {complementary_filter_pitch_rms_deg:>8.1f}")
print()






# Next stage of generating analysis data for my report.
# This part specifically is for table 2 in the report, where I calculate the arithmetic operations counts per sample.
# My technique for this was tracing each function call in the 2 filters I defined already above.
# It was quite complicated to try to figure out the overall operations, which is why I've added in a step-by-step guide to everything below so hopefully you can read through and follow that fine.

# In this analysis, I count 4 operation categories:
#   mul== multiplications & divisions (most expensive, and also the same cost as they're just the same)
#   add == additions & subtractions (also same)
#   trig== sin, cos, acos calls (which I think are a lot more expensive than a multiplcation would ever be)
#   sqrt== square root, which I think is still quite expensive relative to mutliplying

# my strategy for getting this table data is: I name a cost variable for each function/sub-step in the filter,
# I get its value by reading the formula in that function above in this file (which are the ones from engine, just copy-pasted in here) & add them all together at the end.
# this makes it pretty easy to check, as each named variable has only 1 job and 1 formula.


# 1st step is getting cost of each individual function/operation used by the filters
# (which are all derived just through by reading the formula in the matching function above)

# first function I'm looking at is convert_euler_to_quaternion (which is called by both filters)
# looking at the function above: first it 1/2s all the 3 input angles (3 mul),
# then calls cos & sin on each 1/2-angle (so 6 trig as 3 cos + 3 sin),
# then calculates w, x, y, z each as a chain of 3 multiplied terms added/subtracted together:
#  e.g. w = cos_roll*cos_pitch*cos_yaw + sin_roll*sin_pitch*sin_yaw
#  that's 2 chains of 3 values== 4 muls (because two chained pairs) & 1 add per component
#  x4 components gives 16 mul + 4 add
euler_to_quat_mult_total=  3 + 16 # 3 half-angle muls + 4 components x 4 muls each
euler_to_quat_add_total=  4 # 1 + or - per output component (w, x, y, z)
euler_to_quat_trig_total= 6 # 3 cos + 3 sin

# next up is my calculate_quaternion_product function (i.e. the Hamilton product called several times)
# looking at the function above, there's 4 output components, each w/ 4 values multiplied & 3 add/subs:
#  e.g. combined_w= w1*w2 - x1*x2 - y1*y2 - z1*z2 (so 4 mul, 3 sub)
# x4 components then gives me:
quat_prod_mult_total= 4 * 4    # 4 components x 4 multiplications each
quat_prod_add_total= 4 * 3    # 4 components x 3 additions/subtractions each

# then normalising quaternion (which appears twice in my comp filter, 1x for gyro & 1x for fused result)
# normalising invovles squaring each component & summing (so 4 mul + 3 add), 1x sqrt & then divide each by norm (4 mult)
normalise_mult_total=  4 + 4 # 4 squarings + 4 mults
normalise_add_total=  3 # 3 additions to sum the squares
normalise_sqrt_total= 1

# now EMA low-pass filter on one axis (which is step 5a above, its applied to 3 accel axes independently)
# the formula is: lp= smooth_factor * raw_val + (1.0 - smooth_factor) * lp_prev
# so that's 2 multiplications & 2 additions (the subtraction in 1.0-smooth counts as add as expalined):
ema_one_axis_mult_total= 2
ema_one_axis_add_total= 2

# & magnitude gate (step 5b)
# my formula is sqrt(lp_ax^2 + lp_ay^2 + lp_az^2) which is just 3 squarings (=3 mul), 2 additions to sum them, 1 sqrt:
magnitude_gate_mult_total=  3
magnitude_gate_add_total=  2
magnitude_gate_sqrt_total= 1

# then transform_vector_to_global_frame (step 6 above, where I'm rotating my accel reading to the world space)
# that function is calling convert_quaternion_to_conjugate (which is just 3 negations== 3 add)
# & then calls calculate_quaternion_product 2x (sandwich product: q*p* q_conj):
global_frame_mult_total= 2 * quat_prod_mult_total # 2 hamilton products
global_frame_add_total= 3 + (2 * quat_prod_add_total) # 3 negations for conjugate + 2 hamilton products

# then calculate_tilt_axis_and_angle (step 7), which is a function that does 4 sub-operations one after another:
# 1st normalising accel vector (which is just the same cost as quaternion normalising I belive)
# 2nd cross producting normalised_accel x world_up: each of 3 output components = 2 mul + 1 sub, giving 6 mul, 3 add
# 3rd normalising the resulting tilt axis (again, it's just same cost)
# 4th is dot product of the two vectors: 3 mul + 2 add, then acos (1x trig)
tilt_axis_mult_total= normalise_mult_total + 6 + normalise_mult_total + 3 # normalise + cross + normalise + dot
tilt_axis_add_total= normalise_add_total + 3 + normalise_add_total + 2 # normalise + cross + normalise + dot
tilt_axis_trig_total= 1 # the acos at the end
tilt_axis_sqrt_total= 2 # one sqrt inside each of the two normalises

# now I calculate operations from buidling correction quaternion from axis-angle (step 8)
# so, correction_angle= alpha * phi_angle, just 1 mult
# then correction_w = cos(angle/2.0), 1 div (/2), 1 trig (cos)
# then correction_x = axis_x * sin(angle/2.0), 1 div (/2), 1 trig (sin), 1 mul (axis*sin)
# correction_y and z repeat the same, 2 more div, 2 more trig, 2 more mul
build_corr_quat_mult_total=  1 + 4 + 3 # alpha*phi + 4 separate /2 divisions +3 axis*sin muls
build_corr_quat_trig_total= 1 + 3 # 1 cos  +  3 sin (I do 1 separate call per component)



# Now I add up all the total calls for just dead reckoning alone
# those dead reckoning calls in order are:
#   gyro*dt on 3 axes== 3 mul
#   convert_euler_to_quaternion== euler_to_quat_mul, euler_to_quat_add, euler_to_quat_trig
#   calculate_quaternion_product (accumulate)== qprod_mul, qprod_add
#   BUT, I by have no normalise step here (hence why drift accumulates, but I discuss all that in my report)
total_mults_for_dead_reck_per_sample= 3 + euler_to_quat_mult_total + quat_prod_mult_total
total_adds_for_dead_reck_per_sample= euler_to_quat_add_total + quat_prod_add_total
total_trigs_for_dead_reck_per_sample= euler_to_quat_trig_total
total_sqrts_for_dead_reck_per_sample= 0


# then also total up what all the additional operations are on top for complementary filter (it uses all that dead reckoning does, then does stuff on top)
# going by the labls I've used in my functions above:
# - step 4: normalising gyro estimate: normalise_*
# - step 5a: EMA low-pass on 3 accel axes: ema_one_axis_* x3
# - step 5b: magnitude gate (i.e. the accel being about 1g check): magnitude_gate_*
# - step 6: rotate accel to global frame: global_frame_*
# - step 7: calculate tilt axis and angle: tilt_axis_*
# - step 8: build correction quaternion: build_corr_quat_*
# - step 9: apply correction (another qprod): qprod_*
# - step 10: normalising fused result: normalise_*

# I map what I wrote above to this overall calculation here so hopefully it's easy enough to follow
extra_mul_per_sample=  (normalise_mult_total + # step 4: normalise gyro
    (ema_one_axis_mult_total * 3) + # step 5a: ema on x, y, z accel axes
    magnitude_gate_mult_total + # step 5b: magnitude gate
    global_frame_mult_total + # step 6: rotate accel to world frame
    tilt_axis_mult_total + # step 7: tilt axis and phi angle
    build_corr_quat_mult_total + # step 8: build correction quaternion
    quat_prod_mult_total + # step 9: apply correction (hamilton product)
    normalise_mult_total) # step 10: normalise final fused result

extra_add_per_sample=  (normalise_add_total +
    (ema_one_axis_add_total * 3) +
    magnitude_gate_add_total +
    global_frame_add_total +
    tilt_axis_add_total +
    quat_prod_add_total +
    normalise_add_total)

extra_trig_per_sample= tilt_axis_trig_total + build_corr_quat_trig_total
extra_sqrt_per_sample= (normalise_sqrt_total + # gyro normalise
    magnitude_gate_sqrt_total + # my magnitude gate
    tilt_axis_sqrt_total + # tilt axis function (2 normalises inside it)
    normalise_sqrt_total) # the final fused normalise

overall_comp_filt_mults_per_sample=  total_mults_for_dead_reck_per_sample + extra_mul_per_sample
overall_comp_filt_adds_per_sample=  total_adds_for_dead_reck_per_sample  + extra_add_per_sample
overall_comp_filt_trig_per_sample= total_trigs_for_dead_reck_per_sample + extra_trig_per_sample
overall_comp_filt_sqrt_per_sample= total_sqrts_for_dead_reck_per_sample + extra_sqrt_per_sample

overall_dead_reck_mults= total_mults_for_dead_reck_per_sample * total_samples
overall_comp_filt_mults= overall_comp_filt_mults_per_sample * total_samples

print()
print("This is the data from the report showing my calculated overall operations for dead reckoning vs complementary filter:")
print()
print(f"{'Filter':<42} {'Mul':>4} {'Add':>4} {'Trig':>4} {'Sqrt':>4}") # again I've offset it here for the dead header row
print()
print(f"{'Dead reckoning':<42} {total_mults_for_dead_reck_per_sample:>4} {total_adds_for_dead_reck_per_sample:>4} {total_trigs_for_dead_reck_per_sample:>4} {total_sqrts_for_dead_reck_per_sample:>4}")
print(f"{'Complementary filter (additional)':<42} {extra_mul_per_sample:>4} {extra_add_per_sample:>4} {extra_trig_per_sample:>4} {extra_sqrt_per_sample:>4}")
print(f"{'Complementary filter (total)':<42} {overall_comp_filt_mults_per_sample:>4} {overall_comp_filt_adds_per_sample:>4} {overall_comp_filt_trig_per_sample:>4} {overall_comp_filt_sqrt_per_sample:>4}")
print()
print(f"total multiply operations over {total_samples} samples:")
print(f"a. dead reckoning: ~{overall_dead_reck_mults:,}")
print(f"b. complementary filter: ~{overall_comp_filt_mults:,}")
print(f"so overall, complementary filter is sitting at {overall_comp_filt_mults_per_sample/total_mults_for_dead_reck_per_sample:.1f}x more multiply-intensive per sample according to my calculations.")




# Now to pair with all the calculation I did above, I now measure the overall time consumed by both filters, which I've done over 10 runs for stability's sake
OVERALL_RUN_REPEAT_COUNT = 10
print(f"\nOk, next part of the analysis: I'm timing each filter {OVERALL_RUN_REPEAT_COUNT}x for a stable measurement...")

dead_reckoning_run_times_holder = []
for _ in range(OVERALL_RUN_REPEAT_COUNT):
    initial_time = time.perf_counter()
    dead_reckoning_orientation_calc_implementation(imu_data)
    dead_reckoning_run_times_holder.append(time.perf_counter() - initial_time)

comp_filter_run_times_holder = []
for _ in range(OVERALL_RUN_REPEAT_COUNT):
    initial_time = time.perf_counter()
    complementary_filter_orientation_calc(imu_data, alpha=0.02)
    comp_filter_run_times_holder.append(time.perf_counter() - initial_time)

# now I calculate the mean time overall & per-sample for display
dead_reck_mean_time_in_ms= (sum(dead_reckoning_run_times_holder)/OVERALL_RUN_REPEAT_COUNT) * 1000
comp_filter_mean_time_in_ms= (sum(comp_filter_run_times_holder)/OVERALL_RUN_REPEAT_COUNT) * 1000
dead_reckoning_microS_per_sample= (dead_reck_mean_time_in_ms/total_samples) * 1000
comp_filter_microS_per_sample= (comp_filter_mean_time_in_ms/total_samples) * 1000
overall_timing_ratio= comp_filter_mean_time_in_ms/dead_reck_mean_time_in_ms
comp_filter_budget_percent= (comp_filter_microS_per_sample/time_budget_per_sample_us) * 100

print()
print("And now for the timing results in my report (dead reckoning vs complementary filter):")
print(f"firs up, dead reckoning: {dead_reck_mean_time_in_ms:.1f} ms total and ({dead_reckoning_microS_per_sample:.1f} us/sample)")
print(f"complementary filter: {comp_filter_mean_time_in_ms:.1f} ms total and ({comp_filter_microS_per_sample:.1f} us/sample)")
print(f"timing ratio (comp filt to dead reck): {overall_timing_ratio:.2f}x")
print(f"The time budget per sample at {given_imu_smple_rate_hz} Hz: {time_budget_per_sample_us:.0f} us")
print(f"complementary filter uses {comp_filter_microS_per_sample:.1f} us ({comp_filter_budget_percent:.1f}% of budget)")
print(f"budget remaining after complementary filter: {100-comp_filter_budget_percent:.0f}% unused")
print()

print("Now the data for the timing between complementary filter and dead reckoning (mean over 10 runs), which I present in my report") # note again that it's probably a tad different from the actual reported values just because of different cpus and no randomseeding
print(f"{'Filter':<37} {'Total (ms)':>10} {'Per sample (us)':>15}")
print()
print(f"{'Dead reckoning':<37} {dead_reck_mean_time_in_ms:>10.1f} {dead_reckoning_microS_per_sample:>15.1f}")
print(f"{'Complementary filter':<37} {comp_filter_mean_time_in_ms:>10.1f} {comp_filter_microS_per_sample:>15.1f}")
print()
print(f"{'So that puts the timing ratio (of comp filter to dead reckoning) at:':<37} {overall_timing_ratio:>10.2f}x")
print(f"{'and my Mult-op ratio (als of comp filter to dead reckoning) was: ':<37} {overall_comp_filt_mults_per_sample/total_mults_for_dead_reck_per_sample:>10.2f}x")
print()






# Now for the stuff I present in section 5, which is the bunny geometry metrics used in collision detection with spheres
print("\nNow, the bunny mesh is getting loaded and normalised (same as in engine)...")
raw_vertices, face_indices = [], []
with open("bunny.obj") as bunny_obj_file: # opening up the file and getting all the vertices and face indicies
    for obj_line in bunny_obj_file:
        vals_from_bunny = obj_line.split()
        if not vals_from_bunny:
            continue
        if vals_from_bunny[0] == 'v': # verticy line
            raw_vertices.append([float(x) for x in vals_from_bunny[1:4]])
        elif vals_from_bunny[0] == 'f': # face indicy line
            face_indices.append([int(x) for x in vals_from_bunny[1:4]])

V = np.array(raw_vertices)

# here I'm centering and scaling the model, using exactly the same normalisation as in the engine's __main__ block
vmin, vmax = V.min(), V.max()
V = (2 * (V - vmin) / (vmax - vmin) - 1) * 1.25
V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2
V[:, 1] -= (V[:, 1].min() + V[:, 1].max()) / 2

num_vertices= len(V)
num_faces= len(face_indices)

min_x_val, max_x_val= V[:,0].min(), V[:,0].max()
min_y_val, max_y_val= V[:,1].min(), V[:,1].max()
min_z_val, max_z_val= V[:,2].min(), V[:,2].max()

# here is the bounding sphere from the centroid (i.e. min sphere centred at centroid holding all the vertices)
centroid= V.mean(axis=0)
vertex_distances= np.linalg.norm(V - centroid, axis=1)
bounding_sphere_r= vertex_distances.max()

# there are the radii that I preset before in my hyperparams list in render.py
sim_radius_big_bunny= 0.20
sim_radius_small_bunny= 0.12

# & these are the aspect ratios, which I use to quantify how non-spherical the mesh is
height= max_y_val-min_y_val
width= max_x_val-min_x_val
depth= max_z_val-min_z_val

print()
print("Now for the bunny geometry stats I report on in Q6:")
print(f"vertices: {num_vertices}")
print(f"& faces: {num_faces}")
print(f"On X, bunny spans: {min_x_val:.3f} to {max_x_val:.3f} (+-{width/2:.2f})")
print(f"On Y it spans: {min_y_val:.3f} to {max_y_val:.3f} (+-{height/2:.2f})")
print(f"& on Z it spans: {min_z_val:.3f} to {max_z_val:.3f}")
print(f"bounding sphere radius is: {bounding_sphere_r:.4f}")
print(f"aspect ratio height/width: {height/width:.2f}")
print(f"aspect ratio height/depth: {height/depth:.2f}")
print(f"min radius to encompass all vertices: {bounding_sphere_r:.4f}")
print(f"but the current combined collision dist I use (0.20+0.12): {sim_radius_big_bunny+sim_radius_small_bunny:.2f} units")