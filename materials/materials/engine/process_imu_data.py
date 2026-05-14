import numpy as np
import pandas as pd
import math

# the following function is a custom reader for loading in the IMUData.csv file
def readin_imu_dataset(imu_data_filename="IMUData.csv"):
    imu_dataset = pd.read_csv(imu_data_filename, skipinitialspace=True) # (the file seemed to have a leading whitespace, so this operation was added to ditch it)

    # this takes the rotational rates from deg/s to rad/s
    gyroscope_cols_in_rads = ['gyroscope.X', 'gyroscope.Y', 'gyroscope.Z']
    imu_dataset[gyroscope_cols_in_rads] = np.deg2rad(imu_dataset[gyroscope_cols_in_rads])

    return imu_dataset


imu_data = readin_imu_dataset()
print(imu_data.head())

# now for all the manipualtion functions

# i. calculating quaternion from euler radians
def convert_euler_to_quaternion(roll_x, pitch_y, yaw_z):
    # for this I'm using the standard eq.:
    # w = cosroll*cospitch*cosyaw + sinroll*sinpitch*sinyaw ; x = sinroll*cospitch*cosyaw - cosroll*sinpitch*sinyaw
    # y = cosroll*sinpitch*cosyaw + sinroll*cospitch*sinyaw ; z = cosroll*cospitch*sinyaw - sinroll*sinpitch*cosyaw
    # output is the quaternion [w, x, y, z]

    # half-angles in rads are 1st precalculted
    half_roll = roll_x * 0.5
    half_pitch = pitch_y * 0.5
    half_yaw = yaw_z * 0.5

    # now calcluating trigonmetric vals
    cos_roll = math.cos(half_roll)
    sin_roll = math.sin(half_roll)
    cos_pitch = math.cos(half_pitch)
    sin_pitch = math.sin(half_pitch)
    cos_yaw = math.cos(half_yaw)
    sin_yaw = math.sin(half_yaw)

    # now each of the w x y z of the quaternion is constructed
    w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw

    return w, x, y, z



# ii. now the opposite, calculating euler radians from quaternion
def convert_quaternion_to_euler(w, x, y, z):
    # for this I'm using the standard conversion equations:
    # roll = atan2(2(wx + yz), 1 - 2(x^2 + y^2))
    # pitch = asin(2(wy - zx))      
    # & yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    # output are euler angles (roll_x, pitch_y, yaw_z) in radians
    # (note: I verified my equation at: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler
    # firstly, calculating the roll (aka x-axis rotation)
    sin_roll_cos_pitch = 2 * (w * x + y * z)
    cos_roll_cos_pitch = 1 - 2 * (x * x + y * y)
    roll_x = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    # secondly, calculating pitch (aka y-axis rotation)
    sin_pitch = 2 * (w * y - z * x)
    
    # here I handle potential floating-point inaccuracies (or gimbal lock) by clamping sin val to valid [-1.0, 1.0] range
    if abs(sin_pitch) >= 1:
        pitch_y = math.copysign(math.pi / 2, sin_pitch)
    else:
        pitch_y = math.asin(sin_pitch)

    # finally, calculating yaw (aka z-axis rotation)
    sin_yaw_cos_pitch = 2 * (w * z + x * y)
    cos_yaw_cos_pitch = 1 - 2 * (y * y + z * z)
    yaw_z = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

    return roll_x, pitch_y, yaw_z


# iii. next, I convert a quaternion to its conjugate (i.e. inversely rotating it)
def convert_quaternion_to_conjugate(w, x, y, z):
    # eq. is nice and simple for this, just negating x, y & z: q* = [w, -x, -y, -z]
    # note: eq. for conjugate fetched from: https://mathworld.wolfram.com/QuaternionConjugate.html
    return w, -x, -y, -z # noting that w remains the same, only the imaginary vector part gets negated


# iv. finally, this one is calcting the quaternion product of a w/ b
def calculate_quaternion_product(w1, x1, y1, z1, w2, x2, y2, z2):
    # just standard hamilton product formula here that i got from: https://www.tutis.ca/Rotate/7quaternions.htm#:~:text=i2%20=%20j2%20=%20k,%7Cq%7C2%20(7.4)
    combined_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    combined_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    combined_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    combined_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return combined_w, combined_x, combined_y, combined_z

# MYNOTE: DO QUATERNIONS NEED TO BE NORMALISED AFTER PRODUCT?
# this is my attempt at implementing dead reckoning
# I took the maths for this from lecture 4's slides
def dead_reckoning_orientation_calc_implementation(imu_dataframe):
    # I need to evaluate sequential orientation over time w/ this code
    # q[0] per the coursework spec is [1,0,0,0], representing the initial identity state
    current_w, current_x, current_y, current_z = 1.0, 0.0, 0.0, 0.0

    # i'm using an array to store tracked orientations history
    orientation_history_container = [(current_w, current_x, current_y, current_z)]

    # now for the time difference calcs
    # i'm iterating through dataset, but I start at idx1 as the timestamp at idx0 is 0s
    for i in range(1, len(imu_dataframe)):
        # 1st, I calcualte the elapsed time betwween current and previous sensor reading
        sensor_dt = imu_dataframe['time'][i] - imu_dataframe['time'][i-1]

        # 2nd, i calculate the angular displacment for each axis
        roll_change = imu_dataframe['gyroscope.X'][i] * sensor_dt # (i.e. rotational speed (rad/s) * time elapsed (s))
        pitch_change = imu_dataframe['gyroscope.Y'][i] * sensor_dt
        yaw_change = imu_dataframe['gyroscope.Z'][i] * sensor_dt

        # 3rd, I'm converting the small euler rotation -> quaternion using my function I was asked to implmenet before
        w_change, x_change, y_change, z_change = convert_euler_to_quaternion(roll_change, pitch_change, yaw_change)

        # and last but not least, accumulating rotation w/ hamilton product 
        # per lecture slides, eq. is: q_current = delta_q_current * q_previous
        current_w, current_x, current_y, current_z = calculate_quaternion_product(
            w_change, x_change, y_change, z_change,  # quaternion a (==the new movement)
            current_w, current_x, current_y, current_z) # & quaternion b (==the previous state)

        # I also added normalisation to unit length so floating-point drift doesn't accumulate, which seemed like a standard thing in these dead-recknoing filters
        normal_val = math.sqrt(current_w**2 + current_x**2 + current_y**2 + current_z**2)
        current_w, current_x, current_y, current_z = current_w/normal_val, current_x/normal_val, current_y/normal_val, current_z/normal_val

        # now it's done, so I append the new orientation calc to the tracking array & move onto the next
        orientation_history_container.append((current_w, current_x, current_y, current_z))

    return orientation_history_container
    









def test_dead_reckoning_filter():
    # Objective: Verify the sequential accumulation of orientation states.
    # Mechanism: Create a synthetic DataFrame to simulate controlled IMU telemetry.

    # 1. Constructing the synthetic dataset
    # Index 0: Start time 0.0s (baseline)
    # Index 1: dt = 1.0s, zero rotation to test stability
    # Index 2: dt = 1.0s, 90-degree roll (pi/2 rad/s)
    # Index 3: dt = 1.0s, 90-degree roll (pi/2 rad/s) -> total accumulated roll should be 180 degrees
    controlled_data = {
        'time': [0.0, 1.0, 2.0, 3.0],
        'gyroscope.X': [0.0, 0.0, math.pi/2, math.pi/2],
        'gyroscope.Y': [0.0, 0.0, 0.0, 0.0],
        'gyroscope.Z': [0.0, 0.0, 0.0, 0.0]
    }
    synthetic_imu_df = pd.DataFrame(controlled_data)

    # 2. Executing the filter
    tracked_orientations = dead_reckoning_orientation_calc_implementation(synthetic_imu_df)

    # 3. Evaluating the outcomes against expected mathematical states
    
    # State 0: Initial identity state
    w0, x0, y0, z0 = tracked_orientations[0]
    print(f"State 0 (Initial)       : w={w0:.3f}, x={x0:.3f}, y={y0:.3f}, z={z0:.3f}")
    # Expected: (1.000, 0.000, 0.000, 0.000)

    # State 1: Zero rotation applied (should remain identical to State 0)
    w1, x1, y1, z1 = tracked_orientations[1]
    print(f"State 1 (Zero Rotation) : w={w1:.3f}, x={x1:.3f}, y={y1:.3f}, z={z1:.3f}")
    # Expected: (1.000, 0.000, 0.000, 0.000)

    # State 2: 90-degree rotation on X applied
    w2, x2, y2, z2 = tracked_orientations[2]
    print(f"State 2 (90deg Roll)    : w={w2:.3f}, x={x2:.3f}, y={y2:.3f}, z={z2:.3f}")
    # Expected: (0.707, 0.707, 0.000, 0.000) - mathematically equivalent to sin/cos of 45 degrees

    # State 3: Another 90-degree rotation on X applied (Total accumulated: 180 degrees)
    w3, x3, y3, z3 = tracked_orientations[3]
    print(f"State 3 (180deg Roll)   : w={w3:.3f}, x={x3:.3f}, y={y3:.3f}, z={z3:.3f}")
    # Expected: (0.000, 1.000, 0.000, 0.000) - mathematically equivalent to cos(90)=0, sin(90)=1

# Execute the test
test_dead_reckoning_filter()






# def test_quaternion_conversions():
#     # Test Case 1: Identity Rotation (0, 0, 0)
#     # Objective: Ensure no rotation results in the identity quaternion [1, 0, 0, 0]
#     w, x, y, z = convert_euler_to_quaternion(0, 0, 0)
#     print(f"Identity Test: w={w}, x={x}, y={y}, z={z}")
#     # Expected: (1.0, 0.0, 0.0, 0.0)

#     # Test Case 2: 90-degree Roll (PI/2, 0, 0)
#     # Objective: Rotate 90 degrees around the X-axis
#     w, x, y, z = convert_euler_to_quaternion(math.pi/2, 0, 0)
#     print(f"90deg Roll Test: w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}")
#     # Expected: (0.707, 0.707, 0.0, 0.0) -> sin/cos of 45 degrees

#     # Test Case 3: 180-degree Yaw (0, 0, math.pi)
#     # Objective: Rotate 180 degrees around the Z-axis
#     w, x, y, z = convert_euler_to_quaternion(0, 0, math.pi)
#     print(f"180deg Yaw Test: w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}")
#     # Expected: (0.0, 0.0, 0.0, 1.0) -> cos(90)=0, sin(90)=1

# # Execute tests
# test_quaternion_conversions()

# print("\n\n\n")



# def test_euler_extractions():
#     # Test Case 1: Identity Rotation (1, 0, 0, 0)
#     # Objective: Ensure the identity quaternion extracts to zero rotation on all axes
#     roll_x, pitch_y, yaw_z = convert_quaternion_to_euler(1.0, 0.0, 0.0, 0.0)
#     print(f"Identity Test: roll={roll_x:.3f}, pitch={pitch_y:.3f}, yaw={yaw_z:.3f}")
#     # Expected: (0.000, 0.000, 0.000)

#     # Test Case 2: 90-degree Roll (0.7071, 0.7071, 0.0, 0.0)
#     # Objective: Extract a 90-degree rotation around the X-axis
#     # Note: 0.707106 is a more precise approximation of sin(pi/4) and cos(pi/4)
#     roll_x, pitch_y, yaw_z = convert_quaternion_to_euler(0.7071068, 0.7071068, 0.0, 0.0)
#     print(f"90deg Roll Test: roll={roll_x:.3f}, pitch={pitch_y:.3f}, yaw={yaw_z:.3f}")
#     # Expected: (1.571, 0.000, 0.000) -> mathematically equivalent to PI/2

#     # Test Case 3: 180-degree Yaw (0.0, 0.0, 0.0, 1.0)
#     # Objective: Extract a 180-degree rotation around the Z-axis
#     roll_x, pitch_y, yaw_z = convert_quaternion_to_euler(0.0, 0.0, 0.0, 1.0)
#     print(f"180deg Yaw Test: roll={roll_x:.3f}, pitch={pitch_y:.3f}, yaw={yaw_z:.3f}")
#     # Expected: (0.000, 0.000, 3.142) -> mathematically equivalent to PI

# # Execute tests
# test_euler_extractions()

# def test_conjugate_and_product():
#     # Test Case 1: Conjugate of a 90-degree Roll
#     # Objective: Verify that the conjugate correctly negates the vector components
#     w, x, y, z = 0.707, 0.707, 0.0, 0.0
#     cw, cx, cy, cz = convert_quaternion_to_conjugate(w, x, y, z)
#     print(f"Conjugate Test: w={cw}, x={cx}, y={cy}, z={cz}")
#     # Expected: (0.707, -0.707, 0.0, 0.0)

#     # Test Case 2: Product with Identity
#     # Objective: Verify that multiplying any quaternion by the identity results in itself
#     # q1 = 90deg roll, q2 = Identity [1, 0, 0, 0]
#     rw, rx, ry, rz = calculate_quaternion_product(0.707, 0.707, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
#     print(f"Product Identity Test: w={rw}, x={rx}, y={ry}, z={rz}")
#     # Expected: (0.707, 0.707, 0.0, 0.0)

#     # Test Case 3: Product of Two 90-degree Rolls (should equal 180-degree roll)
#     # q1 = [0.707, 0.707, 0, 0], q2 = [0.707, 0.707, 0, 0]
#     # Mathematically: (0.707*0.707 - 0.707*0.707) = 0 for w, (0.707*0.707 + 0.707*0.707) = 1 for x
#     fw, fx, fy, fz = calculate_quaternion_product(0.7071, 0.7071, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0)
#     print(f"180deg Result Test: w={fw:.1f}, x={fx:.1f}, y={fy:.1f}, z={fz:.1f}")
#     # Expected: (0.0, 1.0, 0.0, 0.0)

# # Execute tests
# test_conjugate_and_product()