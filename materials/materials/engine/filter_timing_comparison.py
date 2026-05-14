# run this from your coursework directory
# it times dead reckoning vs complementary filter over the full dataset
# and counts the per-sample operation breakdown

import time
import math
import pandas as pd
import numpy as np

# ---- load your functions from the existing file ----
# (copy these in or import from your file; pasted here for self-containment)

def qprod(w1,x1,y1,z1,w2,x2,y2,z2):
    return (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)

def euler_q(r,p,y):
    hr,hp,hy=r/2,p/2,y/2
    return (math.cos(hr)*math.cos(hp)*math.cos(hy)+math.sin(hr)*math.sin(hp)*math.sin(hy),
            math.sin(hr)*math.cos(hp)*math.cos(hy)-math.cos(hr)*math.sin(hp)*math.sin(hy),
            math.cos(hr)*math.sin(hp)*math.cos(hy)+math.sin(hr)*math.cos(hp)*math.sin(hy),
            math.cos(hr)*math.cos(hp)*math.sin(hy)-math.sin(hr)*math.sin(hp)*math.cos(hy))

def to_global(qw,qx,qy,qz,vx,vy,vz):
    t=qprod(qw,qx,qy,qz,0,vx,vy,vz)
    g=qprod(t[0],t[1],t[2],t[3],qw,-qx,-qy,-qz)
    return g[1],g[2],g[3]

df = pd.read_csv("IMUData.csv", skipinitialspace=True)
df.columns = df.columns.str.strip()
df[['gyroscope.X','gyroscope.Y','gyroscope.Z']] = np.deg2rad(
    df[['gyroscope.X','gyroscope.Y','gyroscope.Z']])

N = len(df)
REPEATS = 10  # run each filter multiple times for stable timing

# ================================================================
# DEAD RECKONING
# ================================================================
def run_dead_reckoning():
    q=(1.,0.,0.,0.)
    for i in range(1, N):
        dt = df['time'][i] - df['time'][i-1]
        dq = euler_q(df['gyroscope.X'][i]*dt,
                     df['gyroscope.Y'][i]*dt,
                     df['gyroscope.Z'][i]*dt)
        q = qprod(dq[0],dq[1],dq[2],dq[3], q[0],q[1],q[2],q[3])
        n = math.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
        q = (q[0]/n, q[1]/n, q[2]/n, q[3]/n)
    return q

# ================================================================
# COMPLEMENTARY FILTER (accel only, alpha=0.02)
# ================================================================
def run_complementary_filter():
    sa_x,sa_y,sa_z = 0.,0.,1.
    q=(1.,0.,0.,0.)
    for i in range(1, N):
        dt = df['time'][i] - df['time'][i-1]
        dq = euler_q(df['gyroscope.X'][i]*dt,
                     df['gyroscope.Y'][i]*dt,
                     df['gyroscope.Z'][i]*dt)
        gw,gx,gy,gz = qprod(dq[0],dq[1],dq[2],dq[3], q[0],q[1],q[2],q[3])
        n = math.sqrt(gw**2+gx**2+gy**2+gz**2)
        gw,gx,gy,gz = gw/n,gx/n,gy/n,gz/n

        rax = df['accelerometer.X'][i]
        ray = df['accelerometer.Y'][i]
        raz = df['accelerometer.Z'][i]
        sa_x = 0.1*rax + 0.9*sa_x
        sa_y = 0.1*ray + 0.9*sa_y
        sa_z = 0.1*raz + 0.9*sa_z
        amag = math.sqrt(sa_x**2+sa_y**2+sa_z**2)

        ga_x,ga_y,ga_z = to_global(gw,gx,gy,gz, sa_x,sa_y,sa_z)
        an = math.sqrt(ga_x**2+ga_y**2+ga_z**2)
        if an > 1e-8:
            nax,nay,naz = ga_x/an,ga_y/an,ga_z/an
        else:
            nax,nay,naz = 0.,0.,1.

        tx = nay; ty = -nax
        tmag = math.sqrt(tx**2+ty**2)
        if tmag > 1e-8: tx/=tmag; ty/=tmag
        phi = math.acos(max(-1.,min(1.,naz)))

        ok = 0.85 < amag < 1.15
        ca = 0.02*phi if ok else 0.
        aw = math.cos(ca/2)
        axc = tx*math.sin(ca/2)
        ayc = ty*math.sin(ca/2)
        afw,afx,afy,afz = qprod(aw,axc,ayc,0., gw,gx,gy,gz)
        n = math.sqrt(afw**2+afx**2+afy**2+afz**2)
        q = (afw/n, afx/n, afy/n, afz/n)
    return q

# ================================================================
# TIMING
# ================================================================
print(f"Dataset: {N} samples at ~256 Hz ({df['time'].iloc[-1]:.1f}s)")
print(f"Each filter run {REPEATS} times for stable measurement.")
print()

dr_times = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    run_dead_reckoning()
    dr_times.append(time.perf_counter() - t0)

ac_times = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    run_complementary_filter()
    ac_times.append(time.perf_counter() - t0)

dr_mean = sum(dr_times)/REPEATS * 1000
ac_mean = sum(ac_times)/REPEATS * 1000

print("=" * 50)
print("TIMING RESULTS")
print("=" * 50)
print(f"Dead reckoning      : {dr_mean:.2f} ms total  ({dr_mean/N*1000:.3f} µs/sample)")
print(f"Complementary filter: {ac_mean:.2f} ms total  ({ac_mean/N*1000:.3f} µs/sample)")
print(f"Ratio (CF / DR)     : {ac_mean/dr_mean:.2f}x more expensive")
print()

# ================================================================
# OPERATION COUNT (analytical, per sample)
# ================================================================
print("=" * 50)
print("ESTIMATED ARITHMETIC OPERATIONS PER SAMPLE")
print("=" * 50)
print()
print("Dead reckoning:")
print("  gyro * dt              :  3 mul")
print("  euler_to_quaternion    :  6 trig,  12 mul,  6 add")
print("  quaternion_product     :  16 mul,  12 add")
print("  normalise              :  4 mul,   3 add,   1 sqrt,  4 div")
print("  --------------------------------")
dr_mul=3+12+16+4; dr_add=6+12+3; dr_trig=6; dr_sqrt=1; dr_div=4
print(f"  TOTAL: ~{dr_mul} mul, ~{dr_add} add, {dr_trig} trig, {dr_sqrt} sqrt")
print()
print("Complementary filter (additional on top of DR):")
print("  EMA (x3 axes)          :  6 mul,   6 add")
print("  magnitude check        :  3 mul,   2 add,   1 sqrt")
print("  transform_to_global    :  2x qprod = 32 mul, 24 add")
print("  tilt_axis_and_angle    :  12 mul,  8 add,   1 trig,  1 sqrt")
print("  build correction quat  :  3 mul,   0 add,   2 trig")
print("  apply correction qprod :  16 mul,  12 add")
print("  normalise again        :  4 mul,   3 add,   1 sqrt,  4 div")
print("  --------------------------------")
extra_mul=6+3+32+12+3+16+4; extra_add=6+2+24+8+0+12+3; extra_trig=3; extra_sqrt=3
print(f"  EXTRA: ~{extra_mul} mul, ~{extra_add} add, {extra_trig} trig, {extra_sqrt} sqrt")
print()
total_mul=dr_mul+extra_mul; total_add=dr_add+extra_add
print(f"CF total : ~{total_mul} mul, ~{total_add} add vs DR ~{dr_mul} mul, ~{dr_add} add")
print(f"CF is approximately {total_mul/dr_mul:.1f}x more multiply-heavy than DR")
print()
print("Over full dataset:")
print(f"  DR  total muls: ~{dr_mul*N:,}")
print(f"  CF  total muls: ~{total_mul*N:,}")
print(f"  Extra operations from accelerometer fusion: ~{(total_mul-dr_mul)*N:,} muls alone")
