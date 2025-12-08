import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import draw_waypoints
import carla
import csv
import math

def ref_speed_by_road(road_id: int) -> float:
    if road_id in (101, 67):
        return 8.0
    if road_id == 76:
        return 6.0
    return 4.5

def seg_dist(a, b) -> float:
    la, lb = a.transform.location, b.transform.location
    dx, dy, dz = la.x - lb.x, la.y - lb.y, la.z - lb.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def trim_head_obtuse(prev_wp, next_wps, angle_eps_deg=2.0, dup_tol=0.05):
    if not next_wps:
        return next_wps

    tf = prev_wp.transform
    loc0 = tf.location
    fwd  = tf.get_forward_vector()
    fn = math.sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z) or 1.0
    fx, fy, fz = fwd.x/fn, fwd.y/fn, fwd.z/fn

    cos_eps = math.cos(math.radians(90.0 - angle_eps_deg))

    keep_idx = 0
    for j, wp in enumerate(next_wps):
        loc = wp.transform.location
        dx, dy, dz = loc.x - loc0.x, loc.y - loc0.y, loc.z - loc0.z
        dn = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0

        cos_theta = (fx*dx + fy*dy + fz*dz) / dn

        if cos_theta <= cos_eps:
            keep_idx = j + 1
            continue
        else:
            keep_idx = j
            break

    if keep_idx < len(next_wps) and seg_dist(prev_wp, next_wps[keep_idx]) < dup_tol:
        keep_idx += 1

    return next_wps[keep_idx:]

client = carla.Client('localhost', 2000)
world = client.get_world()

m = world.get_map()
spawn_points = m.get_spawn_points()

route_points = [spawn_points[169], spawn_points[115], spawn_points[175], spawn_points[165]]
# route_points = [spawn_points[226], spawn_points[164]]

distance = 0.1
grp = GlobalRoutePlanner(m, distance)

wps = []
for i in range(len(route_points) - 1):
    route = grp.trace_route(carla.Location(route_points[i].location),
                            carla.Location(route_points[i+1].location))

    seg = [pair[0] for pair in route]

    if wps:
        seg = trim_head_obtuse(wps[-1], seg, angle_eps_deg=2.0, dup_tol=0.05)

    wps.extend(seg)

wps = wps[:8600]

n = len(wps)
ds = [seg_dist(wps[i], wps[i+1]) for i in range(n-1)]

a_max = 3

v_limit = [ref_speed_by_road(wp.road_id) for wp in wps]
v = v_limit[:]
v[0] = 1.0

for i in range(1, n):
    v_allow = math.sqrt(max(v[i-1]**2 + 2.0*a_max*ds[i-1], 0.0))
    v[i] = min(v[i], v_allow)

for i in range(n-2, -1, -1):
    v_allow = math.sqrt(max(v[i+1]**2 + 2.0*a_max*ds[i], 0.0))
    v[i] = min(v[i], v_allow)

draw_waypoints(world, wps)

with open("route_points.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z", "yaw", "road_id",
                     "target_speed_mps", "target_speed_kph"])
    for wp, v_tar in zip(wps, v):
        loc = wp.transform.location
        rot = wp.transform.rotation
        yaw_deg = 90 - rot.yaw  # 维持你的坐标系转换
        writer.writerow([loc.y, loc.x, loc.z, np.deg2rad(yaw_deg), wp.road_id,
                         round(v_tar, 3), round(v_tar*3.6, 3)])