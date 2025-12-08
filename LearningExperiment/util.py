# route_predictor.py
import math
import csv

import numpy as np

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def yaw_interp(yaw0, yaw1, t):
    d = ((yaw1 - yaw0 + 180.0) % 360.0) - 180.0
    return yaw0 + d * t

def load_route_points(csv_path, z_default=0.0):
    pts, speeds = [], []
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in rdr.fieldnames] if rdr.fieldnames else []
        has_speed = 'target_speed_mps' in fieldnames

        for row in rdr:
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z']) if 'z' in row and row['z'] != '' else z_default
            yaw = float(row['yaw'])
            pts.append((x, y, z, yaw))
            speeds.append(max(0.0, float(row['target_speed_mps'])) if has_speed else 0.0)
    return pts, speeds


class RoutePredictor:
    def __init__(self, csv_path):
        self.points, self.v_target = load_route_points(csv_path)
        self.n = len(self.points)
        if self.n < 2:
            raise ValueError("Route must contain at least 2 points.")

        self.seg_dt = []
        self.cum_t = [0.0]
        for i in range(self.n - 1):
            (x0,y0,z0,_), (x1,y1,z1,_) = self.points[i], self.points[i+1]
            ds = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            v = max(self.v_target[i], 1e-2)   # 防止除零
            dt = ds / v
            self.seg_dt.append(dt)
            self.cum_t.append(self.cum_t[-1] + dt)
        self.total_time = self.cum_t[-1]

    def _index_at_time(self, t):
        t = clamp(t, 0.0, self.total_time)
        lo, hi = 0, self.n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum_t[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _state_at_time(self, t):
        idx = self._index_at_time(t)
        if idx == 0:
            x, y, z, yaw = self.points[0]
            v = self.v_target[0]
            return [x, y, z, yaw, v]

        i0 = idx - 1
        (x0,y0,z0,yaw0) = self.points[i0]
        (x1,y1,z1,yaw1) = self.points[idx]

        dt = max(self.seg_dt[i0], 1e-6)
        tau = clamp((t - self.cum_t[i0]) / dt, 0.0, 1.0)

        x = x0 + (x1 - x0) * tau
        y = y0 + (y1 - y0) * tau
        z = z0 + (z1 - z0) * tau
        yaw = yaw_interp(yaw0, yaw1, tau)
        v = self.v_target[i0]  # 段内用入口速度
        return [x, y, yaw, v, 0, 0]

    def future_from_index(self, idx, N, dt=0.1):
        idx = int(clamp(idx, 0, self.n - 1))
        t0 = self.cum_t[idx]
        out = []
        out.append(self._state_at_time(t0))
        for k in range(N):
            t_query = clamp(t0 + (k + 1) * dt, 0.0, self.total_time)
            out.append(self._state_at_time(t_query))
        return np.array(out)

    def nearest_index(self, x, y):
        route_xy = np.array(self.points)[:,0:2]
        d2 = (route_xy[:, 0] - x) ** 2 + (route_xy[:, 1] - y) ** 2
        return int(np.argmin(d2))


def align_yaw_to_init(cost_other_value, init_phi, phi_col=None):
    arr = np.asarray(cost_other_value, dtype=float)

    phi = arr[:, phi_col]
    phi_un = np.unwrap(phi)
    offset = init_phi - phi_un[0]
    phi_aligned = phi_un + offset

    arr[:, phi_col] = phi_aligned
    return arr