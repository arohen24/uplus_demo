from bisect import bisect_left, bisect_right

import numpy as np
from scipy.stats import norm


def floor_int_to_str(num):
    return '%dF' % num if num >= 0 else 'B%d' % num


kelvin_offset = 273.15
_c0 = -8.3144598 / 9.80665 / 0.0289644  # _R / _g / _M


def p2h(p, t_base=15, p_base=1013.25, temp_lapse_rate=0.0065):
    if temp_lapse_rate != 0:
        return (kelvin_offset + t_base) / temp_lapse_rate * (np.power(p / p_base, temp_lapse_rate * _c0) - 1)
    else:
        return (kelvin_offset + t_base) * np.log(p / p_base) * _c0


def p2h_diff(p1, p2, t_base=15, p_base=1013.25, temp_lapse_rate=0.0065):
    return p2h(p2, t_base, p_base, temp_lapse_rate=temp_lapse_rate) \
           - p2h(p1, t_base, p_base, temp_lapse_rate=temp_lapse_rate)


def p_to_h_diff(p1, p2, t1=15, temp_lapse_rate=0.0065):
    if temp_lapse_rate != 0:
        return (kelvin_offset + t1) / temp_lapse_rate * (np.power(p2 / p1, temp_lapse_rate * _c0) - 1)
    else:
        return (kelvin_offset + t1) * np.log(p2 / p1) * _c0


def estimate_floor(h_diff, floor_to_height, start_floor):
    if h_diff >= 0:
        floors = list(range(start_floor, max(floor_to_height.keys()) + 1))
        heights = np.cumsum([0] + [floor_to_height[floor] for floor in floors[:-1]] + [np.inf])
        heights = (heights[:-1] + heights[1:]) / 2

        result = np.diff(np.r_[0, norm.cdf(heights, h_diff, 0.07 * h_diff)])
    else:
        floors = list(range(start_floor, min(floor_to_height.keys()) - 1, -1))
        heights = -np.cumsum([0] + [floor_to_height[floor] for floor in floors[1:]] + [np.inf])
        heights = (heights[:-1] + heights[1:]) / 2

        result = np.diff(np.r_[0, norm.cdf(-heights, -h_diff, -0.07 * h_diff)])

    #     ret = {floor: prob for floor, prob in zip(floors, result) if prob >= 0.05}
    #     ret_sum = sum([item for _, item in ret.items()])
    #     for key, item in ret.items():
    #         ret[key] = item / ret_sum
    #     return ret
    return floors[np.argmax(result)]


class PressureMonitor:
    def __init__(self, initial_floor, time_window=60_000, valid_interval=3_000, vt_interval=7_000):
        self.initial_floor = initial_floor
        self.prev_floor = self.initial_floor
        self.status = self.initial_floor
        self.estimated_floor = None
        print('init status of pressure_monitor:', self.status)

        self.slopes = None
        self.prev_last = None
        self.f2h = {floor: 4 for floor in range(1, 2 + 1)}
        self.f2h.update({floor: 3.8 for floor in range(3, 9 + 1)})
        self.time_window = time_window
        self.valid_interval = valid_interval
        self.vt_interval = vt_interval

        self.ts = np.empty(0)
        self.ps = np.empty(0)
        self.cs = np.empty(0)
        self.labels = np.empty(0)
        self.index_offset = 0
        self.neighbors = list()
        self.label_num = 0

    def insert_one(self, t, p):
        self.prev_last = len(self.ts) - 1
        self.ts = np.append(self.ts, t)
        self.ps = np.append(self.ps, p)
        self.cs = np.append(self.cs, -1)
        self.labels = np.append(self.labels, -1)
        self.neighbors.append(set())
        if self.prev_last > -1:
            self.slopes = np.append(self.slopes, np.abs((p - self.ps[-1]) / (t - self.ts[-1])))
        else:
            self.slopes = np.zeros(1)
        self._update()

    def insert_many(self, ts, ps):
        self.prev_last = len(self.ts) - 1
        self.ts = np.r_[self.ts, ts]
        self.ps = np.r_[self.ps, ps]
        self.cs = np.r_[self.cs, -np.ones(len(ts))]
        self.labels = np.r_[self.labels, -np.ones(len(ts))]
        self.neighbors.extend([set() for _ in range(len(ts))])
        if self.prev_last > -1:
            self.slopes = np.r_[self.slopes, np.abs(np.diff(self.ps[-1 - len(ps):]) / np.diff(self.ts[-1 - len(ts):]))]
        else:
            self.slopes = np.r_[0, np.abs(np.diff(self.ps) / (np.diff(self.ts) + 1e-8))]
        self._update()

    def _eval_core(self, idx, eps=0.03):
        start_idx = bisect_left(self.ts, self.ts[idx] - self.valid_interval)
        end_idx = bisect_right(self.ts, self.ts[idx] + self.valid_interval)
        ds = np.abs(self.ps[start_idx:end_idx] - self.ps[idx])
        ds += eps * (self.slopes[start_idx:end_idx] + self.slopes[idx]) * 1000 * 0.5
        ds += eps * (np.abs(self.ts[start_idx:end_idx] - self.ts[idx]) > self.valid_interval)

        neighbors = np.where(ds <= eps)[0]
        if neighbors.shape[0] == 0:
            self.cs[idx] = False
            return
        neighbors += start_idx
        self.neighbors[idx].update(neighbors + self.index_offset)
        for neighbor in neighbors:
            self.neighbors[neighbor].add(idx + self.index_offset)
        min_idx, max_idx = neighbors.min(), neighbors.max()
        self.cs[idx] = (self.ts[max_idx] - self.ts[min_idx]) >= self.valid_interval

    def _label_by_neighbors(self, idx):
        for v in self.neighbors[idx]:
            v_idx = v - self.index_offset
            if self.cs[v_idx] and self.labels[v_idx] != -1:
                self.labels[idx] = self.labels[v_idx]
                return True
        return False

    def _label_neighbors(self, idx):
        label = self.labels[idx]
        visited = set()
        targets = [idx]
        while True:
            if len(targets) == 0:
                return

            cur = targets.pop()
            visited.add(cur)
            if self.cs[cur]:
                for v in self.neighbors[cur]:
                    v_idx = v - self.index_offset
                    if self.labels[v_idx] == -1:
                        self.labels[v_idx] = label
                        if v_idx not in visited:
                            targets.append(v_idx)

    def _label(self, targets):
        while True:
            changed = False
            cands = list()
            for idx in targets:
                if self.labels[idx] == -1:
                    assigned = self._label_by_neighbors(idx)
                    if assigned:
                        changed = True
                        self._label_neighbors(idx)
                    elif self.cs[idx]:
                        cands.append(idx)

            if not changed:
                break
            targets = cands

        for idx in cands:
            try:
                pcli = np.argwhere(self.labels == self.label_num).flatten()[-1]
                if self.ts[idx] > self.ts[pcli] + self.vt_interval:
                    self.label_num += 1
                    self.labels[idx] = self.label_num
                    self._label_neighbors(idx)

                    prev_floor_indices = np.argwhere(self.labels == self.label_num - 1).flatten()
                    start_idx = bisect_left(self.ts[prev_floor_indices],
                                            self.ts[prev_floor_indices[-1]] - self.valid_interval)
                    prev_p = np.mean(self.ps[prev_floor_indices][start_idx:])

                    cur_floor_indices = np.argwhere(self.labels == self.label_num).flatten()
                    cur_p = np.mean(self.ps[cur_floor_indices])

                    cur_floor = estimate_floor(p2h_diff(prev_p, cur_p), self.f2h, start_floor=self.prev_floor)
                    if self.prev_floor == cur_floor:
                        self.label_num -= 1
                        self.labels[cur_floor_indices] = self.label_num
                    else:
                        self.status = cur_floor
                        print('status of pressure_monitor:', self.status)
                        self.prev_floor = cur_floor

            except IndexError:
                self.labels[idx] = self.label_num
                self._label_neighbors(idx)

    def _update(self):
        outdated_idx = bisect_left(self.ts, self.ts[-1] - self.time_window)
        for _ in range(outdated_idx):
            self.ts = np.delete(self.ts, 0, 0)
            self.ps = np.delete(self.ps, 0, 0)
            self.cs = np.delete(self.cs, 0, 0)
            self.labels = np.delete(self.labels, 0, 0)
            self.neighbors.pop(0)
        self.prev_last = max(-1, self.prev_last - outdated_idx)
        self.index_offset += outdated_idx

        start_idx = bisect_left(self.ts, self.ts[self.prev_last + 1] - self.valid_interval)
        for idx in range(start_idx, len(self.cs)):
            self._eval_core(idx)

        start_idx = bisect_left(self.ts, self.ts[self.prev_last + 1] - 2 * self.valid_interval)
        self._label(list(range(start_idx, len(self.cs))))

        try:
            last_floor_idx = np.argwhere(self.labels != -1).max()
            self.last_floor_t = self.ts[last_floor_idx]
            if self.ts[-1] - self.last_floor_t > self.valid_interval:
                if self.ps[-1] > self.ps[last_floor_idx]:
                    self.status = -999  # 'floor decreasing'
                else:
                    self.status = 999  # 'floor increasing'
                print('status of pressure_monitor:', self.status)

                prev_floor_indices = np.argwhere(self.labels == self.label_num).flatten()
                start_idx = bisect_left(self.ts[prev_floor_indices],
                                        self.ts[prev_floor_indices[-1]] - self.valid_interval)
                prev_p = np.mean(self.ps[prev_floor_indices][start_idx:])

                start_idx = bisect_left(self.ts, self.ts[-1] - self.valid_interval)
                cur_p = np.average(self.ps[start_idx:], weights=np.power(2, (self.ts[start_idx:] - self.ts[-1]) / 1000))

                cur_floor = estimate_floor(p2h_diff(prev_p, cur_p), self.f2h, start_floor=self.prev_floor)
                self.estimated_floor = cur_floor
                # print('estimated cur_floor:', cur_floor)
            else:
                self.estimated_floor = None
        except ValueError:
            self.prev_floor = self.initial_floor
            self.status = self.initial_floor
            print('status of pressure_monitor:', self.status)
