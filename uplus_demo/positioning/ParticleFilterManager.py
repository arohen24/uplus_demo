import math

import numpy as np

from .PDRManager import PDRManager


class ParticleFilterManager:
    def __init__(self, init_coord, occ_map, res, frequency=100, grav_acc=9.80665, N=4000, init_heading=None):
        self.frequency = frequency
        self._G = grav_acc
        self.res = res

        self.pdr_manager = PDRManager(frequency=frequency)

        self.occ_map = occ_map

        self.particles = np.tile(init_coord, (N, 1))
        print('particles', self.particles.shape)
        # self.particles += np.random.normal(0, 0.5, size=self.particles.shape)
        self.particles += np.random.normal(0, 0.25, size=self.particles.shape)
        self.N = self.particles.shape[0]
        print('# of particles:', self.N)

        self.history_x = np.array(self.particles[:, 0], copy=True)
        self.history_y = np.array(self.particles[:, 1], copy=True)

        self.step_length_noise_lb = 0.95
        self.step_length_noise_ub = 1.2
        self.step_length_noise = np.clip(np.random.normal(1.0, 0.1, size=N),
                                         self.step_length_noise_lb, self.step_length_noise_ub)
        if init_heading is None:
            self.heading = np.random.uniform(-np.pi, np.pi, size=N)
        else:
            heading_rad = np.arctan2(init_heading[1], init_heading[0])
            self.heading = np.random.normal(heading_rad, np.pi / 18, size=N)
        self.drift_lim = 0.02
        self.drift = np.clip(np.random.normal(0, self.drift_lim / 3, size=N), -self.drift_lim, self.drift_lim)

        self.probs = np.full(shape=N, fill_value=1 / N)
        self.records = [np.c_[self.particles, self.probs]]

        self.yaws_to_show = np.empty(0)
        self.step_lengths_to_show = np.empty(0)

    def _is_valid(self, i, j):
        try:
            return self.occ_map[j, i]
        except IndexError:
            return False

    def update(self, T, A, G):
        if self.occ_map is None:
            raise Exception('occ_map is needed')

        self.pdr_manager.update(T, A[:, 0], A[:, 1], A[:, 2], G[:, 0], G[:, 1], G[:, 2])
        step_lengths = self.pdr_manager.strideLength
        if len(step_lengths) == 0:
            return None
        print('# of steps:', len(step_lengths))
        yaws = self.pdr_manager.psi

        self.yaws_to_show = np.r_[self.yaws_to_show, yaws]
        self.step_lengths_to_show = np.r_[self.step_lengths_to_show, step_lengths]

        estimations = list()
        for yaw, step_length in zip(yaws, step_lengths):
            sls = step_length * self.step_length_noise
            self.heading += self.drift
            self.particles[:, 0] += sls * np.cos(self.heading + yaw)
            self.particles[:, 1] += sls * np.sin(self.heading + yaw)

            self.history_x = np.c_[self.history_x, self.particles[:, 0]]
            self.history_y = np.c_[self.history_y, self.particles[:, 1]]

            iii = np.array(self.particles / self.res, dtype=np.int64)
            is_inside = [self._is_valid(*_) for _ in iii]

            if np.sum(is_inside) == 0:
                center = np.mean(self.particles, axis=0)
                dists = np.linalg.norm(self.particles - center, axis=1)
                mean_dist = np.mean(dists)
                if True:  # mean_dist > 0.545:
                    print('--- nothing inside (mean_dist: %.4f) ---' % mean_dist)
                    #                     plt.figure(figsize=(7, 3.5), dpi=300)

                    #                     img = Image.open(BytesIO(floor['floorPlanImage']['binary']))
                    #                     width, height = floor['floorPlanImage']['size']
                    #                     ratio = floor['ppm']
                    #                     plt.imshow(img, extent=[0, width / ratio, 0, height / ratio], alpha=0.5, zorder=0)

                    #                     outer_line = np.array(floor['layout']['outline'] + [floor['layout']['outline'][0]])
                    #                     plt.plot(outer_line[:, 0], outer_line[:, 1], lw=0.1, c='gray', zorder=1)
                    #                     for except_area in floor['layout']['excepts']:
                    #                         except_layout = except_area + [except_area[0]]
                    #                         inner_line = np.array(except_layout)
                    #                         plt.plot(inner_line[:, 0], inner_line[:, 1], lw=0.1, c='gray', zorder=1)

                    #                     position = np.zeros((len(yaws), 2))
                    #                     for i in range(len(yaws)):
                    #                         if i == 0:
                    #                             position[i, 0] = step_lengths[i]*np.cos(yaws[i])
                    #                             position[i, 1] = step_lengths[i]*np.sin(yaws[i])
                    #                         else:
                    #                             position[i, 0] = position[i-1, 0] + step_lengths[i]*np.cos(yaws[i])
                    #                             position[i, 1] = position[i-1, 1] + step_lengths[i]*np.sin(yaws[i])
                    #                     plt.scatter(position[:, 0], position[:, 1], c='none', marker='o', edgecolors='r', alpha=0.3, s=0.1)

                    #                     plt.scatter(self.records[-1][:, 0], self.records[-1][:, 1], c=self.records[-1][:, 2], s=0.2, alpha=0.1)
                    #                     plt.colorbar()
                    #                     plt.axis('equal')

                    #                     plt.show()

                    # self.probs = np.ones(len(is_inside)) / len(is_inside)
                    pass
                    mean_dist = None
                    # return None
            else:
                self.probs = is_inside / np.sum(is_inside)
                esti = np.average(self.particles, weights=self.probs, axis=0)
                estimations.append(esti)

                in_indices = np.where(is_inside)[0]
                center = np.mean(self.particles[in_indices], axis=0)
                dists = np.linalg.norm(self.particles[in_indices] - center, axis=1)
                mean_dist = np.mean(dists)

            self.records.append(np.c_[self.particles, self.probs])

            # resampling
            new_indices = np.random.choice(self.particles.shape[0], self.N, p=self.probs)

            self.history_x = self.history_x[new_indices]
            self.history_y = self.history_y[new_indices]

            std_particle_coord = 0.01
            std_step_length_noise = 0.05
            heading_noise_lim = 0.13
            in_ratio = np.sum(is_inside) / self.N
            if in_ratio < 1e-8:
                mul = 1 + (1 - in_ratio) * (5 - 1)
                std_particle_coord *= mul
            #     std_step_length_noise *= mul
            #     heading_noise_lim *= 1 + (mul - 1) * 0.5
            step_length_noise_lb = self.step_length_noise_lb
            if mean_dist is not None:
                trigger_d = 1.0
                step_length_noise_lb = self.step_length_noise_lb * min(mean_dist, trigger_d) / trigger_d

            self.particles = self.particles[new_indices] + np.random.normal(0, std_particle_coord, size=(self.N, 2))
            self.step_length_noise = np.clip(
                self.step_length_noise[new_indices] * np.random.normal(1, std_step_length_noise, size=self.N),
                step_length_noise_lb, self.step_length_noise_ub)
            self.heading = self.heading[new_indices] + np.clip(np.random.normal(0, heading_noise_lim / 3, size=self.N),
                                                               -heading_noise_lim, heading_noise_lim)
            # self.drift = np.clip(self.drift[new_indices] * np.random.normal(1, 0.01, size=self.N),
            #                      -self.drift_lim, self.drift_lim)

        if len(estimations) != 0:
            return estimations
        else:
            return [np.average(self.particles, weights=self.probs, axis=0)]
