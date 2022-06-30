import json
import math
import os
import sys
import time
from bisect import bisect_right
from io import BytesIO
from threading import Thread

import numpy as np
import requests
from PIL import Image
from matplotlib.gridspec import GridSpec
from pymongo import MongoClient
from pyproj import Proj
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from uplus_demo.positioning.ParticleFilterManager import ParticleFilterManager
from uplus_demo.positioning.PressureMonitor import PressureMonitor


def send_coord(lonlat_coord, bldg_id="", floor_id="", uuid=2):
    data = {
        "lnglat": {"type": "Point", "coordinate": list(lonlat_coord)},
        "level": 0,
        "bd": "",
        "floorID": f'{floor_id}',
        "floorId": f'{floor_id}',
        "buildingId": f'{bldg_id}',
        "fingerprint": {
            "wifi": [],
            "uuid": f'{uuid}'
        },
        "uuid": f'{uuid}',
        "regTime": time.time()
    }

    headers = {'Content-type': 'application/json; charset=utf-8', 'Accept': 'text/plain'}
    cookies = {'session_id': 'server'}
    r = requests.post("http://143.248.56.76:18888/API/pdrlocation",
                      headers=headers, data=json.dumps(data), cookies=cookies)
    return r.status_code, r.reason, r.text


def rasterize(polygon: Polygon, res=0.5):
    start_time = time.time()

    _, _, x_max, y_max = polygon.bounds

    w_len = math.ceil(x_max / res)
    if w_len <= x_max / res:
        w_len += 1

    h_len = math.ceil(y_max / res)
    if h_len <= y_max / res:
        h_len += 1

    result = np.empty((h_len, w_len), dtype=np.bool)
    for i in range(w_len):
        for j in range(h_len):
            x = res / 2 + i * res
            y = res / 2 + j * res
            result[j, i] = polygon.contains(Point(x, y))

    print('rasterize is done: %.3f sec.' % (time.time() - start_time))
    return result


def make_answer_sheet(madgwick_data, labeled_data, floor, score, step_lengths, records, do_save=True):
    import matplotlib.pyplot as plt
    vec = np.subtract(labeled_data[1], labeled_data[0])
    vec_complex = complex(vec[0], vec[1])
    gap = np.angle(vec_complex) - madgwick_data[0]

    fig = plt.figure(figsize=(7, 4), dpi=200, constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[:, 2])
    # ax2 = fig.add_subplot(gs[:, 2], sharey=ax1)

    # PDR 결과 출력
    _xs = [labeled_data[0, 0]]
    _ys = [labeled_data[0, 1]]
    for yaw, step_length in zip(madgwick_data, step_lengths):
        _xs.append(_xs[-1] + step_length * np.cos(yaw + gap))
        _ys.append(_ys[-1] + step_length * np.sin(yaw + gap))
    ax1.plot(_xs, _ys, lw=1, c='C1', alpha=0.8, zorder=9)
    ax2.plot(_xs, _ys, lw=1, c='C1', alpha=0.8)

    # records 출력
    _xs = [labeled_data[0, 0]]
    _ys = [labeled_data[0, 1]]
    for record in records:
        _x, _y = np.average(record[:, :2], weights=record[:, 2], axis=0)
        _xs.append(_x)
        _ys.append(_y)
    ax1.plot(_xs, _ys, lw=1, c='C2', alpha=0.8, zorder=9)
    ax2.plot(_xs, _ys, lw=1, c='C2', alpha=0.8)

    # ax2.scatter([_xs[0]], [_ys[0]], s=16, c='g', zorder=3)
    # ax2.text(_xs[0], _ys[0], seq['ref']['start'])

    ax1.axis('equal')
    color = 'b'
    # color = 'b' if len(seq['ref']['end']) > 0 else 'r'
    # ax1.scatter([_xs[-1]], [_ys[-1]], marker='x', s=16, c=color, alpha=0.5, zorder=3)
    # ax2.scatter([_xs[-1]], [_ys[-1]], marker='x', s=16, c=color, zorder=3)
    # ax2.text(_xs[-1], _ys[-1], seq['ref']['end'])
    # ax2.axis('equal')  # with sharey=ax1
    ax2.set_aspect('equal', 'datalim')  # without sharey=ax1

    # Floorplan 이미지 출력
    img = Image.open(BytesIO(floor['floorPlanBinary']))
    width = floor['width']
    height = floor['height']
    ratio = floor['metric']['ppm']
    ax1.imshow(img, extent=[0, width / ratio, 0, height / ratio], alpha=0.2, zorder=0)

    # 폴리곤 그리는 부분 (외곽선)
    outer_line = np.array(floor['metric']['layout']['outline'] + [floor['metric']['layout']['outline'][0]])
    ax1.plot(outer_line[:, 0], outer_line[:, 1], lw=1, c='gray', zorder=1)

    # 폴리곤 그리는 부분 (제외 영역)
    for except_area in floor['metric']['layout']['excepts']:
        except_layout = except_area + [except_area[0]]
        inner_line = np.array(except_layout)
        ax1.plot(inner_line[:, 0], inner_line[:, 1], lw=1, c='gray', zorder=1)

    # 랜드마크 그리는 부분
    for color_idx, landmark_type in enumerate(['ENT', 'ST', 'EV', 'EC']):
        for idx, elem in enumerate(floor['metric']['landmark'][landmark_type]):
            ax1.text(elem['coord'][0], elem['coord'][1], f'{elem["id"]}', c='k', fontsize='xx-small', alpha=0.3)
            ax1.scatter(elem['coord'][0], elem['coord'][1], c=f'C{color_idx}', s=3)

    # ax1.set_xlim(right=190)
    # ax1.set_ylim(top=154)

    # Particle Filter 결과 출력
    label = labeled_data
    ax1.plot(label[:, 0], label[:, 1], c='C0', lw=1, zorder=10)
    # ax1.scatter(label[0][0], label[0][1], s=16, c='g', label='start', zorder=11)
    # ax1.scatter(label[-1][0], label[-1][1], marker='x', s=16, c=color, label='end', zorder=12)
    # ax1.legend(loc=4)
    fig.suptitle("%s_(score: %.3f)" % (str(floor['_id']), score))

    # 결과물 디스플레이
    if do_save:
        main_dir = 'results/' + str(floor['buildingId']) + '/'
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
        target_dir = main_dir + '%d/' % floor['floorNumber']
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        plt.savefig(target_dir + '%s_(%.3f).png' % (str(floor['_id']), score), dpi='figure')
    else:
        plt.show()

    plt.close()


def _record_labeling(frame_num, data, scatter_area):
    scatter_area.set_offsets(data[frame_num][:, :2])
    scatter_area.set_array(data[frame_num][:, 2])
    return scatter_area,


def make_gif(floor, records):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    metadata = dict(title='Wav Spectrogram', artist='Matplotlib', comment='')
    # writer = animation.FFMpegWriter(fps=5, metadata=metadata, bitrate=5000)

    fig1 = plt.figure(figsize=(7, 3.5), dpi=200)

    img = Image.open(BytesIO(floor['floorPlanBinary']))
    width, height = floor['width'], floor['height']
    ratio = floor['metric']['ppm']
    plt.imshow(img, extent=[0, width / ratio, 0, height / ratio], alpha=0.2, zorder=0)

    # 폴리곤 그리는 부분 (외곽선)
    outer_line = np.array(floor['metric']['layout']['outline'] + [floor['metric']['layout']['outline'][0]])
    plt.plot(outer_line[:, 0], outer_line[:, 1], lw=1, c='gray', zorder=1)

    # 폴리곤 그리는 부분 (제외 영역)
    for except_area in floor['metric']['layout']['excepts']:
        except_layout = except_area + [except_area[0]]
        inner_line = np.array(except_layout)
        plt.plot(inner_line[:, 0], inner_line[:, 1], lw=1, c='gray', zorder=1)

    # 랜드마크 그리는 부분
    for color_idx, landmark_type in enumerate(['ENT', 'ST', 'EV', 'EC']):
        for idx, elem in enumerate(floor['metric']['landmark'][landmark_type]):
            # plt.text(elem['coord'][0], elem['coord'][1], f'{elem["id"]}', c='k', fontsize='xx-small')
            plt.scatter(elem['coord'][0], elem['coord'][1], c=f'C{color_idx}', s=3)

    s = plt.scatter(records[0][:, 0], records[0][:, 1], c=records[0][:, 2], s=0.2, alpha=0.5)
    plt.colorbar()

    plt.axis('equal')

    scatter_ani = animation.FuncAnimation(fig1, _record_labeling, frames=len(records), fargs=(records, s), blit=True)
    scatter_ani.save('test.gif', writer='imagemagick')
    plt.close()


def load_occ_map(db, floor, res):
    layout = Polygon(floor['metric']['layout']['outline'])
    for area in floor['metric']['layout']['excepts']:
        layout = layout - Polygon(area)

    occ_map = None
    try:
        ogm = floor['metric']['ogm']
        if ogm['res'] == res:
            occ_map = np.array(ogm['value'])
            print('Reuse the occupancy grid map!')
            regen_ogm = False
        else:
            print('The resolution does not matched! Need to regenerate!', file=sys.stderr)
            regen_ogm = True
    except KeyError:
        print('Need to generate!', file=sys.stderr)
        regen_ogm = True

    if regen_ogm or occ_map is None:
        occ_map = rasterize(layout, res)
        db.floor.update({'_id': floor['_id']}, {'$set': {'metric.ogm': {'res': res, 'value': occ_map.tolist()}}})

    return occ_map


class PositioningThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.reinit()

    def reinit(self):  # probably unnecessary
        self.db = None
        self.bldg_id = None
        self.current_floor = None
        self.vt_start = None
        self.start_lonlat_coord = None
        self.last_sent_floor = None

        self.pressure_monitor = None
        self.particle_filter_manager = None

        self.started = False
        self.stop_thread = False
        self.last_timestamp = 0

        try:
            client = MongoClient('143.248.56.66:19191')
            db = client['uplus_demo']
            db['data'].delete_many(dict())
            client.close()
        except Exception as e:
            print(e, file=sys.stderr)

    def do_all(self):
        if self.pressure_monitor is None and self.current_floor is not None:
            self.pressure_monitor = PressureMonitor(self.current_floor)

        res = 0.5
        N = 8000

        data = self.db['data'].find({'timestamp': {'$gt': self.last_timestamp}})
        ts = list()
        ps = list()
        acc = list()
        gyro = list()
        landmarks = list()
        pdr_ts = list()
        pdr_step_lengths = list()
        pdr_headings = list()
        for datum in data:
            ts.append(datum['timestamp'])
            ps.append(datum['pressure'])
            # acc.append([datum['ax'], datum['ay'], datum['az']])
            acc.append([datum['az'], -datum['ay'], datum['ax']])
            # acc.append([datum['az'], -datum['ay'], datum['ax']])
            # gyro.append([datum['gx'], datum['gy'], datum['gz']])
            gyro.append([datum['gx'], -datum['gy'], datum['gz']])
            # gyro.append([datum['gz'], -datum['gy'], datum['gx']])

            # landmarks.append(datum['landmark'])
            # if datum['strideLength'] != 0:
            #     pdr_ts.append(datum['timestamp'])
            #     pdr_step_lengths.append(datum['strideLength'])
            #     pdr_headings.append(datum['heading'])
        if len(ts) == 0:
            return
        acc = np.array(acc)
        gyro = np.array(gyro)
        self.last_timestamp = ts[-1]

        coords = list()

        # update pressure monitor
        self.pressure_monitor.insert_many(ts, ps)

        if abs(self.pressure_monitor.status) == 999:  # drop pdr data when changing floor
            end_idx = bisect_right(ts, self.pressure_monitor.last_floor_t) - 1  # pdr on server
            # end_idx = bisect_right(pdr_ts, self.pressure_monitor.last_floor_t) - 1  # pdr on device
        else:
            end_idx = len(ts) - 1  # pdr on server
            if self.current_floor is None:  # after changing floor
                self.current_floor = self.pressure_monitor.status

            # TODO: utilize device's pdr data???
            # end_idx = len(pdr_ts) - 1  # pdr on device

        estis = None
        if end_idx > -1:
            if self.particle_filter_manager is None and self.current_floor is not None:  # after changing floor
                # load map data
                init_coord, self.floor_doc, occ_map, init_heading = self.load_pf_params(self.current_floor, res=res)
                print('sum of occ_map:', np.sum(occ_map))

                self.vt_start = None
                self.particle_filter_manager = ParticleFilterManager(init_coord, occ_map, res, frequency=100, N=N,
                                                                     init_heading=init_heading)

            estis = self.particle_filter_manager.update(ts[:end_idx + 1], acc[:end_idx + 1, :], gyro[:end_idx + 1, :])
            if estis is not None:
                self.last_coord = estis[-1]

            # TODO: utilize device's pdr data???
            # pf_ts = pdr_ts[:end_idx + 1]

        if abs(self.pressure_monitor.status) == 999 and self.vt_start is None:  # begining of changing floor
            # init vt_start as nearest vt
            vt_ids = list()
            vt_coords = list()
            for landmark_type in ['ST', 'EV', 'EC']:
                for vt in self.floor_doc['metric']['landmark'][landmark_type]:
                    if landmark_type != 'EV':
                        if vt['v_dir'] * self.pressure_monitor.status < 0:  # direction mismatch ST/EC
                            continue
                    vt_ids.append(vt['id'])
                    vt_coords.append(vt['coord'])
            nearest_idx = np.argsort(np.linalg.norm(np.array(vt_coords) - self.last_coord, axis=1))[0]
            self.vt_start = vt_ids[nearest_idx]
            coords.append(vt_coords[nearest_idx])

        if estis is not None:
            coords = estis + coords

        if self.particle_filter_manager is not None and len(coords) > 0:  # step detected
            # convert coord -> lonlat
            floor = self.floor_doc
            lonlat_coords = list(map(lambda x: self.proj(*x, inverse=True),
                                     coords + np.array(floor['metric']['extent'][:2])))

            # send to visualizer
            for lonlat_coord in lonlat_coords:
                try:
                    ret = send_coord(lonlat_coord, bldg_id=self.bldg_id, floor_id=floor['_id'])
                    # print(ret)
                except ConnectionError:
                    print('ConnectionError when send_coord')
                    pass
            self.last_sent_floor = floor['floorNumber']

        if abs(self.pressure_monitor.status) == 999:  # when changing floor
            if self.particle_filter_manager is not None:
                self.record_result(self.floor_doc, int(time.time()))  # record result

            self.particle_filter_manager = None
            self.current_floor = None

            if self.pressure_monitor.estimated_floor is not None:
                if self.last_sent_floor != self.pressure_monitor.estimated_floor:
                    init_coord, floor, _, init_heading = self.load_pf_params(self.pressure_monitor.estimated_floor,
                                                                             res=res)
                    lonlat_coord = self.proj(*(init_coord + np.array(floor['metric']['extent'][:2])), inverse=True)
                    try:
                        ret = send_coord(lonlat_coord, bldg_id=self.bldg_id, floor_id=floor['_id'])
                        # print(ret)
                    except ConnectionError:
                        print('ConnectionError when send_coord')
                        pass
                    self.last_sent_floor = self.pressure_monitor.estimated_floor
        pass

    def load_pf_params(self, current_floor, res):
        client = MongoClient('143.248.56.76:19191')
        db = client['kailos_smart_campus']
        floor = db['floor'].find_one({'buildingId': str(self.bldg_id), 'floorNumber': current_floor})
        occ_map = load_occ_map(db, floor, res)
        self.proj = Proj(floor['metric']['coordSys']['name'])

        if self.vt_start is None:
            # TODO: find closest entrance
            start_coord_in_meter = np.array(self.proj(*self.start_lonlat_coord, inverse=False))
            start_coord = start_coord_in_meter - np.array(floor['metric']['extent'][:2])
            ent_ids = list()
            ent_coords = list()
            ent_headings = list()
            for ent in floor['metric']['landmark']['ENT']:
                ent_ids.append(ent['id'])
                ent_coords.append(ent['coord'])
                ent_headings.append(ent['h_dir'])
            nearest_idx = np.argsort(np.linalg.norm(np.array(ent_coords) - start_coord, axis=1))[0]
            init_coord = ent_coords[nearest_idx]
            initial_heading = ent_headings[nearest_idx]

            print('ent_ids[nearest_idx]:', ent_ids[nearest_idx])
        else:
            # find most-fit vt
            vt_type, vt_floor, vt_id = self.vt_start.split('_')
            vt_end_name = '_'.join([vt_type, str(current_floor), vt_id])

            vt_ids = list()
            vt_coords = list()
            vt_headings = list()
            for vt in floor['metric']['landmark'][vt_type]:
                vt_ids.append(vt['id'])
                vt_coords.append(vt['coord'])
                vt_headings.append(vt['h_dir'])
            target_idx = vt_ids.index(vt_end_name)
            init_coord = vt_coords[target_idx]
            initial_heading = vt_headings[target_idx]

            print('vt_end_name:', vt_end_name, 'init_coord:', init_coord)

        client.close()
        return init_coord, floor, occ_map, initial_heading

    def record_result(self, floor, timestamp, store_only=True):
        pfm = self.particle_filter_manager

        history_x = pfm.history_x
        history_y = pfm.history_y

        try:
            result = np.c_[history_x[:, -1], history_y[:, -1]]  # end points of each particle
        except IndexError:  # stopped too soon
            return

        median = np.median(result, axis=0)
        min_dists = np.linalg.norm(result - median, axis=1)
        best_indices = np.argsort(min_dists)[:10]

        score = np.mean(min_dists)

        path = np.c_[history_x[best_indices[0]], history_y[best_indices[0]]][1:]

        # store result
        with open(f"pf_result_{floor['floorNumber']} - {timestamp}.csv", 'w', encoding='UTF8') as f:
            f.write('yaw,step_length,x,y\n')
            for tup in zip(self.particle_filter_manager.yaws_to_show,
                           self.particle_filter_manager.step_lengths_to_show, path[:, 0], path[:, 1]):
                f.write(','.join(list(map(str, tup))) + '\n')

        # store process
        with open(f"pf_process_{floor['floorNumber']} - {timestamp}.csv", 'w', encoding='UTF8') as f:
            f.write('idx,x,y,prob\n')
            for idx, record in enumerate(pfm.records):
                for elem in record:
                    f.write(','.join(list(map(str, [idx, *elem]))) + '\n')

        if not store_only:
            make_answer_sheet(self.particle_filter_manager.yaws_to_show, path, floor, score,
                              self.particle_filter_manager.step_lengths_to_show, pfm.records)
            make_gif(floor, pfm.records)

    def run(self):
        client = MongoClient('143.248.56.66:19191')
        self.db = client['uplus_demo']

        last_time = time.time()
        while not self.stop_thread:
            cur_time = time.time()
            if cur_time > last_time + 0.2:
                last_time = cur_time
                self.do_all()
                # print(time.time() - cur_time)
            else:
                time.sleep(0.1)

        if self.particle_filter_manager is not None:
            self.record_result(self.floor_doc, int(last_time))  # record result

        # record input data
        # header = "timestamp,strideLength,heading,latitude,longitude,landmark,pressure,ax,ay,az,gx,gy,gz,mx,my,mz," \
        #          "Roll,Pitch".split(',')
        header = "timestamp,pressure,ax,ay,az,gx,gy,gz".split(',')

        with open(f'uplus_demo_data - {int(last_time)}.csv', 'w') as f:
            for elem in self.db['data'].find():
                f.write(','.join([str(elem[name]) for name in header]) + '\n')

        client.close()

        self.reinit()

        IndoorPositioningEngine._thread = PositioningThread()

    def terminate(self):
        self.stop_thread = True


class IndoorPositioningEngine:
    _thread = None
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._thread = PositioningThread()
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        pass

    def set_init_params(self, bldg_id, floor_level, last_coord):
        IndoorPositioningEngine._thread.bldg_id = bldg_id
        IndoorPositioningEngine._thread.current_floor = floor_level
        IndoorPositioningEngine._thread.start_lonlat_coord = last_coord

    def start(self, init_params):
        bldg_id = init_params['bldg_id']
        floor_level = init_params['floor_level']
        last_coord = init_params['last_coord']

        if IndoorPositioningEngine._thread.started:
            IndoorPositioningEngine._thread.terminate()
            time.sleep(1)

        IndoorPositioningEngine._thread.started = True
        self.set_init_params(bldg_id, floor_level, last_coord)
        IndoorPositioningEngine._thread.start()

        return True

    def stop(self):
        if IndoorPositioningEngine._thread.started:
            IndoorPositioningEngine._thread.terminate()
            return True
        return False
