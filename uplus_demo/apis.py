import sys
import json
import time
import traceback

from bson import ObjectId
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient

from uplus_demo.positioning import IndoorPositioningEngine
from uplus_demo.misc import log


@csrf_exempt
def upload(request):
    log(f"{request.method} /upload/")
    if request.method != 'POST':
        return JsonResponse({'msg': 'POST only'}, status=404)

    decode_result = request.body.decode('utf-8')
    print(int(time.time() * 1000), decode_result)
    if decode_result == '':
        return JsonResponse(dict())

    try:
        received_json = json.loads(decode_result)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)

        log(''.join(err_msg), console_also=True)

        decode_result_split = decode_result.split('\\r\\n')
        if len(decode_result_split) > 1:
            received_json = json.loads('\\r\\n'.join(decode_result_split[:-1]) + '"}')

    data_str = received_json['data']

    # header = "timestamp, strideLength, heading, latitude, longitude, landmark, pressure, ax, ay, az, gx, gy, gz, mx, " \
    #          "my, mz, Roll, Pitch".split(', ')
    header = "timestamp,heading,pressure,ax,ay,az,gx,gy,gz".split(',')

    # with open('data-device.txt') as f:
    #     lines = f.readlines()
    lines = data_str.split('\n')

    docs = list()
    for line in lines:
        if '+CGNSINF' in line:
            continue

        doc = dict()
        row = line.split(',')
        if len(row) > 1:
            for i in range(len(header)):
                doc[header[i]] = float(row[i])
            docs.append(doc)

    client = MongoClient('143.248.56.66:19191')
    db = client['uplus_demo']

    result = db['data'].insert_many(docs).inserted_ids

    # TODO: upload_success = False if result is bad
    upload_success = True

    client.close()

    if upload_success:
        return JsonResponse(dict(), status=201)

    return JsonResponse(dict())


@csrf_exempt
def start(request):
    log(f"{request.method} /start/")
    if request.method == 'POST':
        decode_result = request.body.decode('utf-8')
        print(decode_result)
        received_json = json.loads(decode_result)

        if 'init_params' in received_json:
            init_params = received_json['init_params']
            bldg_id = ObjectId(init_params['bldg_id'])
            floor_level = init_params['floor_level']
            last_coord = init_params['last_coord']
            IndoorPositioningEngine().set_init_params(bldg_id, floor_level, last_coord)

    IndoorPositioningEngine().start()
    return JsonResponse(dict(), status=201)


@csrf_exempt
def stop(request):
    log(f"{request.method} /stop/")
    IndoorPositioningEngine().stop()
    return JsonResponse(dict(), status=201)
