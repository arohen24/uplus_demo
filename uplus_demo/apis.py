import sys
import json
import time
import datetime
import traceback

from bson import ObjectId
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from uplus_demo.positioning import IndoorPositioningEngine


@csrf_exempt
def upload(request):
    if request.method != 'POST':
        return JsonResponse({'msg': 'POST only'}, status=404)

    decode_result = request.body.decode('utf-8')
    print(int(time.time() * 1000), decode_result)
    if decode_result == '':
        return JsonResponse(dict())

    try:
        received_json = json.loads(decode_result)
    except Exception as e:
        with open('log.txt', 'a') as f:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)

            time_str = datetime.datetime.fromtimestamp(int(time.time())).strftime('[%d/%b/%Y %H:%M:%S]')
            print(f"{time_str} {''.join(err_msg)}", file=f)
            print(f"{time_str} {''.join(err_msg)}")
        decode_result_split = decode_result.split('\\r\\n')
        if len(decode_result_split) > 1:
            received_json = json.loads('\\r\\n'.join(decode_result_split[:-1]) + '"}')

    data_str = received_json['data']

    # header = "timestamp, strideLength, heading, latitude, longitude, landmark, pressure, ax, ay, az, gx, gy, gz, mx, " \
    #          "my, mz, Roll, Pitch".split(', ')
    header = "timestamp,pressure,ax,ay,az,gx,gy,gz".split(',')

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
    if request.method == 'POST':
        decode_result = request.body.decode('utf-8')
        print(decode_result)
        received_json = json.loads(decode_result)

        IndoorPositioningEngine().start(init_params=received_json['init_params'])
        return JsonResponse(dict(), status=201)

    return JsonResponse(dict())


@csrf_exempt
def stop(request):
    IndoorPositioningEngine().stop()
    return JsonResponse(dict(), status=201)
