import requests
import json
import serial
try:
    import thread
except ImportError:
    import _thread as thread
import websocket
import random
import time

ser = 0

def on_message(ws, message):
  print(message)
  result = json.loads(message)
  print(result["data"])
  if result["data"] == "1":
    while 1:
      response = ser.readall()

      result = json.loads(response)
      print("User BMP: %d" % result["bmp"])
      print("Temperature: %f" % result["temperature"])

      url = "https://ml-sheep.azurewebsites.net/predict"

      payload = {
        "Temp": result["epoch"],
        "HR" : result["bmp"]
      }
      headers = {
        'Content-Type': 'application/json'
      }


      response = requests.request("POST", url, headers=headers, data=json.dumps(payload))




      print("User Status is %d" % response.text)

      url = "http://159.203.191.85/api/record/store"
      payload = {
        "user_id": 1,
        "bmp": result["bmp"],
        "status": 0,
        "temperature": result["temperature"]
      }
      body = json.dumps(payload)
      headers = {
        'Content-Type': 'application/json'
      }
      response = requests.request("POST", url, headers=headers, data=body)
      print(response.text)
      time.sleep(5)



  ##result = json.loads(message)


def on_error(ws, error):
  print(error)


def on_close(ws):
  print("### closed ###")


def on_open(ws):
  body = {
    "event": "login",
    "data": {
      "uid": 1
    }
  }
  ws.send(json.dumps(body))
  ser = serial.Serial('/dev/ttyACM1', 9600,timeout=1);

websocket.enableTrace(True)
ws = websocket.WebSocketApp("ws://159.203.191.85:1215",
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws.run_forever()








