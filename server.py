import asyncio
import random
import websockets
import GestureRecognition
import Constants


async def handler(websocket, path):
    """Handler for each connection from Client"""
    # await to receive data from Client
    data = await websocket.recv()

    # print the data received from the client
    print("Client connected")
    print(f"Received message from client: {data}")

    # initiate the gesture recognition module
    gesture_ocr = GestureRecognition.GestureOcr()
    reply = gesture_ocr.recognize_gesture()
    print(f"Gesture recognized, reply to be sent: {reply}")

    # send recognized gesture to client
    await websocket.send(reply)

# start a server on local host
start_server = websockets.serve(handler, "localhost", int(Constants.OPENCV_SERVER_PORT_NUMBER))

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
