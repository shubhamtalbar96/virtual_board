import asyncio
import random
import websockets
import GestureRecognition


# create handler for each connection
async def handler(websocket, path):
    data = await websocket.recv()

    print("A client just connected")
    print(f"Received message {data}")

    gesture_ocr = GestureRecognition.GestureOcr()
    reply = gesture_ocr.recognize_gesture()
    print(f"A client just connected, reply {reply} sent")

    await websocket.send(reply)


start_server = websockets.serve(handler, "localhost", 8000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
