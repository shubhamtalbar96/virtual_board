from websocket import create_connection
ws = create_connection("ws://localhost:8000/websocket")
print("Sending test message 'Hello, World'...")
ws.send("Hello, World")

print("Sent")
print("Waiting to receive response...")
result = ws.recv()
print(f"Received response {result}")

ws.close()
