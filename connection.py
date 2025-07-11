# context_socket_server.py
import socket
import os
import time
from context_engine import ContextEngine

SOCKET_PATH = "/tmp/context.sock"

if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

ctx = ContextEngine()


with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
    server.bind(SOCKET_PATH)
    server.listen(1)
    print("🧠 Context Engine socket server running at", SOCKET_PATH)

    while True:
        conn, _ = server.accept()
        
        print(f"⚡ Client connected.")
        with conn:
            try:
                data = conn.recv(4096).decode().strip()
                if not data:
                    continue
                if data == 'check':
                    conn.sendall("yolo".encode())
                else:
                    print(f"🔍 Query: {data}")
                    results = ctx.retrieve(data)
                    print(results)
                    ctx.add_context(data)
                    if results:
                        for text, score in results:
                            conn.sendall(text.encode())
                            print(f"→ \"{text}\" (distance: {score:.4f})")
                        # Send the best match
                        response = results[0][0]
                        print("what is in res:", response)
                        conn.sendall(response.encode())
            except Exception as e:
                print("Error:", e)
                try:
                    conn.sendall("Error occurred".encode())
                except:
                    pass
