import socket

# 設定伺服器IP和埠號
server_ip = '192.168.0.220'
server_port = 12345

# 建立套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 綁定IP和埠號
server_socket.bind((server_ip, server_port))

# 監聽來自客戶端的連線
server_socket.listen()

print(f"等待來自客戶端的連線...")

# 接受客戶端連線
client_socket, client_address = server_socket.accept()

print(f"已連線到客戶端：{client_address}")

# 傳送資料給客戶端
message_to_send = "Hello from server!"
client_socket.send(message_to_send.encode())

# 關閉連線
client_socket.close()
server_socket.close()
