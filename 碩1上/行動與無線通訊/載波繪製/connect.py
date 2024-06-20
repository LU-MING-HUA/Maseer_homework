import socket

# 設定伺服器IP和埠號
server_ip = '192.168.0.220'
server_port = 12345

# 建立套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 連線到伺服器
client_socket.connect((server_ip, server_port))

# 接收伺服器傳來的資料
received_data = client_socket.recv(1024).decode()

print(f"從伺服器收到的訊息：{received_data}")

# 關閉連線

client_socket.close()
