"""
测试HTTP服务器
用于接收心率和呼吸数据的测试服务器

运行方式：
    python test_server.py

然后在 main.py 中配置：
    HEART_UPLOAD_URL = "http://localhost:8000/api/heart"
    BREATH_UPLOAD_URL = "http://localhost:8000/api/breath"
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataHandler(BaseHTTPRequestHandler):
    """处理心率和呼吸数据的HTTP请求处理器"""
    
    def do_POST(self):
        """处理POST请求"""
        content_length = int(self.headers.get('Content-Length', 0))
        
        if content_length == 0:
            self.send_error(400, "Empty request body")
            return
        
        try:
            # 读取请求体
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            print(data)
            
            # 根据路径处理不同类型的数据
            if self.path == '/api/heart':
                self._handle_heart_data(data)
            elif self.path == '/api/breath':
                self._handle_breath_data(data)
            else:
                self.send_error(404, f"Unknown endpoint: {self.path}")
                return
            
            # 发送成功响应
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # 计算接收数据条数
            if isinstance(data, dict) and 'data' in data:
                received_count = len(data['data'])
            elif isinstance(data, list):
                received_count = len(data)
            else:
                received_count = 1
            
            response = {
                'status': 'success',
                'received': received_count,
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误: {e}")
            self.send_error(400, "Invalid JSON")
        except Exception as e:
            logging.error(f"处理请求时出错: {e}")
            self.send_error(500, "Internal server error")
    
    def _handle_heart_data(self, data):
        """处理心率数据"""
        # 支持新格式: {"type": "on_body", "data": [...]}
        if isinstance(data, dict):
            data_type = data.get('type', 'unknown')
            data_list = data.get('data', [])
            logging.info(f"[心率] 类型: {data_type} | 收到 {len(data_list)} 条数据")
            for item in data_list:
                heart_value = item.get('heart_value')
                timestamp = item.get('timestamp')
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                logging.info(f"  心率: {heart_value} bpm | 时间: {dt.strftime('%H:%M:%S.%f')[:-3]}")
        elif isinstance(data, list):
            # 兼容旧格式
            logging.info(f"[心率] 收到 {len(data)} 条数据 (旧格式)")
            for item in data:
                heart_value = item.get('heart_value')
                timestamp = item.get('timestamp')
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                logging.info(f"  心率: {heart_value} bpm | 时间: {dt.strftime('%H:%M:%S.%f')[:-3]}")
    
    def _handle_breath_data(self, data):
        """处理呼吸数据"""
        # 支持新格式: {"type": "on_body", "data": [...]}
        if isinstance(data, dict):
            data_type = data.get('type', 'unknown')
            data_list = data.get('data', [])
            logging.info(f"[呼吸] 类型: {data_type} | 收到 {len(data_list)} 条数据")
            for item in data_list:
                breath_value = item.get('breath_value')
                timestamp = item.get('timestamp')
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                logging.info(f"  呼吸: {breath_value:.1f} bpm | 时间: {dt.strftime('%H:%M:%S.%f')[:-3]}")
        elif isinstance(data, list):
            # 兼容旧格式
            logging.info(f"[呼吸] 收到 {len(data)} 条数据 (旧格式)")
            for item in data:
                breath_value = item.get('breath_value')
                timestamp = item.get('timestamp')
                dt = datetime.fromtimestamp(timestamp / 1000.0)
                logging.info(f"  呼吸: {breath_value:.1f} bpm | 时间: {dt.strftime('%H:%M:%S.%f')[:-3]}")
    
    def log_message(self, format, *args):
        """禁用默认的日志输出，使用自定义日志"""
        pass


def run_server(host='localhost', port=8000):
    """运行测试服务器"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, DataHandler)
    
    logging.info("=" * 60)
    logging.info("测试服务器启动")
    logging.info(f"监听地址: http://{host}:{port}")
    logging.info(f"心率接口: http://{host}:{port}/api/heart")
    logging.info(f"呼吸接口: http://{host}:{port}/api/breath")
    logging.info("=" * 60)
    logging.info("等待接收数据... (按 Ctrl+C 停止)")
    logging.info("")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("\n服务器已停止")
        httpd.shutdown()


if __name__ == '__main__':
    import sys
    
    # 从命令行参数获取端口（可选）
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"无效的端口号: {sys.argv[1]}")
            sys.exit(1)
    
    run_server(port=port)
