# 串口通信工作线程：负责与嵌入式设备的二进制帧协议收发与解析
import struct, time, threading
import serial
import serial.tools.list_ports
from PySide6.QtCore import QObject, Signal

FRAME_START = 0x7E  # 帧起始标记
FRAME_END = 0x7F  # 帧结束标记
ESCAPE = 0x7D  # 转义字符
ESCAPE_XOR = 0x20  # 转义异或值

TYPE_MEL = 0x01  # MEL频谱数据帧
TYPE_STATUS = 0x85  # 设备状态帧
TYPE_ACK = 0x81  # 应答帧
TYPE_LOG = 0x04  # 日志帧
CMD_QUERY_STATUS = 0x05  # 查询状态命令
CMD_SWITCH_MODEL = 0x04  # 切换模型命令
CMD_RESET_MODEL = 0x06  # 重置模型命令
CMD_SET_MEL_OUTPUT = 0x07  # 设置MEL输出开关命令
CMD_QUERY_MODEL_INFO = 0x08  # 查询模型信息命令
MEL_FILTERS = 20  # MEL滤波器组数
MEL_FRAME_BYTES = MEL_FILTERS * 4  # 单帧MEL数据字节数(20个float)

def _crc8(data):  # CRC-8校验计算
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8C & (-(crc & 1)))
    return crc

def _crc32(data):  # CRC-32校验计算
    c = 0xFFFFFFFF
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ (0xEDB88320 & (-(c & 1)))
    return (~c) & 0xFFFFFFFF

def _escape_byte(b):
    if b in (FRAME_START, FRAME_END, ESCAPE):
        return bytes([ESCAPE, b ^ ESCAPE_XOR])
    return bytes([b])

def build_cmd_frame(cmd, payload=b''):
    """构建命令帧：头(0xAA55) + 命令码 + 长度 + 载荷 + CRC32"""
    length = len(payload)
    frame = bytes([0xAA, 0x55, cmd, length & 0xFF]) + payload
    crc = struct.pack('<I', _crc32(frame))
    return frame + crc

def list_serial_ports():
    """枚举系统可用的串口设备"""
    return [p.device for p in serial.tools.list_ports.comports()]

class SerialWorker(QObject):
    """串口通信工作对象：在独立线程中接收数据，通过Qt信号向上层分发解析结果"""
    mel_received = Signal(object)  # MEL频谱数据
    status_received = Signal(int, int, int, int, int, float)  # 设备状态
    ack_received = Signal(int, int, object)  # 命令应答
    log_received = Signal(str)  # 文本日志
    model_info_received = Signal(int, int, list, list)  # 模型信息(来源, 类数, 名称列表, 映射)
    connection_changed = Signal(bool, str)  # 连接状态变化

    def __init__(self):
        super().__init__()
        self.ser = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self.ser is not None and self.ser.is_open

    def connect_port(self, port, baud=921600):
        """连接串口并启动后台接收线程"""
        self.disconnect_port()
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self._running = True
            self._thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._thread.start()
            self.connection_changed.emit(True, port)
            return True
        except Exception as e:
            self.connection_changed.emit(False, str(e))
            return False

    def disconnect_port(self):
        """断开串口连接，停止接收线程并释放资源"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.connection_changed.emit(False, "")

    def send_command(self, cmd, payload=b''):
        """线程安全地发送命令帧到设备"""
        with self._lock:
            if self.ser and self.ser.is_open:
                frame = build_cmd_frame(cmd, payload)
                self.ser.write(frame)
                self.ser.flush()

    def query_status(self):
        self.send_command(CMD_QUERY_STATUS)

    def switch_model(self, slot):
        self.send_command(CMD_SWITCH_MODEL, bytes([slot]))

    def set_mel_output(self, enabled):
        self.send_command(CMD_SET_MEL_OUTPUT, bytes([1 if enabled else 0]))

    def query_model_info(self):
        self.send_command(CMD_QUERY_MODEL_INFO)

    def _rx_loop(self):
        """串口接收主循环：解析二进制帧和文本行"""
        buf = b''
        while self._running:
            try:
                data = self.ser.read(512)
                if not data:
                    continue
                buf += data
                while len(buf) >= 4:
                    # Try binary frame: 0x7E...0x7F
                    idx = buf.find(FRAME_START)
                    if idx == 0:
                        end_idx = buf.find(FRAME_END)
                        if end_idx < 0:
                            if len(buf) > 256:
                                buf = buf[1:]
                            break
                        raw = buf[1:end_idx]
                        buf = buf[end_idx + 1:]
                        unescaped = self._unescape(raw)
                        if len(unescaped) >= 3:
                            ftype = unescaped[0]
                            flen = unescaped[1]
                            if len(unescaped) >= 2 + flen:
                                payload = unescaped[2:2 + flen]
                                self._dispatch(ftype, payload)
                        continue
                    # Try text line: look for \n
                    nl = buf.find(b'\n')
                    if nl >= 0:
                        line = buf[:nl].decode('utf-8', errors='replace').strip()
                        buf = buf[nl + 1:]
                        if line:
                            self.log_received.emit(line)
                        continue
                    # Not enough data yet
                    if len(buf) > 256:
                        buf = buf[-64:]
                    break
            except Exception:
                pass

    def _unescape(self, data):
        """对帧数据进行反转义处理"""
        result = bytearray()
        i = 0
        while i < len(data):
            if data[i] == ESCAPE and i + 1 < len(data):
                result.append(data[i + 1] ^ ESCAPE_XOR)
                i += 2
            else:
                result.append(data[i])
                i += 1
        return bytes(result)

    def _dispatch(self, ftype, payload):
        """根据帧类型分发数据到对应的Qt信号"""
        if ftype == TYPE_MEL and len(payload) == MEL_FRAME_BYTES:
            mel = list(struct.unpack(f'{MEL_FILTERS}f', payload))
            self.mel_received.emit(mel)
        elif ftype == TYPE_STATUS and len(payload) >= 10:
            import struct as _st
            distance = _st.unpack_from('<f', payload, 6)[0]
            self.status_received.emit(
                payload[0], payload[1], payload[2],
                payload[3], payload[4], distance
            )
        elif ftype in (0x81, 0x82, 0x84, 0x86, 0x87):
            self.ack_received.emit(ftype, payload[0] if payload else 0, payload)
        elif ftype == 0x88 and len(payload) >= 3:
            src = payload[0]
            nc = payload[1]
            state = payload[2]
            name_len = 24
            expected_len = 3 + nc * name_len + nc
            names = []
            maps = []
            if len(payload) >= expected_len:
                for i in range(nc):
                    off = 3 + i * name_len
                    name = bytes(payload[off:off+name_len]).split(b'\x00')[0].decode('utf-8', errors='replace')
                    names.append(name)
                map_off = 3 + nc * name_len
                for i in range(nc):
                    if map_off + i < len(payload):
                        maps.append(payload[map_off + i])
            else:
                name_len = 8
                for i in range(nc):
                    off = 3 + i * name_len
                    if off + name_len <= len(payload):
                        name = bytes(payload[off:off+name_len]).split(b'\x00')[0].decode('utf-8', errors='replace')
                        names.append(name)
                map_off = 3 + nc * name_len
                for i in range(nc):
                    if map_off + i < len(payload):
                        maps.append(payload[map_off + i])
            self.model_info_received.emit(src, nc, names, maps)
        elif ftype == TYPE_LOG:
            try:
                msg = bytes(payload).decode('utf-8', errors='replace').strip()
                self.log_received.emit(msg)
            except:
                pass
