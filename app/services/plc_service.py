
import struct
import snap7

class PLCService():
    def _connect(self):
        try:
            print('[PLC] Đang kết nối tới PLC...')
            self.client = snap7.client.Client().connect(self.ip, self.rack, self.slot)
            if self.client.get_connected():
                print('[PLC] Kết nối thành công!')
            else: 
                raise ConnectionError('Lỗi khi kết nối đến PLC')
        except Exception as e:
            print(e)
        
    def __init__(self):
        self.rack = 0
        self.slot = 1
        self.ip = '192.168.0.1'
        self.client = None 

        self.db_number = 1
        self.db_offset_command = 2
        
        print('===== Khởi tạo dịch vụ PLC ====')
        print('- ip: ', self.ip)
        print('- rack: ', self.rack)
        print('- slot: ', self.slot)

        self._connect()

    def write_command(self, value):
        data = struct.pack(">h", int(value))
        self.client.db_write(self.db_number, self.db_offset_command, data)
        print(f"[PLC] Gửi Command={value} đến DB{self.db_number}.Byte{self.db_offset_command}")

    def read_value(self, byte_index, bit_index):
        data = self.client.db_read(self.db_number, byte_index, 1)
        return bool(data[0] & (1 << bit_index))
    
    def change_ip(self,ip):
        self.ip = ip

    def change_rack(self, rack):
        self.rack = rack

    def change_slot(self, slot):
        self.slot = slot

    def reconnect(self):
        self._connect()