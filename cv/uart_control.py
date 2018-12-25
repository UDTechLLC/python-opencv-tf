import serial
import time

# UART_PORT = '/dev/tty.usbserial-1410' # MAC Sergey
# UART_PORT = '/dev/tty.wchusbserial1420' # Mac Miro
UART_PORT = '/dev/ttyUSB0'


class UartControl(object):
    def __init__(self, debug=False):
        self._debug = debug

        self._port = ''
        try:
            self._port = serial.Serial(
                port=UART_PORT,
                baudrate=9600,
                timeout=1
            )
            out = self._port.read_until()
            if self._debug:
                print('Receiving...' + str(out))
        except (OSError, serial.SerialException):
            pass

        if self._port == '':
            raise OSError('Cannot find port uart device')

    def set_speed(self, value):
        command = "speed=" + str(value) + "\n"
        self._port.write(command.encode("ascii"))
        out = self._port.read_until()
        if self._debug:
            print('Receiving...' + str(out))

    def start_step(self, value):
        command = "step=" + str(value) + "\n"
        self._port.write(command.encode("ascii"))
        out = self._port.read_until()
        if self._debug:
            print('Receiving...' + str(out))

    def enable_led(self, value):
        command = "led=" + str(value) + "\n"
        self._port.write(command.encode("ascii"))
        out = self._port.read_until()
        if self._debug:
            print('Receiving...' + str(out))

    def disable_led(self):
        self._port.write("led=0\n".encode("ascii"))
        out = self._port.read_until()
        if self._debug:
            print('Receiving...' + str(out))


def testUart():
    uart = UartControl(debug=True)
    uart.enable_led(50)
    time.sleep(5)
    uart.disable_led()
    time.sleep(5)
    uart.start_step(3200)
    time.sleep(10)
    uart.start_step(3200)
    time.sleep(10)


if __name__ == '__main__':
    testUart()
