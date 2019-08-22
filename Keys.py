import ctypes
import time
import win32api as wapi
import win32con as wcon

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
N = 0x31
ESC = 0x01
ENTER = 0x1C
SPACE = 0x39
LCTRL = 0x1D

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

class Keys():

    def __init__(self):
	    self.keys_list = self.InitializeKeys()

    def PressKey(self, hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def ReleaseKey(self, hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
		
    def KeyCheck(self):
        keys = []
        for key in self.keys_list:
            if wapi.GetAsyncKeyState(ord(key)):
                if key == ' ':
                    key = 'space'
                keys.append(key)
        return keys

    def KeysActionFreeKicksOutput(self, keys):
        '''
        Convert keys to a ...multi-hot... array

        [Left,Right,LowShoot,HighShoot] boolean values.
        '''

        output = [0,0,0,0]
    
        if 'F' in keys:
            output[0] = 1
        elif 'H' in keys:
            output[1] = 1
        elif 'O' in keys:
            output[2] = 1
        elif 'P' in keys:
            output[3] = 1

        return output

    def InitializeKeys(self):
        keys_list = ["\b"]
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
            keys_list.append(char)
        return keys_list

# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

if __name__ == '__main__':
    direct_keys = Keys()
    while (True):
        direct_keys.PressKey(SPACE)
        time.sleep(1) 