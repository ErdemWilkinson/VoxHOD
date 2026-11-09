import serial, sys, time
s = serial.Serial("COM3", 115200, timeout=1)
s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False  # reset the board
t0 = time.time(); buf = b""
while time.time() - t0 < float(sys.argv[1]):
    d = s.read(4096)
    if d:
        buf += d
        if b"DONE" in buf: break
open("serial.log", "wb").write(buf)
print(buf.decode("utf-8", "replace"))
