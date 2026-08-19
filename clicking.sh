uvx --with pynput python -c "from pynput.mouse import Controller, Button; import time; mouse=Controller(); exec('while True: mouse.click(Button.left); time.sleep(5)')"
