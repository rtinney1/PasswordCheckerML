import os
import time

if __name__ == "__main__":
    with open("experimentPass.txt", "r") as f:
        for line in f.readlines():
            os.system(f"curl -s -o nul -H \"Content-Type: multipart/form-data\" -X POST http://localhost:5000/createPass/ --form \"password={line}\"")
            time.sleep(5)