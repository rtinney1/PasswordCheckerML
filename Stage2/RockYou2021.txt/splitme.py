filename = "rockyou2021_part{}_{}.txt"

with open("realuniq.lst", encoding="utf-8", errors="ignore", mode="r") as f:
    for line in f:
        try:
            if len(line.strip()) > 7 and line.strip().isascii() and line.strip() != "" and line[0] != "\\" and line[0] != " ":
                firstLetter = line[0]
                writeOut = open(filename.format(firstLetter, len(line.strip())), "a")
                writeOut.write(line.strip())
                writeOut.close()
        except:
            pass
