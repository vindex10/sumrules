class Monitor(object):
    def __init__(self, filename, header=None):
        self.fd = open(filename, "a")
        if header is not None:
            self.comment(header)

    def push(self, data, msg=""):
        out = ""
        msg = "" if len(msg) == 0 else " # "+msg
        if len(data.shape) == 1:
            out += " ".join(map(str, data))+msg+"\n"
        else:
            for entry in data:
                out += " ".join(map(str, entry))+msg+"\n"
        self.fd.write(out)
    
    def comment(self, msg):
        self.fd.write("#%s\n" % msg)

