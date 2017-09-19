from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

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
        self.fd.flush()
    
    def comment(self, msg):
        self.fd.write("#%s\n" % msg)
        self.fd.flush()

