"""@package Monitor
Stores class sumrules::misc::Monitor::Monitor, needed to log
an output of itermediate calculations.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

class Monitor(object):
    """ Monitors log data fed to them.
        
        Attributes:
            fd: file descriptor to write to.
    """

    def __init__(self, filename, header=None):
        """ Init.
            Initialize descriptor from `filename`
            and write `header` to file.

            Args:
                filename: name of file to write into.

            Kwargs:
                header: optional. first line to write into file, usually definition
                    of columns.
        """
        self.fd = open(filename, "a")
        if header is not None:
            self.comment(header)

    def push(self, data, msg=""):
        """ Send tabled data to file.
            
            Write row by row from data to file opened in `fd`.

            Args:
                data: NumPy array of rows to be written.

            Kwargs:
                msg: optional. Use it to comment on some lines.

            Returns:
                Nothing.
        """
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
        """ Insert a comment into log.
            
            Args:
                msg: a string to write.

            Returns:
                Nothing.
        """
        self.fd.write("#%s\n" % msg)
        self.fd.flush()

