"""@package Test
Defines sumrules::misc::Test::Test.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import getopt
import os
import sys

from .ConfigManager import ConfigManager

class Test(object):
    """ Template for test.
        
        A bundle which makes use of:
        * sumrules::lib::evaluators and sumrules::lib::analytics
            to construct a scheme for compuatations.
        * sumrules::misc::ConfigManager to manage configs of evaluators.
        
        And provides:
        * Ability to load configuration from file or pass from env.
        * Flexibility in upgrading previously written code.
        * Hence, easy to load in batch on cluster with tools::Batch.

        Attributes:
            title: test title to use when saving data.
            configPath: path to config file, to load cfg from.
            config: instance of sumrules::misc::ConfigManager.
            interactive: bool, identifying whether to write logs to stdout.
            outputPath: path to store all test's outputs
            _keylist: list of attributes to be managed by
                sumrules::misc::ConfigManager. See there for more details.            
    """

    def __init__(self, title="unnamed"):
        self.title = title
        self.configPath = title+".conf"
        self.config = ConfigManager()
        self.config.register(self, "TEST")
        self.interactive = False
        self.outputPath = os.path.join("output", title)
        self._keylist = ["title"
                        ,"interactive"
                        ,"outputPath"
                        ,"configPath"]
        self.parseCmd()

    def parseCmd(self):
        """ Set configs from cli args.

            Available options:
            * -c, --config - path to configuration file, to load cfg from.
        """
        try:
            opts, args = getopt.gnu_getopt(sys.argv[1:], "c:", ["config="])
        except getopt.GetoptError:
            return False
        for opt, arg in opts:
            if opt in ("-c", "--config"):
                self.params({"configPath": arg})
        return True

    def params(self, paramdict=None):
        """ Manages dict values for sumrules::misc::ConfigManager::ConfigManager.

            Method allows to update existing values through `paramdict`
            argument, or get current values by calling it without arguments.
            
            Kwargs:
                paramdict: dict of params with new values to update with.

            Examples:

                >>> cfg = Config({"a": 10})
                >>> cfg.params()
                {"a": 10}

                >>> cfg.params({"a": 8})
                >>> cfg.params()
                {"a": 8}
                
        """
        if paramdict is None:
            return {k: getattr(self, k) for k in self._keylist}

        for key, val in paramdict.items():
            if key in self._keylist:
                setattr(self, key, val)
        return True

    def iwrite(self, f, data):
        """ Write to file and if interactive to stdout.
            
            Args:
                f: file descriptor to write into.
                data: string to write into file.
        """
        if self.interactive:
            print(data)
        f.write(data+"\n")

    def path(self, fname):
        """ Return path relatively to working directory.
            
            Transform path relative to `outputDir` to corresponding
            path relative to working directory.

            Args:
                fname: path relative to `outputDir`.

            Returns:
                Path relative to working directory.
        """
        return os.path.join(self.outputPath, fname)

    def run(self):
        """ Basic run implementation.
            
            Only dump current config to file.
        """
        with open(self.outputPath+"/params", "a") as f:
            for entry, val in self.config:
                    self.iwrite(f, "%s=%s" % (entry, str(val)))
        
