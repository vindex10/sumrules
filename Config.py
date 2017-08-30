class Config(dict):
    def params(self, paramdict=None):
        if paramdict is None:
            return dict(self.items())
        else:
            for key, value in paramdict.items():
                if key in self.keys():
                    self.update({key: value})

