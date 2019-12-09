class FeatureBox:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        return ii[self.y+self.height - 1][self.x+self.width - 1] + ii[self.y][self.x] - (ii[self.y+self.height - 1][self.x]+ii[self.y][self.x+self.width - 1])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "FeatureBox(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)
