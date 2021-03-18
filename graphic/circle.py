class Circle:
    
    DEFAULT_RADIUS = 1
    
    def __init__(self, centerX, centerY, radius = DEFAULT_RADIUS):
        
        self.centerX = centerX
        self.centerY = centerY
        self.radius = radius

        self.x0 = self.centerX - self.radius
        self.y0 = self.centerY - self.radius
        self.x1 = self.centerX + self.radius
        self.y1 = self.centerY + self.radius
    
    def draw(self, canvas):
        
        canvas.create_oval(
                    self.x0, 
                    self.y0, 
                    self.x1, 
                    self.y1, 
                    fill="#476042"
        )
