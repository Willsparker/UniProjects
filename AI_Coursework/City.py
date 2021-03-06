from math import sqrt

class City:
    # Initialised with where it is on the 'board'
    def __init__ (self,id,x,y):
        self.pos = (x,y)
        self.id = id

    def cost_to_city(self,next_city):
        cost = 0
        # Uses pythagoris' theorem to find the cost
        diff_x = self.pos[0] - next_city.pos[0]
        diff_y = self.pos[1] - next_city.pos[1]

        cost = sqrt(diff_x**2 + diff_y**2)
        return cost
    
    def toString(self):
        return "City at " + str(self.pos)
