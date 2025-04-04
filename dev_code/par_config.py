


class Parameter_config:
    
    def __init__(
            self,
            min_distance_ws : int, 
            dis_tracking_ratio: float,
            snake_speed_max: float,
            snake_speed_min: float,
            kernel_max_radius: float,
            kernel_min_radius: float):
        
        self.min_distance_ws = min_distance_ws
        
        self.dis_tracking_ratio = dis_tracking_ratio

        self.snake_speed_max = snake_speed_max
        self.snake_speed_min = snake_speed_min
        
        self.kernel_max_radius = kernel_max_radius
        self.kernel_min_radius = kernel_min_radius
        

    def get_attribute(self):

        for i in (vars(self)):
            print("{0:10}: {1}".format(i, vars(self)[i]))


def match_par(md, md_list, List_Par):
    
    index_match = md_list.index(md) 
    
    return List_Par[index_match]
    
 
  
############################################################################


#######################################################3
List_Par = []

for i in range(12):
    
    id_object = "p" + '{:02d}'.format(i+1)
    
    id_object = Parameter_config(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    List_Par.append(id_object)

                                         # 1-d   2-d   w-1   w-2
List_Par[0] = Parameter_config(12, 0.59, 0.28, 0.77, 1.13, 0.16)

List_Par[1] = Parameter_config(26, 0.87, 0.35, 0.87, 0.70, 0.48)

List_Par[2] = Parameter_config(24, 0.68, 0.31, 1.10, 2.83, 2.83)

List_Par[3] = Parameter_config(30, 0.95, 0.49, 2.67, 1.04, 1.49)

List_Par[4] = Parameter_config(69, 0.66, 0.22, 1.01, 6.9, 0.59)

List_Par[5] = Parameter_config(45, 0.70, 0.35, 0.69, 0.79, 0.94)

List_Par[6] = Parameter_config(37, 0.78, 0.40, 0.84, 0.79, 0.80)

List_Par[7] = Parameter_config(27, 0.53, 0.25, 0.83, 2.06, 0.97)

List_Par[8] = Parameter_config(42, 0.67, 0.29, 0.68, 0.91, 0.57)

List_Par[9] = Parameter_config(25, 0.70, 0.47, 0.87, 0.65, 1.14)

List_Par[10] = Parameter_config(36, 0.80, 0.61, 1.60, 1.05, 0.36)

List_Par[11] = Parameter_config(56, 0.79, 0.39, 1.04, 0.98, 0.40)
                                           #1-d   2-d   w-1   w-2


######################################################
md_list = []

for count, par_object in enumerate(List_Par):
    
    md_list.append(par_object.min_distance_ws)



