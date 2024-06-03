


class Parameter_config:
    
    def __init__(
            self,
            min_distance_ws : int, 
            dis_tracking_ratio: float,
            distance_tracking_max: float,
            distance_tracking_min: float,
            distance_tracking_avg: float,
            thresh_distance_ratio: float,
            snake_speed_max: float,
            snake_speed_min: float,
            kernel_max_radius: float,
            kernel_min_radius: float):
        
        self.min_distance_ws = min_distance_ws
        
        self.dis_tracking_ratio = dis_tracking_ratio
        
        self.distance_tracking_max = distance_tracking_max
        self.distance_tracking_min = distance_tracking_min
        self.distance_tracking_avg = distance_tracking_avg
        self.thresh_distance_ratio = thresh_distance_ratio
        
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


#######################################################
List_Par = []

for i in range(12):
    
    id_object = "p" + '{:02d}'.format(i+1)
    
    id_object = Parameter_config(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    List_Par.append(id_object)

                                         #max  min   d     stem  1-d   2-d   w-1   w-2
List_Par[0] = Parameter_config(34, 0.58, 1.71, 0.63, 1.97, 1.65, 0.64, 1.68, 0.34, 0.26)

List_Par[1] = Parameter_config(48, 0.89, 1.05, 0.16, 0.98, 1.25, 0.35, 0.87, 4.08, 1.37)

List_Par[2] = Parameter_config(35, 0.68, 0.88, 0.52, 1.19, 1.68, 0.35, 1.83, 4.21, 6.87)

List_Par[3] = Parameter_config(43, 0.95, 1.50, 0.51, 2.34, 1.24, 0.51, 2.75, 1.21, 0.40)

List_Par[4] = Parameter_config(65, 0.78, 0.93, 0.32, 1.27, 2.24, 0.30, 1.79, 1.53, 0.31)

List_Par[5] = Parameter_config(55, 0.80, 0.91, 0.31, 0.70, 0.50, 11.20, 25.93, 0.54, 0.38)

List_Par[6] = Parameter_config(59, 0.78, 1.01, 0.52, 1.51, 1.55, 0.71, 1.45, 1.44, 2.56)

List_Par[7] = Parameter_config(27, 0.57, 1.06, 0.65, 1.71, 1.61, 0.28, 0.94, 1.16, 0.48)

List_Par[8] = Parameter_config(40, 0.72, 1.07, 0.46, 0.98, 1.05, 0.35, 0.84, 2.16, 0.71)

List_Par[9] = Parameter_config(26, 0.70, 1.63, 0.42, 1.32, 1.57, 0.47, 0.82, 1.97, 1.00)

List_Par[10] = Parameter_config(58, 0.87, 1.50, 0.56, 1.96, 2.52, 0.87, 2.12, 0.80, 0.14)

List_Par[11] = Parameter_config(56, 0.80, 0.54, 0.48, 0.70, 0.63, 0.41, 1.32, 0.70, 0.37)
                                          #max  min   d     stem  1-d   2-d   w-1   w-2



######################################################
md_list = []

for count, par_object in enumerate(List_Par):
    
    md_list.append(par_object.min_distance_ws)



