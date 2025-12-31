import numpy as np

class DispExt:    

    BUBBLE_RADIUS = 16
    PREPROCESS_CONV_SIZE = 16
    BEST_POINT_CONV_SIZE = 24
    MAX_LIDAR_DIST = 40 
    STRAIGHTS_SPEED = 5.0 # 8.0
    CORNERS_SPEED = 4.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees
    
    def __init__(self, run, conf, init=False, ma_info=[0.0, 0.0]):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None
        self.n_beams = None
        self.max_steer = conf.max_steer
        self.max_speed = run.max_speed
        self.straight_speed = self.STRAIGHTS_SPEED # run.max_speed * 0.7
        self.corner_speed = self.CORNERS_SPEED # self.straight_speed * 0.625
        self.slow_down = 0.85
        self.speed_c, self.steer_c = ma_info
        
        print("ma_info :", ma_info)
    
    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2*np.pi) / len(ranges)
        self.n_beams = len(ranges)
	    # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[self.n_beams//8:-self.n_beams//8])
        # sets each value to the mean over a given window
        pre_conv_size = int(self.PREPROCESS_CONV_SIZE * (1 - 2*self.steer_c))
        proc_ranges = np.convolve(proc_ranges, np.ones(pre_conv_size), 'same') / pre_conv_size
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges==0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        # bp_conv_size = int(self.BEST_POINT_CONV_SIZE * (1 - self.steer_c))
        # averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(bp_conv_size), 'same') / bp_conv_size
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len/2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def plan(self, obs):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = obs["scan"]
        proc_ranges = self.preprocess_lidar(ranges)
        #Find closest point to LiDAR
        closest = proc_ranges.argmin()

        #Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges)-1
        proc_ranges[min_index:max_index] = 0

        #Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        #Find the best point in the gap 
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        #Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.corner_speed
        else: 
            speed = self.straight_speed

        speed = min(speed, self.max_speed) # cap the speed
        speed *= (self.slow_down)
        speed *= (1 + self.speed_c)

        # print('Speed in m/s: {}'.format((speed)))
        # print('Steering angle in degrees: {}'.format((steering_angle/(np.pi/2))*90))
        return steering_angle, speed
