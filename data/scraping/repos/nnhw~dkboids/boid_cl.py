import dronekit
import geopy.distance
import math
import guidance
import buddy_cl


class Boid(dronekit.Vehicle):
    def __init__(self, handler, id):
        super(Boid, self).__init__(handler)

        self._id = id
        self._flight_level = 0
        self._poi = dronekit.LocationGlobalRelative(0, 0, 0)
        self._groundspeed = 0

        self._global_poi = dronekit.LocationGlobalRelative(0, 0, 0)

        self._follow_target_id = 0

        self._swarming = False

        self._buddies = (buddy_cl.Buddy(), buddy_cl.Buddy(), buddy_cl.Buddy())

    def _calculate_distance_fine(self, lat, lon, alt):
        me = (self.location.global_relative_frame.lat,
              self.location.global_relative_frame.lon)
        target = (lat, lon)
        horizontal_distance = geopy.distance.geodesic(me, target).m
        vertical_distance = abs(self.location.global_relative_frame.alt - alt)
        distance = math.sqrt(pow(horizontal_distance, 2) +
                             pow(vertical_distance, 2))
        return distance, horizontal_distance, vertical_distance

    def input_data(self, l_data):
        self.analyze_data(l_data)
        if self._swarming is True:
            self.implement_corrections()
            self.goto_poi()

    def analyze_data(self, l_data):
        if l_data['id'] == self._id:
            return

        if l_data['id'] == 200:
            self._global_poi = dronekit.LocationGlobalRelative(
                l_data['lat'], l_data['lon'], l_data['alt'])
            self.mode = dronekit.VehicleMode("GUIDED")
            print("new poi set")
            return

        if self._follow_target_id != 0:
            if l_data['id'] == self._follow_target_id:
                self._global_poi = dronekit.LocationGlobalRelative(
                    l_data['lat'], l_data['lon'], l_data['alt'])

        distance = self._calculate_distance_fine(
            l_data['lat'], l_data['lon'], l_data['alt'])[0]

        new_id = l_data['id'] != self._buddies[0].id and l_data['id'] != self._buddies[1].id and l_data['id'] != self._buddies[2].id
        for n in range(len(self._buddies)):
            if distance < self._buddies[n].distance and new_id is True:
                self._buddies[n].id = l_data['id']
        self.update_buddy_data(l_data, distance)

    def update_buddy_data(self, l_data, distance):
        if l_data['id'] == self._buddies[0].id:
            self._buddies[0].flight_level = l_data['flight_level']
            self._buddies[0].location.lat = l_data['lat']
            self._buddies[0].location.lon = l_data['lon']
            self._buddies[0].location.alt = l_data['alt']
            self._buddies[0].groundspeed = l_data['groundspeed']
            self._buddies[0].distance = distance
        elif l_data['id'] == self._buddies[1].id:
            self._buddies[1].flight_level = l_data['flight_level']
            self._buddies[1].location.lat = l_data['lat']
            self._buddies[1].location.lon = l_data['lon']
            self._buddies[1].location.alt = l_data['alt']
            self._buddies[1].groundspeed = l_data['groundspeed']
            self._buddies[1].distance = distance
        elif l_data['id'] == self._buddies[2].id:
            self._buddies[2].flight_level = l_data['flight_level']
            self._buddies[2].location.lat = l_data['lat']
            self._buddies[2].location.lon = l_data['lon']
            self._buddies[2].location.alt = l_data['alt']
            self._buddies[2].groundspeed = l_data['groundspeed']
            self._buddies[2].distance = distance

    def get_buddy(self, n):
        return self._buddies[n-1].id, self._buddies[n-1].flight_level, self._buddies[n-1].location.lat, self._buddies[n-1].location.lon, self._buddies[n-1].location.alt, self._buddies[n-1].distance, self._buddies[n-1].groundspeed

    def separation(self):
        NotImplemented

    def alignment(self):
        buddies_average_groundspeed = 0
        for bd in self._buddies:
            buddies_average_groundspeed += bd.groundspeed
        buddies_average_groundspeed = buddies_average_groundspeed/3
        return buddies_average_groundspeed

    def cohesion(self):
        lat_sum = 0
        lon_sum = 0
        alt_sum = 0
        for bd in self._buddies:
            lat_sum += bd.location.lat
            lon_sum += bd.location.lon
            alt_sum += bd.location.alt
        lat_mean = lat_sum/3
        lon_mean = lon_sum/3
        alt_mean = alt_sum/3
        buddies_center = dronekit.LocationGlobalRelative(
            lat_mean, lon_mean, alt_mean)
        distance = self._calculate_distance_fine(
            buddies_center.lat, buddies_center.lon, buddies_center.alt)
        return buddies_center, distance

    def implement_corrections(self):
        cohesion_point = self.cohesion()[0]
        cohesion_distance = self.cohesion()[1]

        correction_poi = cohesion_point

        if correction_poi.lat == 0:
            self._poi = self._global_poi
        else:
            self._poi.lat = (self._global_poi.lat + correction_poi.lat)/2
            self._poi.lon = (self._global_poi.lon + correction_poi.lon)/2
            self._poi.alt = self._flight_level

        alignment_speed = self.alignment()
        if alignment_speed != 0:
            self.groundspeed = alignment_speed

    def goto_poi(self):
        self.simple_goto(self._poi)
