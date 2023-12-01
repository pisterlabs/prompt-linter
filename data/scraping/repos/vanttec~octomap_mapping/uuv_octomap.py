#!/usr/bin/env python3
# -- coding: utf-8 --

import math
import time
import rospy
from std_msgs.msg import Float32MultiArray, Int32, String
from geometry_msgs.msg import Pose, PoseStamped,Point, Vector3
from vanttec_uuv.msg import GuidanceWaypoints, obj_detected_list, clusters_center_list
from nav_msgs.msg import Path
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from joblib import Parallel, delayed, parallel_backend
from joblib import wrap_non_picklable_objects
from octomap_server.msg import points_list
import matplotlib.pyplot as plt

# Class Definition
class UUVOctomap:
    def __init__(self):


        # ROS Subscribers
        #rospy.Subscriber("start_mapping",String, self.mapping_callback)
        rospy.Subscriber("start_clustering",String,self.clustering_callback)
        #rospy.Subscriber("octomap/point_list",points_list,self.points_callback)
        rospy.Subscriber("/uuv_simulation/dynamic_model/pose", Pose, self.ins_pose_callback)
        # ROS Publishers
        self.clusters_pub = rospy.Publisher("clusters_center", clusters_center_list, queue_size=10)
        #self.enablepcl = rospy.Publisher("enablepcl", String, queue_size=10)
        #self.updateposition = rospy.Publisher("updateposition", String, queue_size=10)
        # self.mapping =""
        # self.prevmapping=""
        self.clustering =""
        self.prevclustering=""
        self.df = pd.DataFrame()
        self.obs= Vector3()
        self.ned_x = 0
        self.ned_y = 0
        self.ned_z = 0
        self.yaw = 0
    def points_callback(self,msg):
        self.points = []
        try:
            # rospy.logwarn("Foroctomap")
            # rospy.logwarn(msg.lenpoints)
            # rospy.logwarn("Topico")
            # rospy.logwarn(len(msg.points))
            for i in range(len(msg.points)):
                mydict = {'X':msg.points[i].x,'Y':msg.points[i].y,'Z':msg.points[i].z}
                self.df = self.df.append(mydict,ignore_index=True)
        except Exception as e:
            rospy.logwarn(e)
            
    # def mapping_callback(self, msg):
    #     self.mapping = msg.data
    #     if(self.prevmapping!=self.mapping and self.mapping=="Activate"):
    #         self.prevmapping=self.mapping
    def ins_pose_callback(self,pose):
        self.ned_x = pose.position.x
        self.ned_y = pose.position.y
        self.ned_z = pose.position.z
        self.yaw = pose.orientation.z  
    def clustering_callback(self,msg):
        #Recibe muy pocos pointclous creo, hay q esperar la longitud
        self.clustering=msg.data
    @delayed
    @wrap_non_picklable_objects
    def kMeansRes(self,scaled_data, k, alpha_k=0.02):
        '''
        Parameters 
        ----------
        scaled_data: matrix 
            scaled data. rows are samples and columns are features for clustering
        k: int
            current k for applying KMeans
        alpha_k: float
            manually tuned factor that gives penalty to the number of clusters
        Returns 
        -------
        scaled_inertia: float
            scaled inertia value for current k           
        '''
        
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        # fit k-means
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
        return scaled_inertia

    def chooseBestKforKMeansParallel(self,scaled_data, k_range):
        '''
        Parameters 
        ----------
        scaled_data: matrix 
            scaled data. rows are samples and columns are features for clustering
        k_range: list of integers
            k range for applying KMeans
        Returns 
        -------
        best_k: int
            chosen value of k out of the given k range.
            chosen k is k with the minimum scaled inertia value.
        results: pandas DataFrame
            adjusted inertia value for each k in k_range
        '''
        with parallel_backend('threading', n_jobs=2): 
            ans = Parallel(n_jobs=-1,verbose=10)(delayed(self.kMeansRes)(scaled_data, k) for k in k_range)
        ans = list(zip(k_range,ans))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        a= pd.Series(['k','Scaled Inertia']).values.argmin()
        best_k = a
        return best_k, results

    def uuvoctomap(self):
        cluster_list= clusters_center_list()
        cluster_list.clusters.clear()
        rospy.logwarn("uuvoctomap in")

        try:
            #all obstacles
            self.df = pd.read_csv('/ws/src/octomap_mapping/octomap_server/scripts/inputpointclouds.csv')
            rospy.logwarn(len(self.df.index))
            if len(self.df.index)>0:
                X=np.column_stack((self.df['X'], self.df['Y'], self.df['Z']))
                k_range=range(1,4)
                # starting time
                start = time.time()
                best_k, results = self.chooseBestKforKMeansParallel(X, k_range)                
                kmeans= KMeans(n_clusters=best_k).fit(X)
                rospy.logwarn(kmeans)
                rospy.logwarn(kmeans.cluster_centers_.tolist())
                #Ending time
                end = time.time()
                rospy.logwarn("Clusters calculated")
                rospy.logwarn(end-start)
                for i in kmeans.cluster_centers_.tolist():
                    self.obs.x =i[0]
                    self.obs.y = i[1]
                    self.obs.z = self.ned_z
                    cluster_list.clusters.append(self.obs)
                print('Time taken Kmeans/Scikit: ', end - start)  
                #ax = plt.axes(projection='3d')
                #ax.scatter(self.df['X'], self.df['Y'], c=kmeans.labels_, s=0.1)
                # plt.show()
                # centers = kmeans.cluster_centers_
                # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
                # plt.draw()
                # plt.pause(0.00000000001)

                    
                cluster_list.lenclusters = len(kmeans.cluster_centers_.tolist())
                self.clusters_pub.publish(cluster_list)
            else:
                rospy.logwarn("No cluster octomap")
                

        except Exception as e:
            rospy.logwarn(e)


def main():
    rospy.init_node("uuv_octomap", anonymous=False)
    rate = rospy.Rate(20)
    mission = UUVOctomap()
    while not rospy.is_shutdown():
        mission.uuvoctomap()
        rate.sleep()
    rospy.spin()



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:     
        pass