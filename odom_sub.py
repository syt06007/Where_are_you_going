import rospy
from nav_msgs.msg import Odometry
import csv

ex = Odometry()

ex.pose.pose.position.x
def odom_cb(msg):
    global x, y
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y


def save_to_csv():
    global x_list, y_list
    with open('/home/jun/2gong_turtlebot.csv', 'w', newline='') as csvfile:
        fieldnames = ['x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x_val, y_val in zip(x_list, y_list):
            writer.writerow({'x': x_val, 'y': y_val})
    print("saved")

    
    
if __name__ == '__main__':
    rospy.init_node("odom_sub")
    rospy.Subscriber('/odom', Odometry, odom_cb)
    x_list = []
    y_list = []
    x = None
    y = None
    rospy.wait_for_message('/odom', Odometry)
    rate = rospy.Rate(30)
    while True:
        x_list.append(x)
        y_list.append(y)
        print(x, y, len(x_list))
        if len(x_list) == 6930:
            save_to_csv()
            break
        rate.sleep()

        