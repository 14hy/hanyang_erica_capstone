import rospy
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import SetBool


class MotorPublisher():

    def __init__(self):

        self.pub = rospy.Publisher("motor", Int32MultiArray, queue_size=4)
        self.proxy = rospy.Service("motor_done", SetBool, self.motor_done)
        self.ready = True

    def publish(self, motor_id, direction, step):
        data = Int32MultiArray()

        data.data.append(motor_id)
        data.data.append(direction)
        data.data.append(step)

        rospy.loginfo("Motor: %d, dir: %d, distance: %d" %(motor_id, direction, step))

        self.pub.publish(data)
        self.ready = False

    def is_ready(self):
        return self.ready
        #return True

    def motor_done(self, req):
        if req.data is True:
            rospy.loginfo("Motor done.")
            self.ready = True
            return {"success": True,
                    "message": "True"}
        return None
