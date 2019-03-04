import rospy
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import SetBool


class MotorPublisher():

    def __init__(self):

        self.pub = rospy.Publisher("motor", Int32MultiArray, queue_size=4)
        self.proxy = rospy.Service("motor_done", SetBool, self.motor_done)
        self.ready = True

    def publish(self, motor_id, direction, step):
        data = []

        data.append(motor_id)
        data.append(direction)
        data.append(step)

        rospy.loginfo("Motor: %d, dir: %d, distance: %d" %(motor_id, direction, step))

        self.pub.publish(data)
        self.ready = False

    def is_ready(self):
        return self.ready

    def motor_done(self, req):
        if req.data is True:
            self.ready = True
            return {"success": True,
                    "message": "True"}
        return None
