import numpy as np
import math

def euler2transf(x, y, z, roll, pitch, yaw):

    # translation
    trans = np.transpose(np.array([[x,y,z]]))

    # rotate roll
    roll_transf = np.identity(3)
    roll_transf[1,1] = math.cos(roll)
    roll_transf[1,2] = -math.sin(roll)
    roll_transf[2,1] = math.sin(roll)
    roll_transf[2,2] = math.cos(roll)
    
    # rotate pitch
    pitch_transf = np.identity(3)
    roll_transf[0,0] = math.cos(pitch)
    roll_transf[0,2] = math.sin(pitch)
    roll_transf[2,0] = -math.sin(pitch)
    roll_transf[2,2] = math.cos(pitch)

    # rotate yaw
    yaw_transf = np.identity(3)
    yaw_transf[0,0] = math.cos(yaw)
    yaw_transf[0,1] = -math.sin(yaw)
    yaw_transf[1,0] = math.sin(yaw)
    yaw_transf[1,1] = math.cos(yaw)

    # rotational matrix
    rotational = np.matmul(np.matmul(roll_transf,pitch_transf),yaw_transf)

    # transformation matrix
    result = np.block([[rotational,trans],
                       [np.zeros([1,3]),1]])

    return result

def invtransf(transf):

    rotational_t = np.transpose(transf[0:3,0:3])
    trans = np.transpose([transf[0:3,3]])
    
    result = np.block([[rotational_t,np.matmul(-rotational_t,trans)],
                       [np.zeros([1,3]),1]])
    
    return result

def global2local_legpos(legpos_global, x_global, y_global, z_global, roll, pitch, yaw):
        
    # transformation matrix global to body double 
    T_global_body = euler2transf(x_global, y_global, z_global, roll, pitch, yaw)

    # transformation matrix global to front left leg origin
    T_global_front_left_origin = np.matmul(T_global_body, euler2transf(body_width/2, -body_front, 0, 0, 0, 0))

    # transformation matrix global to front right leg origin
    T_global_front_right_origin = np.matmul(T_global_body, euler2transf(-body_width/2, -body_front, 0, 0, 0, 0))

    # transformation matrix global to back left leg origin
    T_global_back_left_origin = np.matmul(T_global_body, euler2transf(body_width/2, body_back, 0, 0, 0, 0))

    # transformation matrix global to back right leg origin
    T_global_back_right_origin = np.matmul(T_global_body, euler2transf(-body_width/2, body_back, 0, 0, 0, 0))

    # convert leg pos to 4 legs x (4x1) vectors
    legpos_global_matix = np.concatenate((legpos_global, np.ones([1,4])), axis=0)

    # find local leg position
    legpos_local = np.zeros([4,4])
    legpos_local[:,0] =  np.matmul(invtransf(T_global_front_left_origin), legpos_global_matix[:,0])
    legpos_local[:,1] =  np.matmul(invtransf(T_global_front_right_origin), legpos_global_matix[:,1])
    legpos_local[:,2] =  np.matmul(invtransf(T_global_back_left_origin), legpos_global_matix[:,2])
    legpos_local[:,3] =  np.matmul(invtransf(T_global_back_right_origin), legpos_global_matix[:,3])

    return legpos_local[0:3,:]

def inv_kine(legpos_local): # 3x4 array
    
    result = np.zeros([3,4])
    
    for i in range(4):
        x = legpos_local[0,i]
        y = legpos_local[1,i]
        z = legpos_local[2,i]

        F = math.sqrt(x*x + z*z - hip_length*hip_length)

        if (i==1 or i==2):
            hip_rolling_angle = math.atan2(z, -x*pow((-1),i)) + math.atan2(F, -hip_length)
        else:
            hip_rolling_angle = -math.atan2(z, -x*pow((-1),i)) - math.atan2(F, -hip_length)
        

        D = (F*F + y*y - upperleg_length*upperleg_length - lowerleg_length*lowerleg_length) / (2*upperleg_length*lowerleg_length)
        
        knee_angle = -math.atan2((math.sqrt(1-D*D)), D)

        hip_pitching_angle = pow((-1),i)*(math.atan2(y,F) - math.atan2(lowerleg_length * math.sin(knee_angle), upperleg_length + lowerleg_length * math.cos(knee_angle)))

        result[0,i] = hip_rolling_angle
        result[1,i] = hip_pitching_angle
        result[2,i] = knee_angle
    
    return result

# testudog dimension
hip_length = 0.0623
upperleg_length = 0.118
lowerleg_length = 0.118
body_front = 0.102
body_back = 0.252
body_width = 0.150


