from ctypes import pointer
from click import pass_context
import pybullet as p
import pybullet_envs
import pybullet_data
import torch 
import gym
from gym import spaces
import time
from stable_baselines3 import PPO 
from stable_baselines3 import SAC 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import numpy as np        
import math
import os
import inv_kine.inv_kine as ik
import matplotlib.pyplot as plt

# for plot
posx_list = []
posy_list = []
posz_list = []
velx_list = []
rot_list = []
pow_list = []

# see tensorboard : tensorboard --logdir=log (open terminal in final_project dir)
    
class TestudogEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TestudogEnv, self).__init__()
        self.state = self.init_state()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(45,), dtype=np.float32)
        
    def init_state(self):
        self.count = 0
        # p.connect(p.DIRECT)
        p.connect(p.GUI)
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
        self.testudogid = p.loadURDF("/ENPM690/final_project/urdf/testudog.urdf",[0,0,0.25],[0,0,0,1])
        focus,_ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=-90,cameraPitch=0,cameraTargetPosition=focus)
        
        # observation --> body: pos, rot, lin_vel ang_vel / joints: pos, vel / foot position? / foot contact?
        body_pos = p.getLinkState(self.testudogid,0)[0]
        body_rot = p.getLinkState(self.testudogid,0)[1]
        body_rot_rpy = p.getEulerFromQuaternion(body_rot) 
        body_lin_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        body_ang_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        joint_pos = []
        joint_vel = []
        joint_torque = []
        for i in range(12):
            joint_pos.append(p.getJointState(self.testudogid,i)[0])
            joint_vel.append(p.getJointState(self.testudogid,i)[1])        
            joint_torque.append(p.getJointState(self.testudogid,i)[3]) 
        # obs = list(body_pos) + list(body_rot_rpy)[0:2] + list(body_lin_vel) + list(body_ang_vel) + joint_pos + joint_vel + joint_torque
        obs = list(body_pos) + list(body_lin_vel) + list(body_rot_rpy) + joint_pos + joint_vel + joint_torque
        obs = np.array(obs).astype(np.float32)
        return obs
    
    def reset(self):
        p.disconnect()
        obs = self.init_state()
        self.state = obs
        return obs
        
    def step(self,action):
        # action_legpos = np.array([[(action[0]*0.02)+0.1373, (action[3]*0.02)-0.1373, (action[6]*0.02)+0.1373, (action[9]*0.02)-0.1373],
        #                         [(action[1]*0.15)-0.102, (action[4]*0.15)-0.102, (action[7]*0.15)+0.252, (action[10]*0.15)+0.252],
        #                         [(action[2]*0.05)+0.05, (action[5]*0.05)+0.05, (action[8]*0.05)+0.05, (action[11]*0.05)+0.05]])
        action_legpos = np.array([[(action[0]*0)+0.1373, (action[0]*0)-0.1373, (action[3]*0)+0.1373, (action[3]*0)-0.1373],
                                [(action[1]*0.15)-0.102, (action[1]*0.15)-0.102, (action[4]*0.15)+0.252, (action[4]*0.15)+0.252],
                                [(action[2]*0.05)+0.05, (action[2]*0.05)+0.05, (action[5]*0.05)+0.05, (action[5]*0.05)+0.05]])
        # testudog legpos range
        # [[ 0.1373 -0.1373  0.1373 -0.1373] +-0.02
        # [-0.102  -0.102   0.252   0.252 ] +-0.1
        # [ 0.1     0.      0.      0.    ]] 0-0.1
        joint_angle = ik.inv_kine(ik.global2local_legpos(action_legpos,x_global,y_global,z_global,roll,pitch,yaw))
        joint_angle = np.reshape(np.transpose(joint_angle),[1,12])[0]
        # p.setJointMotorControlArray(self.testudogid,list(range(12)),p.POSITION_CONTROL,targetPositions=joint_angle)
        vel1 = action[6:9]
        vel2 = action[9:12]
        p.setJointMotorControlArray(self.testudogid,list(range(12)),p.POSITION_CONTROL,\
            targetPositions=joint_angle,targetVelocities=np.block([vel1,vel1,vel2,vel2]),positionGains=4*[0.02,0.02,0.02],velocityGains=4*[0.1,0.1,0.1])
        focus,_ = p.getBasePositionAndOrientation(self.testudogid)
        p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=-90,cameraPitch=0,cameraTargetPosition=focus)
        p.stepSimulation()
        
        # observation --> body: pos, rot, lin_vel ang_vel / joints: pos, vel / foot position? / foot contact?
        body_pos = p.getLinkState(self.testudogid,0)[0]
        body_rot = p.getLinkState(self.testudogid,0)[1]
        body_rot_rpy = p.getEulerFromQuaternion(body_rot) 
        body_lin_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[6]
        body_ang_vel = p.getLinkState(self.testudogid,0,computeLinkVelocity=1)[7]
        joint_pos = []
        joint_vel = []
        joint_torque = []
        joint_pow = []
        for i in range(12):
            joint_pos.append(p.getJointState(self.testudogid,i)[0])
            joint_vel.append(p.getJointState(self.testudogid,i)[1]) 
            joint_torque.append(p.getJointState(self.testudogid,i)[3]) 
            joint_pow.append(p.getJointState(self.testudogid,i)[1]*p.getJointState(self.testudogid,i)[3]) # pow = torque*vel
                  
        # obs = list(body_pos) + list(body_rot_rpy)[0:2] + list(body_lin_vel) + list(body_ang_vel) + joint_pos + joint_vel + joint_torque
        obs = list(body_pos) + list(body_lin_vel) + list(body_rot_rpy) + joint_pos + joint_vel + joint_torque
        obs = np.array(obs).astype(np.float32)
        info = {}
        self.count += 1
        w1 = 2
        w2 = 0.1
        w3 = 0.5
        w4 = 2
        w5 = 0.5
        w6 = 0.2
        w7 = 0.5
        dt = 1/240
        
        # terminal fail condition eg robot fall  
        if (body_rot_rpy[1]<0):
            reward = -20
            # reward = -40
            done = True
            return obs, reward, done, info
        
        # survival reward
        if (self.count>5000):
            reward = 10
            # reward = 10
            done = True
            return obs, reward, done, info
        
        done = False
        # reward --> forward velocity + consumed energy
        # reward = -w1*body_pos[1] -w2*sum(np.abs(joint_pow))*dt -w3*abs(body_pos[0]) -w4*abs(body_pos[2]-0.15) + 0.01
        reward = -w1*body_pos[1] -w2*sum(np.abs(joint_pow))*dt -w3*abs(body_pos[0]) -w4*abs((math.pi/2)-body_rot_rpy[1])\
                    -w5*body_lin_vel[1] -w6*abs(body_lin_vel[0]) -w7*abs(body_pos[2]-0.16) + 0.5
        # reward = w1*(obs[1]-self.state[1]) - w2*sum(joint_pow)*dt + w3*(body_pos[2]-0.2) + 0.01
        # print(- w5*body_lin_vel[1])
        
        global posx_list, posy_list, posz_list, velx_list, rot_list, pow_list
        posx_list.append(-body_pos[1])
        posy_list.append(body_pos[0])
        posz_list.append(body_pos[2])
        velx_list.append(-body_lin_vel[1])
        rot_list.append((math.pi/2)-body_rot_rpy[1])
        pow_list.append(sum(np.abs(joint_pow)))
        
        self.state = obs
        return obs, reward, done, info 

if (__name__ == '__main__'):
    # set save directory
    model_dir ="ENPM690/final_project/models/PPO"
    log_dir = "ENPM690/final_project/log"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # testudog initial state
    x_global = 0
    y_global = 0
    z_global = 0.15
    roll = 0
    pitch = 0
    yaw = 0
        
    # create model
    env = TestudogEnv()
    # check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    TIMESTEPS = 50000
    count = 1
    
    # load model
    model_path = f"{model_dir}/15450000.zip"
    model = PPO.load(model_path,env=env)
    count = int(15450000/TIMESTEPS)
    
    # # train loop   
    # while(True):
    #     print(count)
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    #     model.save(f"{model_dir}/{TIMESTEPS*count}")
    #     count += 1
    #     if True == False:
    #         break
    
    # run trained model
    episodes = 1
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done and env.count<2000:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(1/240)
    
    size = len(posx_list)
    time_sim = np.arange(0,size,1)/240
    fig, axes = plt.subplots(3, 2)   
    axes[0,0].plot(time_sim, posx_list) 
    axes[1,0].plot(time_sim, posy_list) 
    axes[2,0].plot(time_sim, posz_list) 
    axes[0,1].plot(time_sim, velx_list) 
    axes[1,1].plot(time_sim, rot_list) 
    axes[2,1].plot(time_sim, pow_list) 
    
    plt.plot()
            
