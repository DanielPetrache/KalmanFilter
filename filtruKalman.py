import numpy as np

class Kalman:
	def __init__(self,pozitie_initiala,viteza_initiala,varianta_accel,eroare_pozitie,eroare_viteza):
		self.I = np.eye(2)
		self.H = np.eye(2)
		self.stare = np.array([[pozitie_initiala],[viteza_initiala]])
		self.Q = np.array([[varianta_accel,0],[0,varianta_accel]])
		self.R = np.array([[eroare_pozitie**2, 0],[0,eroare_viteza**2]])
		self.P = np.array([[0.55,0],[0,0.55]])
		self.acc0 = 0
		self.v0 = 0
		self.nr = 0
		
	def setare_interval(self,t):
		self.dt = t
		self.A = np.array([[1,self.dt],[0,1]])
		self.B = np.array([[0.5*self.dt**2],[self.dt]])
		
	def setare_acceleratie(self,acc):
		self.u1 = acc
		
	def set_dif_marker(self,x):
		self.dif = abs(x)
		self.nr = 0
		
	def prezicere_stare(self,acc):
		if abs(acc)-self.dif<=0.11:
			self.nr+=1
		else:
			self.nr=0
		if self.nr < 175:
			self.stare[1,0] += self.acc0*self.dt+(self.u1-self.acc0)*self.dt/2
			self.stare[0,0] += self.v0*self.dt+(self.stare[1,0]-self.v0)*self.dt/2
			self.v0 = self.stare[1,0]
		else:
			self.stare[1,0]-=self.stare[1,0]/15
		self.acc0 = acc
		self.P = np.matmul(np.matmul(self.A,self.P),np.transpose(self.A))+self.Q
		
	def update(self,pozitie_gps,viteza_gps):
		self.z = np.array([[pozitie_gps],[viteza_gps]])
		self.K = np.matmul(self.P,np.linalg.inv(self.P+self.R))
		self.stare = self.stare + np.matmul(self.K,(self.z-self.stare))
		self.P = np.matmul(np.matmul((self.I-self.K),self.P),np.transpose((self.I-self.K)))
