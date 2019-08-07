# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def initializer(N, dim, up, down):
       
    x = np.random.rand(N, dim)
    x = np.multiply(x,up-down)+down
    
    return x

def fobj(x):
    dim=x.shape[0]
    o=100*(np.power(x[1:dim]-np.power(x[0:dim-1],2),2))+np.power(x[0:dim-1]-1,2)
    return sum(o)

def Levy(d):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2))/(math.gamma((1+beta)/2)*beta*2)**((beta-1)/2)**(1/beta);
    u=np.random.randn(1,d)*sigma
    v=np.random.randn(1,d)
    step=np.divide(u,np.power(abs(v),(1/beta)))
    o=step
    return o  



def HHO(N,T,lb,ub,dim):
    print("Optimization is underway")
    Rabbit_Location =np.zeros(dim)
    Rabbit_Energy = math.inf
    
    XO = initializer(N,dim,ub,lb)
    CNVG = np.zeros(T,dtype=int)
    q=0
    t=0
    #xnew_rand= np.empty()
    point=[]
  
    
 
    while(t<T):
        fu=np.zeros((XO.shape[0],XO.shape[1]),dtype=int)
        fl=np.zeros((XO.shape[0],XO.shape[1]),dtype=int)
        fu.reshape(XO.shape[0],XO.shape[1])
        fl.reshape(XO.shape[0],XO.shape[1])
        flu = np.zeros((XO.shape[0],XO.shape[1]),dtype=int)
        fl.reshape(XO.shape[0],XO.shape[1])
        for i in range(XO.shape[0]):
            
           """Check Boundaries"""
           """Loope Otimization Required"""
           for j in range(XO.shape[1]):
                if(XO[i][j]<lb):
                    fl[i][j]=1
                elif(XO[i][j]>ub):
                    fu[i][j]=1
                flu[i][j] = fl[i][j]+fu[i][j]
                if(flu[i][j]==0):
                    flu[i][j]=1
                else:
                    flu[i][j]=0
             
           xnew = np.multiply(XO,flu)+np.multiply(ub,fu)+np.multiply(lb,fl)
           """Using xnew from here on"""
           """ CHeck Fitness"""
           fitness = fobj(xnew[i,:])
           #print(fitness)
           if fitness<Rabbit_Energy:
            Rabbit_Energy=fitness
            Rabbit_Location=xnew[i,:]
           
           E1=2*(1-(t/T)) 
            
        for i in range(xnew.shape[0]):
            E0=2*(random.random())-1 #-1<E0<1
            Escaping_Energy=E1*(E0)
    
            if abs(Escaping_Energy)>=1:
                q=random.random();
                rand_Hawk_index = math.floor(N*random.random()+1);
                x_rand = xnew[rand_Hawk_index%xnew.shape[0], :];
                #print(xnew_rand)
                if q<0.5:
                    xnew[i,:]=x_rand-random.random()*abs(x_rand-2*random.random()*xnew[i,:]);
                elif q>=0.5:    
                    xnew[i,:]=(Rabbit_Location[:]-np.average(xnew,axis=0))-random.random()*((ub-lb)*random.random()+lb)
            
            elif abs(Escaping_Energy)<1:
                r=random.random()
                
                if r>=0.5 and abs(Escaping_Energy)<0.5:
                    xnew[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-xnew[i,:]);
                
                if r>=0.5 and abs(Escaping_Energy)>=0.5:
                    Jump_strength=2*(1-random.random())
                    xnew[i,:]=(Rabbit_Location-xnew[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-xnew[i,:])
            
                if r<0.5 and abs(Escaping_Energy)>=0.5:
                    Jump_strength=2*(1-random.random());
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-xnew[i,:]);
                
                    if fobj(X1)<fobj(xnew[i,:]):
                        xnew[i,:]=X1
                    else:
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-xnew[i,:])+np.multiply(np.random.rand(dim),Levy(dim))
                        if fobj(X2)<fobj(xnew[i,:]):
                            xnew[i,:]=X2;
                            
                if r<0.5 and abs(Escaping_Energy)<0.5:
                    Jump_strength=2*(1-random.random())
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-np.average(xnew,axis=0));
                    
                    if fobj(X1)<fobj(xnew[i,:]):
                        xnew[i,:]=X1
                    else:
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-np.average(xnew,axis=0))+np.multiply(np.random.rand(dim),Levy(dim));
                        if fobj(X2)<fobj(xnew[i,:]):
                            xnew[i,:]=X2
        t=t+1
        CNVG[t-1]=Rabbit_Energy
        print(CNVG[t-1])
    
HHO(30,100,-100,100,3)    
   
                
