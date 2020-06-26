# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:28:19 2020

@author: balaji
"""
import xlrd 
import numpy as np
import pandas as pd
import math

def area_triangle(x1,y1,x2,y2,x3,y3):
    A=[[x1,y1,1],[x2,y2,1],[x3,y3,1]]
    A=np.array(A)
    return 0.5*(np.linalg.det(A))
def B_matrix(x1,y1,x2,y2,x3,y3):
    x1=float(x1)
    y1=float(y1)
    x2=float(x2)
    y2=float(y2)
    x3=float(x3)
    y3=float(y3)
    A=area_triangle(x1,y1,x2,y2,x3,y3)
    a1=(0.5*A)*(y2-y3)
    a2=(0.5*A)*(y1-y3)
    a3=(0.5*A)*(y2-y1)
    b1=(0.5*A)*(x2-x3)
    b2=(0.5*A)*(x1-x3)
    b3=(0.5*A)*(x1-x2)
    c1=(0.5*A)*(x2*y3-x3*y2)
    c2=(0.5*A)*(x1*y3-x3*y1)
    c3=(0.5*A)*(x1*y2-x2*y1)
    B=[[a1, 0, b1], [0, b1, a1], [a2, 0, b2],[ 0, b2, a2],[ a3, 0, b3], [0, b3, a3]]
    B=np.array(B)
    return B
def k_matrix(h,A,C,B):
    temp1=np.dot(h*A,B)
    temp2=np.dot(C,np.transpose(B))
    k=np.dot(temp1,temp2)
    return k
#bearing capacity calculations
def bearing_capacity(c,N_c,q,N_q,gamma,N_gamma,B):
    q_ult=c*N_c + q*N_q+0.5*gamma*N_gamma*B
    return q_ult
Q=bearing_capacity(5,95.66,35,81.27,17.5,115.31,3)# phi=40
#inputs
fck=int(input('enter the grade of concrete'));
E=5000*math.sqrt(fck);
h=int(input("thickness of the slab")); 
mu=0.1; #considering high strength concrete
C=np.dot(E/((1+mu)*(1-2*mu)),[[1-mu,mu,0],[mu,1-mu,0],[0,0,1-2*mu]])
#Q=350 #Q=input("enter Q in 'kN'")
udl=300 #udl=input("enter q in 'kN/m'")

#element information
el=xlrd.open_workbook("elemt_inf.xlsx")
sheet1 = el.sheet_by_index(0)
el1=np.zeros((10,6))
for i in range(0,10):
    for j in range(0,6):
        el1[i][j]=(sheet1.cell_value(i,j))
#formation of B matrix
writer = pd.ExcelWriter('k.xlsx', engine='xlsxwriter')
writer1=pd.ExcelWriter('B.xlsx', engine='xlsxwriter')
for i in range(10):
    A= area_triangle(el1[i][0],el1[i][1],el1[i][2],el1[i][3],el1[i][4],el1[i][5])
    B=B_matrix(el1[i][0],el1[i][1],el1[i][2],el1[i][3],el1[i][4],el1[i][5])
    k=k_matrix(h,A,C,B)
    df1 = pd.DataFrame(k)
    df1.to_excel(writer,sheet_name="Sheet"+str(i))
    df2 = pd.DataFrame(B)
    df2.to_excel(writer1,sheet_name="Sheet"+str(i))
writer.save()
writer1.save()
wb =xlrd.open_workbook("k.xlsx")
sheet1 = wb.sheet_by_index(0)
sheet2= wb.sheet_by_index(1)
sheet3 = wb.sheet_by_index(2)
sheet4 = wb.sheet_by_index(3)
sheet5 = wb.sheet_by_index(4)
sheet6 = wb.sheet_by_index(5)
sheet7 = wb.sheet_by_index(6)
sheet8 = wb.sheet_by_index(7)
sheet9 = wb.sheet_by_index(8)
sheet10 = wb.sheet_by_index(9)
k1=[]
k2=[]
k3=[]
k4=[]
k5=[]
k6=[]
k7=[]
k8=[]
k9=[]
k10=[]
for i in range(1,7):
    for j in range(1,7):
        k1.append(sheet1.cell_value(i,j))
        k2.append(sheet2.cell_value(i,j))
        k3.append(sheet3.cell_value(i,j))
        k4.append(sheet4.cell_value(i,j))
        k5.append(sheet5.cell_value(i,j))
        k6.append(sheet6.cell_value(i,j))
        k7.append(sheet7.cell_value(i,j))
        k8.append(sheet8.cell_value(i,j))
        k9.append(sheet9.cell_value(i,j))
        k10.append(sheet10.cell_value(i,j))
l1=[19,20,9,10,3,4]
l2=[19,20,1,2,9,10]
l3=[1,2,11,12,9,10]
l4=[1,2,17,18,11,12]
l5=[17,18,5,6,11,12]
l6=[5,6,15,16,11,12]
l7=[15,16,7,8,11,12]
l8=[7,8,9,10,11,12]
l9=[7,8,13,14,9,10]
l10=[13,14,3,4,9,10]
indx1=[]
indx2=[]
indx3=[]
indx4=[]
indx5=[]
indx6=[]
indx7=[]
indx8=[]
indx9=[]
indx10=[]
for i in l1:
    for j in l1:
        indx1.append((i-1,j-1))
for i in l2:
    for j in l2:
        indx2.append((i-1,j-1))
for i in l3:
    for j in l3:
        indx3.append((i-1,j-1))
for i in l4:
    for j in l4:
        indx4.append((i-1,j-1))
for i in l5:
    for j in l5:
        indx5.append((i-1,j-1))
for i in l6:
    for j in l6:
        indx6.append((i-1,j-1))
for i in l7:
    for j in l7:
        indx7.append((i-1,j-1))
for i in l8:
    for j in l8:
        indx8.append((i-1,j-1))
for i in l9:
    for j in l9:
        indx9.append((i-1,j-1))
for i in l10:
    for j in l10:
        indx10.append((i-1,j-1))        
dict1={}
dict2={}
dict3={}
dict4={}
dict5={}
dict6={}
dict7={}
dict8={}
dict9={}
dict10={}
for i in range(len(k1)):
    dict1[indx1[i]]=k1[i]
    dict2[indx2[i]]=k2[i]
    dict3[indx3[i]]=k3[i]
    dict4[indx4[i]]=k4[i]
    dict5[indx5[i]]=k5[i]
    dict6[indx6[i]]=k6[i]
    dict7[indx7[i]]=k7[i]
    dict8[indx8[i]]=k8[i]
    dict9[indx9[i]]=k9[i]
    dict10[indx10[i]]=k10[i]
K=np.zeros((20,20))
for key in dict1:
    K[key[0]][key[1]]=dict1[key]
for key in dict2:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict2[key]
    else:
        K[key[0]][key[1]]+=dict2[key]
for key in dict2:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict2[key]
    else:
        K[key[0]][key[1]]+=dict2[key]
for key in dict3:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict3[key]
    else:
        K[key[0]][key[1]]+=dict3[key]
for key in dict4:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict4[key]
    else:
        K[key[0]][key[1]]+=dict4[key]
for key in dict5:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict5[key]
    else:
        K[key[0]][key[1]]+=dict5[key]
for key in dict6:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict6[key]
    else:
        K[key[0]][key[1]]+=dict6[key]
for key in dict7:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict7[key]
    else:
        K[key[0]][key[1]]+=dict7[key]
for key in dict8:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict8[key]
    else:
        K[key[0]][key[1]]+=dict8[key]
for key in dict9:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict9[key]
    else:
        K[key[0]][key[1]]+=dict9[key]
for key in dict10:
    if K[key[0]][key[1]]==0:
        K[key[0]][key[1]]=dict10[key]
    else:
        K[key[0]][key[1]]+=dict10[key]


#solving the values
q=np.zeros((16,1))
F=np.zeros((16,1))
F[7]=Q
q[7]=udl*(1.5+1.5)/2
q[-1]=udl*(1.5/2)
q[-3]=udl*(1.5/2)
k_fin=np.zeros((16,16))
for i in range(16):
    for j in range(16):
        k_fin[i][j]=K[i][j]
k_inverse = np.linalg.inv(k_fin)
u = np.dot(k_inverse,F+q) 
zer=np.zeros((4,1))
u_final=np.append(u,zer)
F_final=np.append(F,zer)
q_final=np.append(q,zer)
writer = pd.ExcelWriter('k_assembly and u_final.xlsx', engine='xlsxwriter')
df1 = pd.DataFrame(K)
df1.to_excel(writer,sheet_name='k_assembly')
df2 = pd.DataFrame(u_final)
df2.to_excel(writer,sheet_name='u_final')
writer.save()
'''
from this stage, strain values of each element is obtained to calculate the stresses
'''
d1=[]
d2=[] 
d3=[]
d4=[] 
d5=[]
d6=[] 
d7=[]
d8=[] 
d9=[]
d10=[] 
for i in l1:
    d1.append(u_final[i-1])
for i in l2:
    d2.append(u_final[i-1])
for i in l3:
    d3.append(u_final[i-1])
for i in l4:
    d4.append(u_final[i-1])
for i in l5:
    d5.append(u_final[i-1])
for i in l6:
    d6.append(u_final[i-1])
for i in l7:
    d7.append(u_final[i-1])
for i in l8:
    d8.append(u_final[i-1])
for i in l9:
    d9.append(u_final[i-1])
for i in l10:
    d10.append(u_final[i-1])
wb =xlrd.open_workbook("B.xlsx")
sheet1 = wb.sheet_by_index(0)
sheet2= wb.sheet_by_index(1)
sheet3 = wb.sheet_by_index(2)
sheet4 = wb.sheet_by_index(3)
sheet5 = wb.sheet_by_index(4)
sheet6 = wb.sheet_by_index(5)
sheet7 = wb.sheet_by_index(6)
sheet8 = wb.sheet_by_index(7)
sheet9 = wb.sheet_by_index(8)
sheet10 = wb.sheet_by_index(9)
b1=np.zeros((6,3))
b2=np.zeros((6,3))
b3=np.zeros((6,3))
b4=np.zeros((6,3))
b5=np.zeros((6,3))
b6=np.zeros((6,3))
b7=np.zeros((6,3))
b8=np.zeros((6,3))
b9=np.zeros((6,3))
b10=np.zeros((6,3))
for i in range(1,7):
    for j in range(1,4):
        b1[i-1][j-1]=sheet1.cell_value(i,j)
        b2[i-1][j-1]=sheet2.cell_value(i,j)
        b3[i-1][j-1]=sheet3.cell_value(i,j)
        b4[i-1][j-1]=sheet4.cell_value(i,j)
        b5[i-1][j-1]=sheet5.cell_value(i,j)
        b6[i-1][j-1]=sheet6.cell_value(i,j)
        b7[i-1][j-1]=sheet7.cell_value(i,j)
        b8[i-1][j-1]=sheet8.cell_value(i,j)
        b9[i-1][j-1]=sheet9.cell_value(i,j)
        b10[i-1][j-1]=sheet10.cell_value(i,j)
eps1=np.dot(np.transpose(b1),d1)
eps2=np.dot(np.transpose(b2),d2)
eps3=np.dot(np.transpose(b3),d3)
eps4=np.dot(np.transpose(b4),d4)
eps5=np.dot(np.transpose(b5),d5)
eps6=np.dot(np.transpose(b6),d6)
eps7=np.dot(np.transpose(b7),d7)
eps8=np.dot(np.transpose(b8),d8)
eps9=np.dot(np.transpose(b9),d9)
eps10=np.dot(np.transpose(b10),d10)
sig1=np.dot(C,eps1)
sig2=np.dot(C,eps2)
sig3=np.dot(C,eps3)
sig4=np.dot(C,eps4)
sig5=np.dot(C,eps5)
sig6=np.dot(C,eps6)
sig7=np.dot(C,eps7)
sig8=np.dot(C,eps8)
sig9=np.dot(C,eps9)
sig10=np.dot(C,eps10)
"""
solving [k]{d}={F}+{q} for unknowns in F
"""
Kd=np.dot(K,u_final)
F_final=np.append(F_final,Kd[len(Kd)-4:len(Kd)])