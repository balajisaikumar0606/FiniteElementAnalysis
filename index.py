# -*- coding: utf-8 -*-
"""
Lang - Python3.7
@author: balaji
"""
import pip
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
    a1=((1/(2*A)))*(y2-y3)
    a2=((1/(2*A)))*(y1-y3)
    a3=((1/(2*A)))*(y2-y1)
    b1=((1/(2*A)))*(x2-x3)
    b2=((1/(2*A)))*(x1-x3)
    b3=((1/(2*A)))*(x1-x2)
    B=[[a1, 0, b1], [0, b1, a1], [a2, 0, b2],[ 0, b2, a2],[ a3, 0, b3], [0, b3, a3]]
    B=np.array(B)
    return B
def k_matrix(h,A,C,B):
    C=np.array(C)
    B=np.array(B)
    k=np.dot(h*A,B.dot(C).dot(B.T))
    return k
#bearing capacity calculations
def bearing_capacity(c,N_c,q,N_q,gamma,N_gamma,B):
    q_ult=c*N_c + q*N_q+0.5*gamma*N_gamma*B
    return q_ult
Q=bearing_capacity(5,95.66,24,81.27,16,115.31,1.5)# phi=40

#inputs
fck=float(input('enter the grade of concrete , For my problem select 40'));
E=5000*math.sqrt(fck)*10**3;
h=float(input("Length of the slab, taken as 10m")); 
mu=0.1; #for high strength concrete,mu=0.1
C=np.dot((E)/((1+mu)*(1-2*mu)),[[1-mu,mu,0],[mu,1-mu,0],[0,0,0.5*(1-2*mu)]])
#Q=350 #Q=input("enter Q in 'kN'")
udl=1000 #udl=input("enter q in 'kN/m'")
udl2=24*1.5;
udl3=24*3;


#element information
el=xlrd.open_workbook("input_files/elemt_inf.xlsx")
sheet1 = el.sheet_by_index(0)
el1=np.zeros((10,6))
for i in range(0,10):
    for j in range(0,6):
        el1[i][j]=(sheet1.cell_value(i,j))
#formation of B matrix
writer = pd.ExcelWriter("output_files/k.xlsx", engine='xlsxwriter')
writer1=pd.ExcelWriter("output_files/B.xlsx", engine='xlsxwriter')
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
wb =xlrd.open_workbook("output_files/k.xlsx")
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

def append(sheet):
    k=[]
    for i in range(1,7):
        for j in range(1,7):
            k.append(sheet.cell_value(i,j))
    return k
k1=append(sheet1)
k2=append(sheet2)
k3=append(sheet3)
k4=append(sheet4)
k5=append(sheet5)
k6=append(sheet6)
k7=append(sheet7)
k8=append(sheet8)
k9=append(sheet9)
k10=append(sheet10)
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
qB=np.zeros((16,1))
F=np.zeros((16,1))
F[7]=Q
q[7]=udl*(1.5+1.5)/2
q[-1]=udl*(1.5/2)
q[-3]=udl*(1.5/2)
qB[3]=udl2*(0.75/2)
qB[9]=udl2*(0.75)
qB[11]=udl2*(0.75)
qB[5]=udl2*(0.75/2)
qB[1]=udl2*(1.5)
k_fin=np.zeros((16,16))
for i in range(16):
    for j in range(16):
        k_fin[i][j]=K[i][j]
k_inverse = np.linalg.inv(k_fin)
k_inverse=np.array(k_inverse)
F=np.array(F)
q=np.array(q)
final=[0,udl3*(1.5/2),0,udl3*(1.5/2)]
u = k_inverse.dot(F+q+qB)
u_final=np.append(u,np.zeros((4,1)))
F_final=np.append(F,np.zeros((4,1)))
q_final=np.append(q,np.zeros((4,1)))
qB_final=np.append(qB,final)
del final
'''
from this stage, strain values of each element is obtained to calculate the stresses
'''
def d(l,u_final):
    d=[]
    for i in l:
        d.append(u_final[i-1])
    return d
d1=d(l1,u_final)
d2=d(l2,u_final)
d3=d(l3,u_final)
d4=d(l4,u_final)
d5=d(l5,u_final)
d6=d(l6,u_final)
d7=d(l7,u_final)
d8=d(l8,u_final)
d9=d(l9,u_final)
d10=d(l10,u_final)
wb =xlrd.open_workbook("output_files/B.xlsx")
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
def assembly_b(sheet):
    b=np.zeros((6,3))
    for i in range(1,7):
        for j in range(1,4):
            b[i-1][j-1]=sheet.cell_value(i,j)
    return b
b1=assembly_b(sheet1)
b2=assembly_b(sheet2)
b3=assembly_b(sheet3)
b4=assembly_b(sheet4)
b5=assembly_b(sheet5)
b6=assembly_b(sheet6)
b7=assembly_b(sheet7)
b8=assembly_b(sheet8)
b9=assembly_b(sheet9)
b10=assembly_b(sheet10)
eps={}
eps1=np.dot(np.transpose(b1),d1)
eps["eps1"]=eps1
eps2=np.dot(np.transpose(b2),d2)
eps["eps2"]=eps2
eps3=np.dot(np.transpose(b3),d3)
eps["eps3"]=eps3
eps4=np.dot(np.transpose(b4),d4)
eps["eps4"]=eps4
eps5=np.dot(np.transpose(b5),d5)
eps["eps5"]=eps5
eps6=np.dot(np.transpose(b6),d6)
eps["eps6"]=eps6
eps7=np.dot(np.transpose(b7),d7)
eps["eps7"]=eps7
eps8=np.dot(np.transpose(b8),d8)
eps["eps8"]=eps8
eps9=np.dot(np.transpose(b9),d9)
eps["eps9"]=eps9
eps10=np.dot(np.transpose(b10),d10)
eps["eps10"]=eps10
sig={}
sig1=np.dot(C,eps1)
sig["sig1"]=sig1
sig2=np.dot(C,eps2)
sig["sig2"]=sig2
sig3=np.dot(C,eps3)
sig["sig3"]=sig3
sig4=np.dot(C,eps4)
sig["sig4"]=sig4
sig5=np.dot(C,eps5)
sig["sig5"]=sig5
sig6=np.dot(C,eps6)
sig["sig6"]=sig6
sig7=np.dot(C,eps7)
sig["sig7"]=sig7
sig8=np.dot(C,eps8)
sig["sig8"]=sig8
sig9=np.dot(C,eps9)
sig["sig9"]=sig9
sig10=np.dot(C,eps10)
sig["sig10"]=sig10
"""
solving [k]{d}={F}+{q} for unknowns in F
"""
Kd=(K.dot(u_final))
F_final=Kd-qB_final
for i in range(16):
    F_final[i]=0
writer = pd.ExcelWriter('output_files/k-assembly_u_F_eps_sig.xlsx', engine='xlsxwriter')
df1 = pd.DataFrame(K)
df1.to_excel(writer,sheet_name='k_assembly')
df2 = pd.DataFrame(u_final)
df2.to_excel(writer,sheet_name='u_final')
df3 = pd.DataFrame(eps)
df3.to_excel(writer,sheet_name='eps')
df4 = pd.DataFrame(sig)
df4.to_excel(writer,sheet_name='sig')
df5 = pd.DataFrame(F_final.round(decimals=2))
df5.to_excel(writer,sheet_name='F_final')
writer.save()

