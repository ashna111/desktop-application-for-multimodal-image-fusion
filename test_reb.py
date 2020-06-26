from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import os
from datetime import datetime
from flask import Flask, render_template, request, url_for
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import cv2
import imageio
import scipy.ndimage as ndi
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19
import pywt
import pywt.data
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater

# Points for registration
mri_points=[]
ct_points=[]

def procrustes(X, Y, scaling=True, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #rot =1
    #scale=2
    #translate=3
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def register():
    frame_file.destroy()
    canvas_mri.destroy()
    canvas_ct.destroy()
    registration_button.destroy()
    my_label_1.destroy()
    my_label_2.destroy()
    # mri_x_label.destroy()
    # mri_y_label.destroy()
    # ct_x_label.destroy()
    # ct_y_label.destroy()

    mri_registered_label=Label(root,text="MRI Rregistered Image").grid(row=0,column=0)
    ct_registered_label=Label(root,text="CT Image").grid(row=0,column=1)

    ct=cv2.imread(r'C:\Users\rebec\Desktop\ct.jpg')
    ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)

    ct_registered_image = Image.fromarray(ct)
    ct_registered_image = ImageTk.PhotoImage(image=ct_registered_image) 

    ct_registered_image_label=Label(image=ct_registered_image)
    ct_registered_image_label.grid(row=1,column=1)

    ct_registered_image_label.image=ct_registered_image

    # canvas_ct = Canvas(root,width=512, height=512)
    # canvas_ct.grid(row=2,column=1)

    # canvas_ct.create_image(0, 0, image=ct_registered_image, anchor="nw")
    # canvas_ct.config(scrollregion=canvas_ct.bbox(ALL))

    

    # print(ct_registered_image_label.winfo_exists())

def openMRI():
    global my_image_1,mri_image,my_label_1

    root.filename=filedialog.askopenfilename(initialdir="/", title="Select MRI Image")
    my_label_1=Label(root,text="MRI Image")
    my_label_1.grid(row=1,column=0)
    my_image_1=ImageTk.PhotoImage(Image.open(root.filename))
    mri_image=my_image_1

    canvas_mri.create_image(0, 0, image=my_image_1, anchor="nw")
    canvas_mri.config(scrollregion=canvas_mri.bbox(ALL))

    def printcoordsMRI(event):
        global mri_x_label,mri_y_label

        mri_x_label=Label(root, text="MRI X:"+str(event.x))
        mri_x_label.grid(row=3,column=0)
        mri_y_label=Label(root, text="MRI Y:"+str(event.y))
        mri_y_label.grid(row=4,column=0)
        # print (event.x,event.y)
        mri_points.append([event.x,event.y])

        mri_x_label.after(3000, mri_x_label.destroy)
        mri_y_label.after(3000, mri_y_label.destroy)

    #mouseclick event
    canvas_mri.bind("<Button 1>",printcoordsMRI)

def openCT():
    global my_image_2,ct_image,registration_button,my_label_2,ct

    root.filename=filedialog.askopenfilename(initialdir="/", title="Select CT Image")
    my_label_2=Label(root,text="CT Image")
    my_label_2.grid(row=1,column=1)
    ct=cv2.imread(r'C:\Users\rebec\Desktop\ct.jpg')
    ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
    my_image_2=ImageTk.PhotoImage(Image.open(root.filename))
    ct_image=my_image_2

    canvas_ct.create_image(0, 0, image=my_image_2, anchor="nw")
    canvas_ct.config(scrollregion=canvas_ct.bbox(ALL))

    # submit registration points button
    registration_button=Button(root,text="Submit Points",command=register)
    registration_button.grid(row=5,column=0)

    def printcoordsCT(event):
        global ct_x_label,ct_y_label

        ct_x_label=Label(root, text="CT X:"+str(event.x))
        ct_x_label.grid(row=3,column=1)
        ct_y_label=Label(root, text="CT Y:"+str(event.y))
        ct_y_label.grid(row=4,column=1)
        # print (event.x,event.y)
        ct_points.append([event.x,event.y])

        ct_x_label.after(3000, ct_x_label.destroy)
        ct_y_label.after(3000, ct_y_label.destroy)
    #mouseclick event
    canvas_ct.bind("<Button 1>",printcoordsCT)


# Root window
root=Tk()
root.title('Multi-modal medical image fusion to detect brain tumors')
# root.geometry('900x500')
scroll_bar = Scrollbar(root) 

#  Upload Files frame
frame_file=LabelFrame(root, text="Select files:",pady=20)
frame_file.grid(row=0,column=0)

# Display Uploaded Images
canvas_mri = Canvas(root,width=512, height=512)
canvas_mri.grid(row=2,column=0)

canvas_ct = Canvas(root,width=512, height=512)
canvas_ct.grid(row=2,column=1)


mri_button=Button(frame_file,text="Select MRI File",command=openMRI)
mri_button.grid(row=0,column=0,pady=10,padx=10)

ct_button=Button(frame_file,text="Select CT File",command=openCT)
ct_button.grid(row=0,column=1,pady=10,padx=10)





root.mainloop()