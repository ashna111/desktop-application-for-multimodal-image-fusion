from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog

# Root window
root=Tk()
root.title('Multi-modal medical image fusion to detect brain tumors')
# root.geometry('900x500')
scroll_bar = Scrollbar(root) 

# Upload Files frame
frame_file=LabelFrame(root, text="Select files:",pady=20)
frame_file.grid(row=0,column=0)

# Display Uploaded Images
canvas_mri = Canvas(root,width=512, height=512)
canvas_mri.grid(row=2,column=0)

canvas_ct = Canvas(root,width=512, height=512)
canvas_ct.grid(row=2,column=1)

# Points for registration
mri_points=[]
ct_points=[]

def openMRI():
    global my_image_1

    root.filename=filedialog.askopenfilename(initialdir="/", title="Select MRI Image")
    my_label_1=Label(root,text="MRI Image").grid(row=1,column=0)
    my_image_1=ImageTk.PhotoImage(Image.open(root.filename))
    mri_image=my_image_1

    canvas_mri.create_image(0, 0, image=my_image_1, anchor="nw")
    canvas_mri.config(scrollregion=canvas_mri.bbox(ALL))

    def printcoordsMRI(event):
        mri_x_label=Label(root, text="MRI X:"+str(event.x)).grid(row=3,column=0)
        mri_y_label=Label(root, text="MRI Y:"+str(event.y)).grid(row=4,column=0)
        # print (event.x,event.y)
        mri_points.append([event.x,event.y])
    #mouseclick event
    canvas_mri.bind("<Button 1>",printcoordsMRI)

def openCT():
    global my_image_2

    root.filename=filedialog.askopenfilename(initialdir="/", title="Select CT Image")
    my_label_2=Label(root,text="CT Image").grid(row=1,column=1)
    my_image_2=ImageTk.PhotoImage(Image.open(root.filename))
    ct_image=my_image_2

    canvas_ct.create_image(0, 0, image=my_image_2, anchor="nw")
    canvas_ct.config(scrollregion=canvas_ct.bbox(ALL))

    def printcoordsCT(event):
        ct_x_label=Label(root, text="CT X:"+str(event.x)).grid(row=3,column=1)
        ct_y_label=Label(root, text="CT Y:"+str(event.y)).grid(row=4,column=1)
        # print (event.x,event.y)
        ct_points.append([event.x,event.y])
    #mouseclick event
    canvas_ct.bind("<Button 1>",printcoordsCT)

mri_button=Button(frame_file,text="Select MRI File",command=openMRI)
mri_button.grid(row=0,column=0,pady=10,padx=10)

ct_button=Button(frame_file,text="Select CT File",command=openCT)
ct_button.grid(row=0,column=1,pady=10,padx=10)

# submit registration points
registration_button=Button(root,text="Submit Points")
registration_button.grid(row=5,column=0)






root.mainloop()