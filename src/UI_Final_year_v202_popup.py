#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
"""
Created on Sat Aug 22 21:35:54 2020

@author: fahim
"""


import sys
from PIL import Image, ImageTk
import cv2
from PIL import Image, ImageTk
from tkinter import messagebox
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    #UI_Final_year_v202_popup_support.set_Tk_var()
    top = Toplevel1 (root)
    #UI_Final_year_v202_popup_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    #UI_Final_year_v202_popup_support.set_Tk_var()
    top = Toplevel1 (w)
    #UI_Final_year_v202_popup_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
       
    def c_01_active( self,event=None):
        print('c_01 value = '+c_01_val.get())
        center_coordinates = (223, 145)
        if c_01_val.get()=='1':
            self.highlight(255,223, 145)
        else :
            self.highlight(0,223, 145)
        sys.stdout.flush()    #########################################################    
        
    def c_02_active( self,event=None):
        print('c_02 value = '+c_02_val.get())
        center_coordinates = (149, 195)
        if c_02_val.get()=='1':
            self.highlight(255,125, 195)
        else :
            self.highlight(0,125, 195)
        sys.stdout.flush()    #########################################################    

    def c_03_active( self,event=None):
        print('c_03 value = '+c_03_val.get())
        center_coordinates = (170 ,195)
        if c_03_val.get()=='1':
            self.highlight(255,173 ,195)
        else :
            self.highlight(0,173 ,195)
        sys.stdout.flush()    #########################################################    

        
    def c_04_active( self,event=None):
        print('c_04 value = '+c_04_val.get())
        center_coordinates = (223 ,195)
        if c_04_val.get()=='1':
            self.highlight(255,223 ,195)
        else :
            self.highlight(0,223 ,195)
        sys.stdout.flush()    #########################################################    

    def c_05_active( self,event=None):
        print('c_05 value = '+c_05_val.get())
        center_coordinates = (298 ,195)
        if c_05_val.get()=='1':
            self.highlight(255,271 ,195)
        else :
            self.highlight(0,271 ,195)
        sys.stdout.flush()    #########################################################    

    def c_06_active( self,event=None):
        print('c_06 value = '+c_06_val.get())
        center_coordinates = (377 ,195)
        if c_06_val.get()=='1':
            self.highlight(255,319 ,195)
        else :
            self.highlight(0,319 ,195)
        sys.stdout.flush()    #########################################################    

############################################################################################
    def c_07_active( self,event=None):
        print('c_07 value = '+c_07_val.get())
        center_coordinates = (20 ,247)   
        if c_07_val.get()=='1':
            self.highlight(255,78 ,247)
        else :
            self.highlight(0,78 ,247)
        sys.stdout.flush()    #########################################################
    def c_08_active( self,event=None):
        print('c_08 value = '+c_08_val.get())
        center_coordinates = (162, 247)
        if c_08_val.get()=='1':
            self.highlight(255,126, 247)
        else :
            self.highlight(0,126, 247)
        sys.stdout.flush()    #########################################################    

    def c_09_active( self,event=None):
        print('c_09 value = '+c_09_val.get())
        center_coordinates = (143, 247)
        if c_09_val.get()=='1':
            self.highlight(255,174, 247)
        else :
            self.highlight(0,174, 247)
        sys.stdout.flush()    #########################################################    

    def c_10_active( self,event=None):
        print('c_10 value = '+c_10_val.get())
        center_coordinates = (224, 247)
        if c_10_val.get()=='1':
            self.highlight(255,224, 247)
        else :
            self.highlight(0,224, 247)
        sys.stdout.flush()    #########################################################    

    def c_11_active( self,event=None):
        print('c_11 value = '+c_11_val.get())
        center_coordinates = (305, 247)
        if c_11_val.get()=='1':
            self.highlight(255,272, 247)
        else :
            self.highlight(0,272, 247)
        sys.stdout.flush()    #########################################################    

    def c_12_active( self,event=None):
        print('c_12 value = '+c_12_val.get())
        center_coordinates = (385, 247)
        if c_12_val.get()=='1':
            self.highlight(255,320, 247)
        else :
            self.highlight(0,320, 247)
        sys.stdout.flush()    #########################################################    

    def c_13_active( self,event=None):
        print('c_13 value = '+c_13_val.get())
        center_coordinates = (425 ,247)   
        if c_13_val.get()=='1':
            self.highlight(255,368 ,247)
        else :
            self.highlight(0,368 ,247)
        sys.stdout.flush()    #########################################################        
        
##############################################################################################
    def c_14_active( self,event=None):
        print('c_14 value = '+c_14_val.get())
        center_coordinates = (126, 301)
        if c_14_val.get()=='1':
            self.highlight(255,126, 301)
        else :
            self.highlight(0,126, 301)
        sys.stdout.flush()    #########################################################    

    def c_15_active( self,event=None):
        print('c_15 value = '+c_15_val.get())
        center_coordinates = (173,273)
        if c_15_val.get()=='1':
            self.highlight(255,173,301)
        else :
            self.highlight(0,173,301)
        sys.stdout.flush()    #########################################################    

    def c_16_active( self,event=None):
        print('c_16 value = '+c_16_val.get())
        center_coordinates = (224, 301)
        if c_16_val.get()=='1':
            self.highlight(255,223, 301)
        else :
            self.highlight(0,223, 301)
        sys.stdout.flush()    #########################################################    

    def c_17_active( self,event=None):
        print('c_17 value = '+c_17_val.get())
        center_coordinates = (272,301)
        if c_17_val.get()=='1':
            self.highlight(255,272,301)
        else :
            self.highlight(0,272,301)
        sys.stdout.flush()    

    def c_18_active( self,event=None):
        print('c_18 value = '+c_18_val.get())
        center_coordinates = (377,300)
        if c_18_val.get()=='1':
            self.highlight(255,320,300)
        else :
            self.highlight(0,320,300)
        sys.stdout.flush()    #########################################################


    def c_19_active( self,event=None):
        print('c_19 value = '+c_19_val.get())
        center_coordinates = (173,317)
        if c_19_val.get()=='1':
            self.highlight(255,173,352)
        else :
            self.highlight(0,173,352)
        sys.stdout.flush()    #########################################################

    def c_20_active( self,event=None):
        print('c_20 value = '+c_20_val.get())
        center_coordinates = (224, 300)
        if c_20_val.get()=='1':
            self.highlight(255,224, 352)
        else :
            self.highlight(0,224, 352)
        sys.stdout.flush()  #########################################################  

    def c_21_active( self,event=None):
        print('c_21 value = '+c_21_val.get())
        center_coordinates = (272 ,352)
        if c_21_val.get()=='1':
            self.highlight(255,272 ,352)
        else :
            self.highlight(0,272 ,352)
        sys.stdout.flush()   ######################################################### 



    def c_22_active( self,event=None):
        print('c_22 value = '+c_22_val.get())
        center_coordinates = (224, 388)
        if c_22_val.get()=='1':
            self.highlight(255,222, 397)
        else :
            self.highlight(0,222, 397)
        sys.stdout.flush()    #########################################################


    def highlight(self,color,x,y):
        color = (0,color,0)
        center_coordinates = (x, y)
        thickness = 2
        radius =12
        try:
            self.img =cv2.circle(self.img, center_coordinates, radius, color, thickness)

            self.im = Image.fromarray(self.img)
            self.imgtk = ImageTk.PhotoImage(image=self.im)
            self.Canvas1.delete("all")
            self.Canvas1.create_image(0, 0, anchor=tk.NW, image=self.imgtk)
        except :
            print("failed")
            
    def selc_all(self,event=None):

        c_01_val.set('1')
        
        c_02_val.set('1') 
        c_03_val.set('1') 
        c_04_val.set('1') 
        c_05_val.set('1')
        c_06_val.set('1')
        
        c_07_val.set('1')
        c_08_val.set('1')
        c_09_val.set('1')
        c_10_val.set('1')
        c_11_val.set('1')
        c_12_val.set('1')
        c_13_val.set('1')
        
        c_14_val.set('1') 
        c_15_val.set('1') 
        c_16_val.set('1') 
        c_17_val.set('1') 
        c_18_val.set('1')
             
        c_19_val.set('1')
        c_20_val.set('1')
        c_21_val.set('1')
        
        c_22_val.set('1')

        self.all_action()        
        
    def uncheck_func(self,event=None):
        c_01_val.set('0')
        
        c_02_val.set('0') 
        c_03_val.set('0') 
        c_04_val.set('0') 
        c_05_val.set('0')
        c_06_val.set('0')
        
        c_07_val.set('0')
        c_08_val.set('0')
        c_09_val.set('0')
        c_10_val.set('0')
        c_11_val.set('0')
        c_12_val.set('0')
        c_13_val.set('0')
        
        c_14_val.set('0') 
        c_15_val.set('0') 
        c_16_val.set('0') 
        c_17_val.set('0') 
        c_18_val.set('0')
             
        c_19_val.set('0')
        c_20_val.set('0')
        c_21_val.set('0')
        
        c_22_val.set('0')
        
        self.all_action()
        
    def all_action(self):
        self.c_01_active()
        
        self.c_02_active() 
        self.c_03_active() 
        self.c_04_active() 
        self.c_05_active()
        self.c_06_active()
        
        self.c_07_active()
        self.c_08_active()
        self.c_09_active()
        self.c_10_active()
        self.c_11_active()
        self.c_12_active()
        self.c_13_active()
        
        self.c_14_active() 
        self.c_15_active() 
        self.c_16_active() 
        self.c_17_active() 
        self.c_18_active()
             
        self.c_19_active()
        self.c_20_active()
        self.c_21_active()
        
        self.c_22_active()

        
    def finalize_input(self,event=None):
        f = open("images/selected_Channels.txt", "w")

        f.write(c_01_val.get()+"\n")

        f.write(c_02_val.get()+"\n")
        f.write(c_03_val.get()+"\n")
        f.write(c_04_val.get()+"\n")
        f.write(c_05_val.get()+"\n")
        f.write(c_06_val.get()+"\n")
        
        f.write(c_07_val.get()+"\n")
        f.write(c_08_val.get()+"\n")
        f.write(c_09_val.get()+"\n")
        f.write(c_10_val.get()+"\n")
        f.write(c_11_val.get()+"\n")
        f.write(c_12_val.get()+"\n")
        f.write(c_13_val.get()+"\n")
        
        f.write(c_14_val.get()+"\n")
        f.write(c_15_val.get()+"\n")
        f.write(c_16_val.get()+"\n")
        f.write(c_17_val.get()+"\n")
        f.write(c_18_val.get()+"\n")

        f.write(c_19_val.get()+"\n")
        f.write(c_20_val.get()+"\n")
        f.write(c_21_val.get()+"\n")

        f.write(c_22_val.get()+"\n")
           
        f.close()
        w.destroy()

    def cncl_f(self,event):
        
        MsgBox = tk.messagebox.askquestion ('Exit Manual Selection','Are you sure you want to exit? The changes will not be saved.',icon = 'warning')
        if MsgBox == 'yes':
            w.destroy()
        
    def __init__(self, top=None):

        global c_11_val
        c_11_val = tk.StringVar()
        global c_01_val
        c_01_val = tk.StringVar()

        global c_07_val
        c_07_val = tk.StringVar()
        
        global c_08_val
        c_08_val = tk.StringVar()
        global c_10_val
        c_10_val = tk.StringVar()
        global c_09_val
        c_09_val = tk.StringVar()
        global c_19_val
        c_19_val = tk.StringVar()
        global c_20_val
        c_20_val = tk.StringVar()
        global c_21_val
        c_21_val = tk.StringVar()


        global c_22_val
        c_22_val = tk.StringVar()


        global c_12_val
        c_12_val = tk.StringVar()
        global c_13_val
        c_13_val = tk.StringVar()
        global c_14_val
        c_14_val = tk.StringVar()
        global c_15_val
        c_15_val = tk.StringVar()
        global c_16_val
        c_16_val = tk.StringVar()
        global c_17_val
        c_17_val = tk.StringVar()
        global c_18_val
        c_18_val = tk.StringVar()
        global c_02_val
        c_02_val = tk.StringVar()
        global c_03_val
        c_03_val = tk.StringVar()
        global c_04_val
        c_04_val = tk.StringVar()
        global c_05_val
        c_05_val = tk.StringVar()
        global c_06_val
        c_06_val = tk.StringVar()
        

        self.selc_all()


        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'

        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1100x466+255+143")
        top.minsize(148, 1)
        top.maxsize(1924, 1055)
        top.resizable(1, 0)
        top.title("Manual Channel Selection")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")
        
        p1 = ImageTk.PhotoImage(file = 'images/icon.png')
        top.iconphoto(False, p1)
        

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.018, rely=0.043, height=36, width=592)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(activeforeground="black")
        self.Label1.configure(background="#b9bef9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Segoe UI} -size 11 -weight bold")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Manual  Channel selection''')

        self.style.map('TCheckbutton',background=
            [('selected', _bgcolor), ('active', _ana2color)])
        
        
        # self.fp1_cb = ttk.Checkbutton(top)
        # self.fp1_cb.place(relx=0.179, rely=0.172, relwidth=0.042, relheight=0.0
        #         , height=26)
        # self.fp1_cb.configure(variable= fp1_val,onvalue = 1, offvalue = 0)
         
        # self.fp1_cb.configure(command= self.fp1_active)
        # self.fp1_cb.configure(takefocus="")
        # self.fp1_cb.configure(text='''Fp1''')

        # self.fp2_cb = ttk.Checkbutton(top)
        # self.fp2_cb.place(relx=0.304, rely=0.172, relwidth=0.064, relheight=0.0
        #         , height=26)
        # self.fp2_cb.configure(variable= fp2_val)
        # self.fp2_cb.configure(command= self.fp2_active)
        # self.fp2_cb.configure(takefocus="")
        # self.fp2_cb.configure(text='''Fp2''')

        # self.f3_cb = ttk.Checkbutton(top)
        # self.f3_cb.place(relx=0.17, rely=0.258, relwidth=0.037, relheight=0.0
        #         , height=26)
        # self.f3_cb.configure(variable= f3_val)
        # self.f3_cb.configure(command= self.f3_active)
        # self.f3_cb.configure(takefocus="")
        # self.f3_cb.configure(text='''F3''')

        # self.f4_cb = ttk.Checkbutton(top)
        # self.f4_cb.place(relx=0.322, rely=0.258, relwidth=0.064, relheight=0.0
        #         , height=26)
        # self.f4_cb.configure(variable= f4_val)
        # self.f4_cb.configure(command= self.f4_active)
        # self.f4_cb.configure(takefocus="")
        # self.f4_cb.configure(text='''F4''')

        # self.f7_cb = ttk.Checkbutton(top)
        # self.f7_cb.place(relx=0.125, rely=0.258, relwidth=0.046, relheight=0.0
        #         , height=26)
        # self.f7_cb.configure(variable= f7_val)
        # self.f7_cb.configure(command= self.f7_active)
        # self.f7_cb.configure(takefocus="")
        # self.f7_cb.configure(text='''F7''')





        # self.f8_cb = ttk.Checkbutton(top)
        # self.f8_cb.place(relx=0.376, rely=0.258, relwidth=0.064, relheight=0.0
        #         , height=26)
        # self.f8_cb.configure(variable= f8_val)
        # self.f8_cb.configure(command= self.f8_active)
        # self.f8_cb.configure(takefocus="")
        # self.f8_cb.configure(text='''F8''')
        self.c_01_cb = ttk.Checkbutton(top)
        self.c_01_cb.place(relx=0.25, rely=0.258, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_01_cb.configure(variable= c_01_val)
        self.c_01_cb.configure(command= self.c_01_active)
        self.c_01_cb.configure(takefocus="")
        self.c_01_cb.configure(text='''1''')
        
        
        self.c_02_cb = ttk.Checkbutton(top)
        self.c_02_cb.place(relx=0.100, rely=0.343, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_02_cb.configure(variable= c_02_val)
        self.c_02_cb.configure(command= self.c_02_active)
        self.c_02_cb.configure(takefocus="")
        self.c_02_cb.configure(text='''2''')

        self.c_03_cb = ttk.Checkbutton(top)
        self.c_03_cb.place(relx=0.175, rely=0.343, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_03_cb.configure(variable= c_03_val)
        self.c_03_cb.configure(command= self.c_03_active)
        self.c_03_cb.configure(takefocus="")
        self.c_03_cb.configure(text='''3''')

        self.c_04_cb = ttk.Checkbutton(top)
        self.c_04_cb.place(relx=0.25, rely=0.343, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_04_cb.configure(variable= c_04_val)
        self.c_04_cb.configure(command= self.c_04_active)
        self.c_04_cb.configure(takefocus="")
        self.c_04_cb.configure(text='''4''')

        self.c_05_cb = ttk.Checkbutton(top)
        self.c_05_cb.place(relx=0.325, rely=0.343, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_05_cb.configure(variable= c_05_val)
        self.c_05_cb.configure(command= self.c_05_active)
        self.c_05_cb.configure(takefocus="")
        self.c_05_cb.configure(text='''5''')

        self.c_06_cb = ttk.Checkbutton(top)
        self.c_06_cb.place(relx=0.400, rely=0.343, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_06_cb.configure(variable= c_06_val)
        self.c_06_cb.configure(command= self.c_06_active)
        self.c_06_cb.configure(takefocus="")
        self.c_06_cb.configure(text='''6''')

        self.c_07_cb = ttk.Checkbutton(top)
        self.c_07_cb.place(relx=0.050, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_07_cb.configure(variable= c_07_val)
        self.c_07_cb.configure(command= self.c_07_active)
        self.c_07_cb.configure(takefocus="")
        self.c_07_cb.configure(text='''7''')

        self.c_08_cb = ttk.Checkbutton(top)
        self.c_08_cb.place(relx=0.100, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_08_cb.configure(variable= c_08_val)
        self.c_08_cb.configure(command= self.c_08_active)
        self.c_08_cb.configure(takefocus="")
        self.c_08_cb.configure(text='''8''')

        self.c_09_cb = ttk.Checkbutton(top)
        self.c_09_cb.place(relx=0.175, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_09_cb.configure(variable= c_09_val)
        self.c_09_cb.configure(command= self.c_09_active)
        self.c_09_cb.configure(takefocus="")
        self.c_09_cb.configure(text='''9''')

        self.c_10_cb = ttk.Checkbutton(top)
        self.c_10_cb.place(relx=0.25, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_10_cb.configure(variable= c_10_val)
        self.c_10_cb.configure(command= self.c_10_active)
        self.c_10_cb.configure(takefocus="")
        self.c_10_cb.configure(text='''10''')

        self.c_11_cb = ttk.Checkbutton(top)
        self.c_11_cb.place(relx=0.325, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_11_cb.configure(variable= c_11_val)
        self.c_11_cb.configure(command= self.c_11_active)
        self.c_11_cb.configure(takefocus="")
        self.c_11_cb.configure(text='''11''')
        
        self.c_12_cb = ttk.Checkbutton(top)
        self.c_12_cb.place(relx=0.400, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_12_cb.configure(variable= c_12_val)
        self.c_12_cb.configure(command= self.c_12_active)
        self.c_12_cb.configure(takefocus="")
        self.c_12_cb.configure(text='''12''')

        self.c_13_cb = ttk.Checkbutton(top)
        self.c_13_cb.place(relx=0.460, rely=0.429, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_13_cb.configure(variable= c_13_val)
        self.c_13_cb.configure(command= self.c_13_active)
        self.c_13_cb.configure(takefocus="")
        self.c_13_cb.configure(text='''13''')

        self.c_14_cb = ttk.Checkbutton(top)
        self.c_14_cb.place(relx=0.100, rely=0.515, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_14_cb.configure(variable= c_14_val)
        self.c_14_cb.configure(command= self.c_14_active)
        self.c_14_cb.configure(takefocus="")
        self.c_14_cb.configure(text='''14''')

        self.c_15_cb = ttk.Checkbutton(top)
        self.c_15_cb.place(relx=0.175, rely=0.515, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_15_cb.configure(variable= c_15_val)
        self.c_15_cb.configure(command= self.c_15_active)
        self.c_15_cb.configure(takefocus="")
        self.c_15_cb.configure(text='''15''')

        self.c_16_cb = ttk.Checkbutton(top)
        self.c_16_cb.place(relx=0.25, rely=0.515,relwidth=0.055, relheight=0.0
                , height=26)
        self.c_16_cb.configure(variable= c_16_val)
        self.c_16_cb.configure(command= self.c_16_active)
        self.c_16_cb.configure(takefocus="")
        self.c_16_cb.configure(text='''16''')

        self.c_17_cb = ttk.Checkbutton(top)
        self.c_17_cb.place(relx=0.325, rely=0.515, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_17_cb.configure(variable= c_17_val)
        self.c_17_cb.configure(command= self.c_17_active)
        self.c_17_cb.configure(takefocus="")
        self.c_17_cb.configure(text='''17''')

        self.c_18_cb = ttk.Checkbutton(top)
        self.c_18_cb.place(relx=0.400, rely=0.515, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_18_cb.configure(variable= c_18_val)
        self.c_18_cb.configure(command= self.c_18_active)
        self.c_18_cb.configure(takefocus="")
        self.c_18_cb.configure(text='''18''')


        self.c_19_cb = ttk.Checkbutton(top)
        self.c_19_cb.place(relx=0.175, rely=0.601, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_19_cb.configure(variable= c_19_val)
        self.c_19_cb.configure(command= self.c_19_active)
        self.c_19_cb.configure(takefocus="")
        self.c_19_cb.configure(text='''19''')

        self.c_20_cb = ttk.Checkbutton(top)
        self.c_20_cb.place(relx=0.25, rely=0.601, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_20_cb.configure(variable= c_20_val)
        self.c_20_cb.configure(command= self.c_20_active)
        self.c_20_cb.configure(takefocus="")
        self.c_20_cb.configure(text='''20''')

        self.c_21_cb = ttk.Checkbutton(top)
        self.c_21_cb.place(relx=0.325, rely=0.601, relwidth=0.055, relheight=0.0
                , height=26)
        self.c_21_cb.configure(variable= c_21_val)
        self.c_21_cb.configure(command= self.c_21_active)
        self.c_21_cb.configure(takefocus="")
        self.c_21_cb.configure(text='''21''')
        
        
        self.c_22_cb = ttk.Checkbutton(top)
        self.c_22_cb.place(relx=0.25, rely=0.687, relwidth=0.046, relheight=0.0
                , height=26)
        self.c_22_cb.configure(variable= c_22_val)
        self.c_22_cb.configure(command= self.c_22_active)
        self.c_22_cb.configure(takefocus="")
        self.c_22_cb.configure(text='''22''')

        # self.t6_cb = ttk.Checkbutton(top)
        # self.t6_cb.place(relx=0.385, rely=0.601, relwidth=0.064, relheight=0.0
        #         , height=26)
        # self.t6_cb.configure(variable= t6_val)
        # self.t6_cb.configure(command= self.t6_active)
        # self.t6_cb.configure(takefocus="")
        # self.t6_cb.configure(text='''T6''')

        # self.o1_cb = ttk.Checkbutton(top)
        # self.o1_cb.place(relx=0.179, rely=0.687, relwidth=0.046, relheight=0.0
        #         , height=26)
        # self.o1_cb.configure(variable= o1_val)
        # self.o1_cb.configure(command= self.o1_active)
        # self.o1_cb.configure(takefocus="")
        # self.o1_cb.configure(text='''O1''')



        # self.o2_cb = ttk.Checkbutton(top)
        # self.o2_cb.place(relx=0.322, rely=0.687, relwidth=0.064, relheight=0.0
        #         , height=26)
        # self.o2_cb.configure(variable= o2_val)
        # self.o2_cb.configure(command= self.o2_active)
        # self.o2_cb.configure(takefocus="")
        # self.o2_cb.configure(text='''O2''')

        # self.t5_cb = ttk.Checkbutton(top)
        # self.t5_cb.place(relx=0.125, rely=0.601, relwidth=0.046, relheight=0.0
        #         , height=26)
        # self.t5_cb.configure(variable= t5_val)
        # self.t5_cb.configure(command= self.t5_active)
        # self.t5_cb.configure(takefocus="")
        # self.t5_cb.configure(text='''T5''')

#        self.fpz_cb = ttk.Checkbutton(top)
#        self.fpz_cb.place(relx=0.25, rely=0.172, relwidth=0.046, relheight=0.0
#                , height=26)
#        self.fpz_cb.configure(variable= fpz_val)
#        self.fpz_cb.configure(command= fpz_active)
#        self.fpz_cb.configure(takefocus="")
#        self.fpz_cb.configure(text='''FPZ''')


        
###################################################################################################
        self.sel_all = tk.Button(top)
        self.sel_all.place(relx=0.072, rely=0.815, height=33, width=76)
        self.sel_all.configure(activebackground="#ececec")
        self.sel_all.configure(activeforeground="#000000")
        self.sel_all.configure(background="#8cff8c")
        self.sel_all.configure(disabledforeground="#a3a3a3")
        self.sel_all.configure(foreground="#000000")
        self.sel_all.configure(highlightbackground="#d9d9d9")
        self.sel_all.configure(highlightcolor="black")
        self.sel_all.configure(pady="0")
        self.sel_all.configure(text='''Check All''')
        self.sel_all.bind("<Button-1>",self.selc_all)

        self.uncheck_all = tk.Button(top)
        self.uncheck_all.place(relx=0.17, rely=0.815, height=33, width=96)
        self.uncheck_all.configure(activebackground="#ececec")
        self.uncheck_all.configure(activeforeground="#000000")
        self.uncheck_all.configure(background="#fb5177")
        self.uncheck_all.configure(disabledforeground="#a3a3a3")
        self.uncheck_all.configure(foreground="#000000")
        self.uncheck_all.configure(highlightbackground="#d9d9d9")
        self.uncheck_all.configure(highlightcolor="black")
        self.uncheck_all.configure(pady="0")
        self.uncheck_all.configure(text='''Uncheck All''')
        self.uncheck_all.bind("<Button-1>",self.uncheck_func)
        
        self.selection_finished = tk.Button(top)
        self.selection_finished.place(relx=0.295, rely=0.815, height=33
                , width=85)
        self.selection_finished.configure(activebackground="#ececec")
        self.selection_finished.configure(activeforeground="#000000")
        self.selection_finished.configure(background="#00db37")
        self.selection_finished.configure(disabledforeground="#a3a3a3")
        self.selection_finished.configure(foreground="#000000")
        self.selection_finished.configure(highlightbackground="#d9d9d9")
        self.selection_finished.configure(highlightcolor="black")
        self.selection_finished.configure(pady="0")
        self.selection_finished.configure(text='''Okay''')
        self.selection_finished.bind("<Button-1>",self.finalize_input)

        self.cancel = tk.Button(top)
        self.cancel.place(relx=0.411, rely=0.815, height=33, width=86)
        self.cancel.configure(activebackground="#ececec")
        self.cancel.configure(activeforeground="#000000")
        self.cancel.configure(background="#ff0000")
        self.cancel.configure(disabledforeground="#a3a3a3")
        self.cancel.configure(foreground="#000000")
        self.cancel.configure(highlightbackground="#d9d9d9")
        self.cancel.configure(highlightcolor="black")
        self.cancel.configure(pady="0")
        self.cancel.configure(text='''Cancel''')
        self.cancel.bind("<Button>",self.cncl_f)
        
        self.Canvas1 = tk.Canvas(top)
        self.Canvas1.place(relx=0.564, rely=0.021, height=450
                , width=450)
        self.Canvas1.configure(background="#d9d9d9")
        self.Canvas1.configure(borderwidth="3")
        self.Canvas1.configure(closeenough="2.0")
        self.Canvas1.configure(cursor="circle")
        self.Canvas1.configure(highlightbackground="#d9d9d9")
        self.Canvas1.configure(highlightcolor="black")
        self.Canvas1.configure(insertbackground="black")
        self.Canvas1.configure(relief="ridge")
        self.Canvas1.configure(selectbackground="#c4c4c4")
        self.Canvas1.configure(selectforeground="black")


#####################################################################################
        
        self.img = cv2.imread('images/eeg_sensors2.jpg')
        self.img = cv2.resize(self.img, (450, 450)) 
        
        b,g,r = cv2.split(self.img)
        self.img = cv2.merge((r,g,b))
        
        self.im = Image.fromarray(self.img)
        self.imgtk = ImageTk.PhotoImage(image=self.im) 
        self.Canvas1.create_image(0, 0, anchor='nw', image=self.imgtk)
        
        self.selc_all()
        

       
if __name__ == '__main__':
    vp_start_gui()