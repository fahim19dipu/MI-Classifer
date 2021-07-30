#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# clss.classifier(self.path_traiin.get("1.0",'end-1c'), self.path_test.get("1.0",'end-1c'),
#                 UI_v_203_support.bandpass_cheby.get(),self.lowcut.get(),self.highcut.get(),
#                 self.ord.get(),self.fs.get(),self.n_cmp.get(),self.n_ft.get(),
#                 UI_v_203_support.is_keppa.get(),
#                 UI_v_203_support.svm_lin.get(),
#                 UI_v_203_support.svm_poli.get(),
#                 UI_v_203_support.svm_rbf.get(),
#                 UI_v_203_support.lda.get(),
#                 UI_v_203_support.rf.get(),
#                 UI_v_203_support.knn.get(),
#                 UI_v_203_support.dbn.get(),
#                 UI_v_203_support.ann.get(),
#                 UI_v_203_support.cnn.get()
#                 )
"""        
    def wrt_options(self):
        f = open("UI_selects.txt", "w")
        f.write(str(UI_v_203_support.auto_task_selection.get())+' autoselect\n')
        
        f.write(str(UI_v_203_support.lf_hnd.get())+' left hand\n')
        f.write(str(UI_v_203_support.rht_hnd.get())+' right hand \n')
        f.write(str(UI_v_203_support.feet.get())+' feet \n')
        f.write(str(UI_v_203_support.tng.get())+' Tongue\n')
        
        f.write(UI_v_203_support.combobox_train.get()+' train_type\n')
        f.write(UI_v_203_support.combobox_test.get()+' test_type\n')
        
        f.write(str(UI_v_203_support.bandpass_cheby.get())+' is cheby\n')
        f.write(str(self.lowcut.get())+' lowcut\n') 
        f.write(str(self.highcut.get())+' highcut\n') 
        f.write(str(self.ord.get())+' Order\n') 
        f.write(str(self.fs.get())+' Sampling Frequency\n')
        
        f.write(str(self.n_cmp.get())+' Number of component\n') 
        f.write(str(self.n_ft.get())+' Number of Feature\n') 
        
        f.write(str(UI_v_203_support.is_keppa.get())+' keppa\n') 

        f.write(str(UI_v_203_support.cnn.get())+' cnn\n')
        f.write(str(UI_v_203_support.ann.get())+' ann\n')
        f.write(str(UI_v_203_support.dbn.get())+' dbn\n')
        f.write(str(UI_v_203_support.knn.get())+' knn\n') 
        f.write(str(UI_v_203_support.lda.get())+' lda\n')
        f.write(str(UI_v_203_support.rf.get())+' rf\n')
        f.write(str(UI_v_203_support.svm_lin.get())+' svm lin\n')
        f.write(str(UI_v_203_support.svm_poli.get())+' svm poli\n')
        f.write(str(UI_v_203_support.svm_rbf.get())+' svm rbf\n') 
          
        f.close()
"""
       
# Support module generated by PAGE version 4.26
#  in conjunction with Tcl version 8.6
#    Feb 04, 2021 01:12:44 AM +06  platform: Windows NT

import sys

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

def set_Tk_var():
    
    global lf_hnd
    lf_hnd = tk.IntVar()

    
    global rht_hnd
    rht_hnd = tk.IntVar()

    global feet
    feet = tk.IntVar()

    global tng
    tng = tk.IntVar()

    global auto_task_selection
    
    auto_task_selection = tk.IntVar()
    global combobox_test,combobox_train
    combobox_test = tk.StringVar()
    combobox_train= tk.StringVar()
    
    global cnn
    cnn = tk.IntVar()
    global ann
    ann = tk.IntVar()
    global dbn
    dbn = tk.IntVar()
    global knn
    knn = tk.IntVar()
    global lda
    lda = tk.IntVar()
    global svm_lin
    svm_lin = tk.IntVar()
    global svm_poli
    svm_poli = tk.IntVar()    
    global svm_rbf
    svm_rbf = tk.IntVar()    
    global rf
    rf = tk.IntVar()
    
    global bandpass_cheby
    bandpass_cheby = tk.IntVar()
    
    global is_keppa
    is_keppa = tk.IntVar()
    
    global all_sel
    all_sel = tk.IntVar()

def set_Tk_val(event=None):
   
    lf_hnd.set(0)
    rht_hnd.set(0)
    feet.set(0)
    tng.set(0)
    #################################
    auto_task_selection.set(0)
    ##############################
    cnn.set(0)
    ann.set(0)
    dbn.set(0)
    knn.set(0) 
    lda.set(0) 
    rf.set(0)
    svm_lin.set(0)
    svm_poli.set(0)
    svm_rbf.set(0)
    bandpass_cheby.set(0)
    is_keppa.set(0)
    all_sel.set(0)

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import UI_v_203
    UI_v_203.vp_start_gui()




