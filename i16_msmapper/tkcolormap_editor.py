"""
Colormap editor
"""

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog

from i16_msmapper.tkwidgets import TF, BF, SF, TTF, HF, bkg, ety, btn, opt, btn_active, opt_active, txtcol, \
    ety_txt, popup_about, popup_help, topmenu


def get_clim():
    """Get current figure clim"""
    current_image = plt.gci()
    if current_image:
        return current_image.get_clim()
    print('No figure or image selected')
    return 0, 1


class ColourCutoffGui:
    """
    Change the vmin/vmax colormap limits of the current figure
    Activate form the console by typing:
        ColourCutoffs()
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initilisation-----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self):
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('I16 MSMapper Colour Maps')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        ini_vmin, ini_vmax = get_clim()

        # Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, anchor=tk.N)

        # GCF button
        frm_btn = tk.Button(frame, text='Get Current Figure', font=BF, command=self.f_gcf)
        frm_btn.pack(fill=tk.X)

        # increment setting
        f_inc = tk.Frame(frame)
        f_inc.pack(fill=tk.X)

        inc_btn1 = tk.Button(f_inc, text='1', font=BF, command=self.f_but1)
        inc_btn1.pack(side=tk.LEFT)

        inc_btn2 = tk.Button(f_inc, text='100', font=BF, command=self.f_but2)
        inc_btn2.pack(side=tk.LEFT)

        inc_btn3 = tk.Button(f_inc, text='1000', font=BF, command=self.f_but3)
        inc_btn3.pack(side=tk.LEFT)

        self.increment = tk.DoubleVar(f_inc, 1.0)
        inc_ety = tk.Entry(f_inc, textvariable=self.increment, width=6)
        inc_ety.pack(side=tk.LEFT)

        # Upper clim
        f_upper = tk.Frame(frame)
        f_upper.pack(fill=tk.X)

        up_left = tk.Button(f_upper, text='<', font=BF, command=self.f_upper_left)
        up_left.pack(side=tk.LEFT)

        self.vmin = tk.DoubleVar(f_upper, ini_vmin)
        up_edit = tk.Entry(f_upper, textvariable=self.vmin, width=12)
        up_edit.bind('<Return>', self.update)
        up_edit.bind('<KP_Enter>', self.update)
        up_edit.pack(side=tk.LEFT, expand=tk.YES)

        up_right = tk.Button(f_upper, text='>', font=BF, command=self.f_upper_right)
        up_right.pack(side=tk.LEFT)

        # Lower clim
        f_lower = tk.Frame(frame)
        f_lower.pack(fill=tk.X)

        lw_left = tk.Button(f_lower, text='<', font=BF, command=self.f_lower_left)
        lw_left.pack(side=tk.LEFT)

        self.vmax = tk.DoubleVar(f_lower, ini_vmax)
        lw_edit = tk.Entry(f_lower, textvariable=self.vmax, width=12)
        lw_edit.bind('<Return>', self.update)
        lw_edit.bind('<KP_Enter>', self.update)
        lw_edit.pack(side=tk.LEFT, expand=tk.YES)

        lw_right = tk.Button(f_lower, text='>', font=BF, command=self.f_lower_right)
        lw_right.pack(side=tk.LEFT)

        # Update button
        frm_btn = tk.Button(frame, text='Update', font=BF, command=self.update)
        frm_btn.pack(fill=tk.X)

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def f_gcf(self):
        # fig = plt.gcf()
        # fig.canvas.manager.window.raise_()

        new_vmin, new_vmax = get_clim()
        self.vmin.set(new_vmin)
        self.vmax.set(new_vmax)

    def f_but1(self):
        self.increment.set(1.0)

    def f_but2(self):
        self.increment.set(100)

    def f_but3(self):
        self.increment.set(1e3)

    def f_upper_left(self):
        inc = self.increment.get()
        cur_vmin = self.vmin.get()
        self.vmin.set(cur_vmin - inc)
        self.update()

    def f_upper_right(self):
        inc = self.increment.get()
        cur_vmin = self.vmin.get()
        self.vmin.set(cur_vmin + inc)
        self.update()

    def f_lower_left(self):
        inc = self.increment.get()
        cur_vmax = self.vmax.get()
        self.vmax.set(cur_vmax - inc)
        self.update()

    def f_lower_right(self):
        inc = self.increment.get()
        cur_vmax = self.vmax.get()
        self.vmax.set(cur_vmax + inc)
        self.update()

    def update(self, event=None):
        cur_vmin = self.vmin.get()
        cur_vmax = self.vmax.get()

        # fig = plt.gcf()
        # fig.canvas.manager.window.raise_()
        plt.clim(cur_vmin, cur_vmax)
        plt.show()