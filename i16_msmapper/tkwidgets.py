# This Python code is encoded in: utf-8
"""
General tkinter widgets

By Dan Porter
Oct 2022
"""

import os, re
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

TF = ["Times", 12]  # entry
BF = ["Times", 14]  # Buttons
SF = ["Times New Roman", 14]  # Title labels
MF = ["Courier", 12]  # fixed distance format
LF = ["Times", 14]  # Labels
HF = ["Courier", 12]  # Text widgets (big)
TTF = ("Helvetica", 14, "bold italic")  # Title
bkg = 'white smoke'
ety = 'white'  # white
btn = 'azure'  # 'light slate blue'
opt = 'azure'  # 'light slate blue'
btn2 = 'gold'
btn_active = 'grey'
opt_active = 'grey'
txtcol = 'black'
btn_txt = 'black'
ety_txt = 'black'
opt_txt = 'black'
ttl_txt = 'red'


def popup_message(parent, title, message):
    """Create a message box"""
    root = tk.Toplevel(parent)
    root.title(title)
    #frame = tk.Frame(root)
    tk.Label(root, text=message, padx=20, pady=20).pack()
    #root.after(2000, root.destroy)
    return root


def popup_about():
    """Create about message"""
    from i16_msmapper import version_info, module_info
    msg = "%s\n\n" \
          "A simple GUI to run the msmapper code on Beamline I16." \
          "\n\n" \
          "Module Info:\n%s\n\n" \
          "By Dan Porter, Diamond Light Source Ltd" % (version_info(), module_info())
    messagebox.showinfo('About', msg)


def popup_help():
    """Create help message"""
    from i16_msmapper import doc_str
    return StringViewer(doc_str(), 'i16_peakfit Help', width=121)


def topmenu(root, menu_dict):
    """
    Add a menu to root
    :param root: tkinter root
    :param menu_dict: {Menu name: {Item name: function}}
    :return: None
    """
    """Setup menubar"""
    menubar = tk.Menu(root)

    for item in menu_dict:
        men = tk.Menu(menubar, tearoff=0)
        for label, function in menu_dict[item].items():
            men.add_command(label=label, command=function)
        menubar.add_cascade(label=item, menu=men)
    root.config(menu=menubar)


class StringViewer:
    """
    Simple GUI that displays strings
        StringViewer('output string', 'title', width, max_height)
    """

    def __init__(self, string, title='', width=40, max_height=40):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title(title)
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Textbox height
        height = string.count('\n')
        if height > 40: height = max_height

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # --- label ---
        # labframe = tk.Frame(frame,relief='groove')
        # labframe.pack(side=tk.TOP, fill=tk.X)
        # var = tk.Label(labframe, text=label_text,font=SF,justify='left')
        # var.pack(side=tk.LEFT)

        # --- Button ---
        frm1 = tk.Frame(frame)
        frm1.pack(side=tk.BOTTOM, fill=tk.X)
        var = tk.Button(frm1, text='Close', font=BF, command=self.fun_close, bg=btn, activebackground=btn_active)
        var.pack(fill=tk.X)

        # --- Text box ---
        frame_box = tk.Frame(frame)
        frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        scanx = tk.Scrollbar(frame_box, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)

        scany = tk.Scrollbar(frame_box)
        scany.pack(side=tk.RIGHT, fill=tk.Y)

        # Editable string box
        self.text = tk.Text(frame_box, width=width, height=height, font=HF, wrap=tk.NONE)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.text.insert(tk.END, string)

        self.text.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.text.xview)
        scany.config(command=self.text.yview)

    def fun_close(self):
        """close window"""
        self.root.destroy()


"------------------------------------------------------------------------"
"----------------------------Selection Box-------------------------------"
"------------------------------------------------------------------------"


class SelectionBox:
    """
    Displays all data fields and returns a selection
    Making a selection returns a list of field strings

    out = SelectionBox(['field1','field2','field3'], current_selection=['field2'], title='', multiselect=False).show()
    # Make selection and press "Select" > box disappears
    out = ['list','of','strings']
    """
    "------------------------------------------------------------------------"
    "--------------------------GUI Initilisation-----------------------------"
    "------------------------------------------------------------------------"

    def __init__(self, parent, data_fields, current_selection=(), title='Make a selection', multiselect=True):
        self.data_fields = data_fields
        self.initial_selection = current_selection

        # Create Tk inter instance
        self.root = tk.Toplevel(parent)
        self.root.wm_title(title)
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        self.output = []

        # Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, anchor=tk.N)

        "---------------------------ListBox---------------------------"
        # Eval box with scroll bar
        frm = tk.Frame(frame)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        sclx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        sclx.pack(side=tk.BOTTOM, fill=tk.BOTH)

        scly = tk.Scrollbar(frm)
        scly.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.lst_data = tk.Listbox(frm, font=MF, selectmode=tk.SINGLE, width=60, height=20, bg=ety,
                                   xscrollcommand=sclx.set, yscrollcommand=scly.set)
        self.lst_data.configure(exportselection=True)
        if multiselect:
            self.lst_data.configure(selectmode=tk.EXTENDED)
        self.lst_data.bind('<<ListboxSelect>>', self.fun_listboxselect)
        self.lst_data.bind('<Double-Button-1>', self.fun_exitbutton)

        # Populate list box
        for k in self.data_fields:
            # if k[0] == '_': continue # Omit _OrderedDict__root/map
            strval = '{}'.format(k)
            self.lst_data.insert(tk.END, strval)

        self.lst_data.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        for select in current_selection:
            if select in data_fields:
                idx = data_fields.index(select)
                self.lst_data.select_set(idx)

        sclx.config(command=self.lst_data.xview)
        scly.config(command=self.lst_data.yview)

        # self.txt_data.config(xscrollcommand=scl_datax.set,yscrollcommand=scl_datay.set)

        "----------------------------Search Field-----------------------------"
        frm = tk.LabelFrame(frame, text='Search', relief=tk.RIDGE)
        frm.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        self.searchbox = tk.StringVar(self.root, '')
        var = tk.Entry(frm, textvariable=self.searchbox, font=TF, bg=ety, fg=ety_txt)
        var.bind('<Key>', self.fun_search)
        var.pack(fill=tk.X, expand=tk.YES, padx=2, pady=2)

        "----------------------------Exit Button------------------------------"
        frm_btn = tk.Frame(frame)
        frm_btn.pack(fill=tk.X, expand=tk.YES)

        self.numberoffields = tk.StringVar(self.root, '%3d Selected Fields' % len(self.initial_selection))
        var = tk.Label(frm_btn, textvariable=self.numberoffields, width=20)
        var.pack(side=tk.LEFT)
        btn_exit = tk.Button(frm_btn, text='Select', font=BF, command=self.fun_exitbutton, bg=btn,
                             activebackground=btn_active)
        btn_exit.pack(side=tk.RIGHT)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        # self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def show(self):
        """Run the selection box, wait for response"""

        # self.root.deiconify()  # show window
        self.root.wait_window()  # wait for window
        return self.output

    def fun_search(self, event=None):
        """Search the selection for string"""
        search_str = self.searchbox.get()
        search_str = search_str + event.char
        search_str = search_str.strip().lower()
        if not search_str: return

        # Clear current selection
        self.lst_data.select_clear(0, tk.END)
        view_idx = None
        # Search for whole words first
        for n, item in enumerate(self.data_fields):
            if re.search(r'\b%s\b' % search_str, item.lower()):  # whole word search
                self.lst_data.select_set(n)
                view_idx = n
        # if nothing found, search anywhere
        if view_idx is None:
            for n, item in enumerate(self.data_fields):
                if search_str in item.lower():
                    self.lst_data.select_set(n)
                    view_idx = n
        if view_idx is not None:
            self.lst_data.see(view_idx)
        self.fun_listboxselect()

    def fun_listboxselect(self, event=None):
        """Update label on listbox selection"""
        self.numberoffields.set('%3d Selected Fields' % len(self.lst_data.curselection()))

    def fun_exitbutton(self, event=None):
        """Closes the current data window and generates output"""
        selection = self.lst_data.curselection()
        self.output = [self.data_fields[n] for n in selection]
        self.root.destroy()

    def f_exit(self, event=None):
        """Closes the current data window"""
        self.output = self.initial_selection
        self.root.destroy()


