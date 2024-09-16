"""
tkGui: MsMapper
"""
import os.path
import datetime

import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog

from i16_msmapper.tkwidgets import TF, BF, SF, TTF, HF, bkg, ety, btn, opt, btn_active, opt_active, txtcol, \
    ety_txt, popup_about, popup_help, topmenu
from i16_msmapper import mapper_runner
from i16_msmapper import mapper_plotter

TOPDIR = datetime.datetime.now().strftime('/dls/i16/data/%Y')


class MsMapperGui:
    """
    tkinter GUI: MSMapper
    Graphical user interface front-end for the MillerSpaceMapper program
    The msmapper program converts x-ray diffraction scans with area detectors into reciprocal space units.
    """

    def __init__(self, topdir=TOPDIR):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('I16 MSMapper')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)
        self.topdir = topdir
        mapper_plotter.set_plot_defaults()
        from i16_msmapper import title

        # Variables
        self.join_files = tk.BooleanVar(self.root, True)
        self.input_label = tk.StringVar(self.root, 'Scan Files\nto combine:')
        self.output_file = tk.StringVar(self.root, '')
        self.output_size = tk.StringVar(self.root, '')
        self.output_type = tk.StringVar(self.root, 'Volume_HKL')
        self.normby = tk.StringVar(self.root, 'None')
        self.polarisation = tk.BooleanVar(self.root, False)
        self.setbox = tk.BooleanVar(self.root, True)
        self.hkl_centre = tk.StringVar(self.root, '[0, 0, 0]')
        self.hkl_start = tk.StringVar(self.root, '[-0.1, -0.1, -0.1]')
        self.hkl_step = tk.StringVar(self.root, '[0.002, 0.002, 0.002]')
        self.box_size = tk.StringVar(self.root, '[100, 100, 100]')

        output_types = ['Volume_HKL', 'Volume_Q', 'Line_2Theta', 'Coords_HKL', 'Coords_Q']
        norm_types = ['None', 'rc', 'ic1monitor']
        width = 45
        self.hide_boxes = []  # hide boxes on setbox un-tick

        # Top menu
        menu = {
            'File': {
                'Select Scan File(s)': self.btn_browse,
                'Select Output File': self.btn_browse_output,
                'Quit': self.btn_close,
            },
            'Plot': {
                'Scan': self.btn_plot_scan,
                'Scan images (slider)': self.btn_plot_images,
                'Remap HKL': self.btn_plot_hkl,
                'Remap HKL (slider)': self.btn_plot_hkl_images,
                'Remap Q': self.btn_plot_q,
                'Remap Two-Theta': self.btn_plot_tth,
            },
            'Tools': {
                'rs_map command': self.menu_rsmap,
                'msmapper script': self.menu_msmapper,
                'Batch commands': self.menu_batch_commands,
                'Run batch commands': self.menu_run_batch,
                'set tmp directory': self.menu_settmp,
            },
            'Help': {
                'Docs': popup_help,
                'About': popup_about,
            }
        }
        topmenu(self.root, menu)

        # Create Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        var = tk.Label(frame, text=title(), font=TTF, foreground='red')
        var.pack(side=tk.TOP)

        # --- Input Files ---
        top = tk.Frame(frame, relief='groove')
        top.pack(side=tk.TOP, fill=tk.BOTH)
        # label
        frm = tk.Frame(top)
        frm.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Label(frm, textvariable=self.input_label, width=15, font=SF, justify='right')
        var.pack(side=tk.TOP)
        var = tk.Checkbutton(frm, text='Join Scans', variable=self.join_files, font=TF, command=self.tck_join)
        var.pack(side=tk.TOP)
        # textbox
        frm = tk.Frame(top)
        frm.pack(side=tk.LEFT, fill=tk.Y)
        scanx = tk.Scrollbar(frm, orient=tk.HORIZONTAL)
        scanx.pack(side=tk.BOTTOM, fill=tk.X)
        scany = tk.Scrollbar(frm)
        scany.pack(side=tk.RIGHT, fill=tk.Y)
        # Editable string box
        self.files = tk.Text(frm, width=width, height=4, font=HF, wrap=tk.NONE)
        self.files.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        # self.files.insert(tk.END, "")
        self.files.config(xscrollcommand=scanx.set, yscrollcommand=scany.set)
        scanx.config(command=self.files.xview)
        scany.config(command=self.files.yview)
        # Button
        var = tk.Button(top, text='Browse', font=BF, command=self.btn_browse, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)

        # --- Output File ---
        top = tk.Frame(frame, relief='groove')
        top.pack(side=tk.TOP, fill=tk.BOTH)
        # label
        var = tk.Label(top, text='Output File: ', width=15, font=SF, justify='right')
        var.pack(side=tk.LEFT)
        # textbox
        var = tk.Entry(top, textvariable=self.output_file, font=TF, width=int(width * 1.3), bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        # Button
        var = tk.Button(top, text='SaveAs', font=BF, command=self.btn_saveas, bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y)

        # size
        var = tk.Label(top, textvariable=self.output_size, font=SF)
        var.pack(side=tk.LEFT)

        # --- Options box ---
        mid = tk.LabelFrame(frame, text='Options', relief='groove')
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Output options
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, expand=True)
        var = tk.OptionMenu(frm, self.output_type, *output_types)
        var.config(font=SF, width=14, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)

        # Normalise by
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, expand=True)
        var = tk.Label(frm, text='Normalisation: ', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.OptionMenu(frm, self.normby, *norm_types)
        var.config(font=SF, width=8, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Checkbutton(frm, text='Polarisation', variable=self.polarisation, font=SF)
        var.pack(side=tk.LEFT, padx=6)
        var = tk.Checkbutton(frm, text='set HKL Box', variable=self.setbox, font=SF, command=self.tck_hide_hkl)
        var.pack(side=tk.LEFT, padx=6)

        # hkl step
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(frm, text='HKL step: ', width=15, font=SF, justify='right')
        var.pack(side=tk.LEFT)
        var = tk.Entry(frm, textvariable=self.hkl_step, font=TF, width=25, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frm, text='Calculate min step', font=BF, command=self.btn_get_step, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # hkl cen
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(frm, text='HKL centre: ', width=15, font=SF, justify='right')
        var.pack(side=tk.LEFT)
        self.hide_boxes.append(var)
        var = tk.Entry(frm, textvariable=self.hkl_centre, font=TF, width=25, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        self.hide_boxes.append(var)
        var = tk.Button(frm, text='Get from Nexus', font=BF, command=self.btn_nexus_hkl, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # hkl start
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(frm, text='HKL start: ', width=15, font=SF, justify='right')
        var.pack(side=tk.LEFT)
        self.hide_boxes.append(var)
        var = tk.Entry(frm, textvariable=self.hkl_start, font=TF, width=25, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        self.hide_boxes.append(var)

        # box size
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, fill=tk.X)
        var = tk.Label(frm, text='Box size: ', width=15, font=SF, justify='right')
        var.pack(side=tk.LEFT)
        self.hide_boxes.append(var)
        var = tk.Entry(frm, textvariable=self.box_size, font=TF, width=25, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        self.hide_boxes.append(var)

        # --- Run ---
        bot = tk.Frame(frame)
        bot.pack(side=tk.TOP, fill=tk.BOTH)
        var = tk.Button(bot, text='Run MSMapper', font=BF, command=self.btn_run, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

        # --- Plot ---
        bot = tk.LabelFrame(frame, text='Plotting', relief='groove')
        bot.pack(side=tk.TOP, fill=tk.BOTH)
        var = tk.Button(bot, text='Scan', font=BF, command=self.btn_plot_scan, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(bot, text='Images', font=BF, command=self.btn_plot_images, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(bot, text='HKL', font=BF, command=self.btn_plot_hkl, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(bot, text='Slider', font=BF, command=self.btn_plot_hkl_images, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(bot, text='Q', font=BF, command=self.btn_plot_q, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES)
        var = tk.Button(bot, text='Two-Theta', font=BF, command=self.btn_plot_tth, bg=btn,
                        activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.btn_close)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def get_files(self):
        """Get files"""
        return [file for file in self.files.get('1.0', tk.END).split('\n') if file.strip()]

    def set_output_file(self):
        files = self.get_files()
        if len(files) == 0:
            return
        if not self.join_files.get():
            path = os.path.dirname(files[0])
            output = os.path.join(path, 'processing')
        elif len(files) == 1:
            path, name = os.path.split(files[0])
            name, ext = os.path.splitext(name)
            output = os.path.join(path, 'processing', name + '_rsmap' + ext)
        else:
            path, name1 = os.path.split(files[0])
            name1, ext = os.path.splitext(name1)
            name2 = os.path.basename(files[-1])
            name2, ext = os.path.splitext(name2)
            output = os.path.join(path, 'processing', f"{name1}-{name2}_rsmap{ext}")
        self.output_file.set(output)

    def get_hkl(self):
        """Get start, step, size"""
        hkl_start = np.array(eval(self.hkl_start.get()), dtype=float).reshape(-1).tolist()
        hkl_step = np.array(eval(self.hkl_step.get()), dtype=float).reshape(-1).tolist()
        box_size = np.array(eval(self.box_size.get()), dtype=int).reshape(-1).tolist()
        return hkl_start, hkl_step, box_size

    def get_options(self):
        """Get options"""
        outp = self.output_type.get()
        norm = self.normby.get()
        norm = None if norm == 'None' else norm
        pol = self.polarisation.get()
        pol = None if not pol else True
        return outp, norm, pol

    def get_output_size(self):
        """Get size of output file"""
        output_file = self.output_file.get()
        try:
            size = os.path.getsize(output_file)
            self.output_size.set('%.2f MB' % (size / 1048576))
        except OSError:
            pass

    "------------------------------------------------------------------------"
    "----------------------------Menu Functions------------------------------"
    "------------------------------------------------------------------------"

    def menu_rsmap(self):
        """Menu item rsmap command"""
        files = self.get_files()
        output_file = self.output_file.get()
        hkl_start, hkl_step, box_size = self.get_hkl()
        rsmap = mapper_runner.rsmap_command(files, output_file, hkl_start, box_size, hkl_step)

        screenwidth = self.root.winfo_screenwidth()
        width = len(rsmap) + 2 if len(rsmap) < screenwidth * 0.67 else screenwidth // 2
        from i16_msmapper.tkwidgets import StringViewer
        StringViewer(rsmap, 'i16 msmapper', width=width, max_height=2)

    def menu_msmapper(self):
        """Menu item msmapper script"""
        files = self.get_files()
        output_file = self.output_file.get()
        hkl_start, hkl_step, box_size = self.get_hkl()
        script = mapper_runner.msmapper_script(files, output_file, hkl_start, box_size, hkl_step)

        from i16_msmapper.tkwidgets import StringViewer
        StringViewer(script, 'i16 msmapper', width=101, max_height=12)

    def menu_batch_commands(self):
        """Menu item batch commands"""
        files = self.get_files()
        output_file = self.output_file.get()
        output_directory = os.path.dirname(output_file)
        hkl_start, hkl_step, box_size = self.get_hkl()
        script = '\n'.join(mapper_runner.rsmap_batch(files, output_directory, step=hkl_step))

        from i16_msmapper.tkwidgets import StringViewer
        StringViewer(script, 'i16 msmapper', width=101, max_height=12)

    def menu_run_batch(self):
        """Menu item run batch"""
        files = self.get_files()
        output_file = self.output_file.get()
        output_directory = os.path.dirname(output_file)
        hkl_start, hkl_step, box_size = self.get_hkl()
        cmd_list = mapper_runner.rsmap_batch(files, output_directory, step=hkl_step)
        mapper_runner.batch_commands(cmd_list)

    def menu_settmp(self):
        """Menu item set temp directory"""
        # new_dir = simpledialog.askstring(
        #     title='i16 msmapper',
        #     prompt='Enter the temp directory',
        #     initialvalue=mapper_runner.TEMPDIR,
        # )
        new_dir = filedialog.askdirectory(
            title='Select TMP directory',
            initialdir=mapper_runner.TEMPDIR,
        )
        if new_dir:
            new_dir = os.path.abspath(os.path.expanduser(new_dir))
            if os.path.isdir(new_dir) and os.access(new_dir, os.W_OK):
                mapper_runner.TEMPDIR = new_dir
                messagebox.showinfo('i16 mapper', 'New Temp dir is set:\n%s' % new_dir)
            else:
                messagebox.showinfo('i16 mapper', 'Directory does not exist or is not writable:\n%s' % new_dir)

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def btn_browse(self):
        """File browse"""
        filename = filedialog.askopenfilenames(
            title='Open I16 Scan file',
            initialdir=self.topdir,
            filetypes=(("Nexus files", "*.nxs"), ("All files", "*.*"))
        )
        if filename:
            self.files.delete('1.0', tk.END)
            self.files.insert(tk.END, '\n'.join(filename))
            # Set saveas
            self.set_output_file()
            self.topdir = os.path.dirname(filename[0])

    def btn_browse_output(self):
        """File browse for output"""
        filename = filedialog.askopenfilename(
            title='Open msmapper file',
            initialdir=self.topdir,
            filetypes=(("Nexus files", "*.nxs"), ("HDF files", "*.hdf"), ("All files", "*.*"))
        )
        if filename:
            self.output_file.set(filename)
            self.get_output_size()
            self.topdir = os.path.dirname(filename)

    def btn_saveas(self):
        """Select output file"""
        filename = filedialog.asksaveasfilename(
            title='Save Remapped file as',
            initialfile=self.output_file.get(),
            defaultextension='*.nxs',
            filetypes=(("Nexus files", "*.nxs"), ("All files", "*.*"))
        )
        if filename:
            self.output_file.set(filename)
            self.get_output_size()
            self.topdir = os.path.dirname(filename)

    def tck_join(self, event=None):
        tick = self.join_files.get()
        if tick:
            self.input_label.set('Scan Files\n to combine:')
        else:
            self.input_label.set('Scan Files\n to process:')
            self.setbox.set(False)
        self.set_output_file()

    def tck_hide_hkl(self, event=None):
        tick = self.setbox.get()
        if tick:
            for var in self.hide_boxes:
                var['fg'] = 'black'
        else:
            for var in self.hide_boxes:
                var['fg'] = 'grey'

    def btn_nexus_hkl(self):
        """Get HKL start value from nexus file"""
        files = self.get_files()
        hkl_cen = mapper_runner.get_nexus_data(files[0])
        hkl_start, hkl_step, box_size = self.get_hkl()
        h, k, l = np.asarray(hkl_cen) - (np.asarray(hkl_step) * np.asarray(box_size) / 2.)
        hi, ki, li = hkl_cen
        self.hkl_centre.set(f"[{hi:.3f},{ki:.3f},{li:.3f}]")
        self.hkl_start.set(f"[{h:.3f},{k:.3f},{l:.3f}]")

    def btn_get_step(self):
        """Run msmapper to get minimum pixel step"""
        files = self.get_files()
        dh, dk, dl = mapper_runner.get_pixel_steps(files[0])
        self.hkl_step.set(f"[{dh:.4f}, {dk:.4f}, {dl:.4f}]")

    def btn_run(self):
        """Run MSMapper"""
        files = self.get_files()
        output = self.output_file.get()
        hkl_start, hkl_step, box_size = self.get_hkl()
        out_type, norm, pol = self.get_options()
        if self.join_files.get():
            if self.setbox.get():
                mapper_runner.run_msmapper(files, output, start=hkl_start,
                                           shape=box_size, step=hkl_step,
                                           output_mode=out_type, normalisation=norm,
                                           polarisation=pol, detector_region=None)
            else:
                mapper_runner.run_msmapper(files, output, start=None,
                                           shape=None, step=hkl_step,
                                           output_mode=out_type, normalisation=norm,
                                           polarisation=pol, detector_region=None)
        else:
            cmd_list = mapper_runner.rsmap_batch(files, output, step=hkl_step)
            mapper_runner.batch_commands(cmd_list)
        self.get_output_size()

    def btn_plot_scan(self):
        """Plot auto scan"""
        files = self.get_files()
        if files:
            mapper_plotter.plot_scan(files[0])

    def btn_plot_images(self):
        """Plot scan images"""
        files = self.get_files()
        if files:
            mapper_plotter.slider_scan(files[0])

    def btn_plot_hkl(self):
        """Plot HKL cuts and planes"""
        output_file = self.output_file.get()
        if output_file:
            mapper_plotter.plot_remap_hkl(output_file)

    def btn_plot_hkl_images(self):
        """Plot HKL planes with slider"""
        output_file = self.output_file.get()
        if output_file:
            mapper_plotter.slider_remap(output_file)

    def btn_plot_q(self):
        """Plot Q cuts and planes"""
        output_file = self.output_file.get()
        if output_file:
            mapper_plotter.plot_remap_q(output_file)

    def btn_plot_tth(self):
        """Plot magnitude of two-theta and q"""
        output_file = self.output_file.get()
        if output_file:
            mapper_plotter.plot_qmag(output_file)

    def btn_close(self):
        """close window"""
        self.root.destroy()
