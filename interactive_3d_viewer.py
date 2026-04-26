import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import tkinter as tk
from tkinter import ttk, messagebox
from cylindrical_waveguide_solver import CylindricalWaveguide

class InteractiveWaveguideViewer:
    def __init__(self, wg, grid_params):
        self.wg = wg
        self.grid_params = grid_params # {Nr, Np, Nz}
        
        self.fig = plt.figure(figsize=(15, 8))
        # 3D plot areas (Electric and Magnetic)
        self.ax_E = self.fig.add_axes([0.05, 0.25, 0.35, 0.7], projection='3d')
        self.ax_H = self.fig.add_axes([0.42, 0.25, 0.35, 0.7], projection='3d')
        
        # Radio Buttons for Display Mode
        self.ax_mode_radio = self.fig.add_axes([0.82, 0.75, 0.15, 0.12])
        self.mode_radio = RadioButtons(self.ax_mode_radio, ('Cross Sections', 'Full Volume'))
        self.ax_mode_radio.set_title('Display Mode')
        
        # Radio Buttons for vector style
        self.ax_style_radio = self.fig.add_axes([0.82, 0.58, 0.15, 0.12])
        self.style_radio = RadioButtons(self.ax_style_radio, ('Uniform Length', 'Proportional Length'))
        self.ax_style_radio.set_title('Vector Style')
        
        # Sliders
        self.ax_phi = self.fig.add_axes([0.2, 0.15, 0.5, 0.03])
        self.s_phi = Slider(self.ax_phi, r'r-z Angle $\phi$ [deg]', 0, 360, valinit=0)
        
        self.ax_z = self.fig.add_axes([0.2, 0.1, 0.5, 0.03])
        self.s_z = Slider(self.ax_z, r'x-y Position $z$ [m]', 0, self.wg.L, valinit=self.wg.L/2)
        
        T = 1.0 / self.wg.f
        self.ax_t = self.fig.add_axes([0.2, 0.05, 0.5, 0.03])
        self.s_t = Slider(self.ax_t, r'Time $t/T$', 0, 1, valinit=0)
        
        # --- 全空間の最大振幅を求めてカラーバーを固定する ---
        r_test = np.linspace(0, self.wg.a, 10)
        phi_test = np.linspace(0, 2*np.pi, 12)
        z_test = np.linspace(0, self.wg.L, 10)
        R_t, P_t, Z_t = np.meshgrid(r_test, phi_test, z_test)
        Et_r, Et_p, Et_z, Ht_r, Ht_p, Ht_z = self.wg.get_fields(R_t, P_t, Z_t, 0)
        
        self.global_max_E = np.max(np.sqrt(np.abs(Et_r)**2 + np.abs(Et_p)**2 + np.abs(Et_z)**2))
        self.global_max_H = np.max(np.sqrt(np.abs(Ht_r)**2 + np.abs(Ht_p)**2 + np.abs(Ht_z)**2))
        if self.global_max_E == 0: self.global_max_E = 1.0
        if self.global_max_H == 0: self.global_max_H = 1.0

        # Draw waveguide cylinder
        self._draw_cylinder(self.ax_E, "Electric Field (E)")
        self._draw_cylinder(self.ax_H, "Magnetic Field (H)")
        
        # Store quiver objects
        self.q_E = None
        self.q_H = None
        
        # Connect events
        self.mode_radio.on_clicked(self.update)
        self.style_radio.on_clicked(self.update)
        self.s_phi.on_changed(self.update)
        self.s_z.on_changed(self.update)
        self.s_t.on_changed(self.update)
        
        # Colorbar axes
        self.cax_E = self.fig.add_axes([0.05, 0.25, 0.015, 0.3])
        self.cax_H = self.fig.add_axes([0.78, 0.25, 0.015, 0.3])
        self.cmap = plt.cm.plasma
        
        self.sm_E = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=self.global_max_E))
        self.cbar_E = self.fig.colorbar(self.sm_E, cax=self.cax_E, label='|E| [V/m]')
        self.cax_E.yaxis.set_ticks_position('left')
        self.cax_E.yaxis.set_label_position('left')
        
        self.sm_H = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=self.global_max_H))
        self.cbar_H = self.fig.colorbar(self.sm_H, cax=self.cax_H, label='|H| [A/m]')
        
        # Set aspect and view
        for ax in [self.ax_E, self.ax_H]:
            try:
                ax.set_box_aspect((1, 1, self.wg.L/(2*self.wg.a)))
            except AttributeError:
                pass
            ax.view_init(elev=20, azim=45)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            
        # --- 視点の同期機能 ---
        def sync_views(event):
            if event.inaxes not in [self.ax_E, self.ax_H]:
                return
            src_ax = event.inaxes
            dst_ax = self.ax_H if src_ax == self.ax_E else self.ax_E
            needs_sync = (dst_ax.elev != src_ax.elev) or (dst_ax.azim != src_ax.azim)
            has_roll = hasattr(src_ax, 'roll')
            if has_roll and hasattr(dst_ax, 'roll'):
                needs_sync = needs_sync or (dst_ax.roll != src_ax.roll)
            if needs_sync:
                if has_roll:
                    dst_ax.view_init(elev=src_ax.elev, azim=src_ax.azim, roll=src_ax.roll)
                else:
                    dst_ax.view_init(elev=src_ax.elev, azim=src_ax.azim)
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('motion_notify_event', sync_views)
        self.fig.canvas.mpl_connect('button_release_event', sync_views)
        
        self.update(None)

    def _draw_cylinder(self, ax, title):
        z_cyl = np.linspace(0, self.wg.L, 30)
        theta_cyl = np.linspace(0, 2*np.pi, 30)
        Z_cyl, Theta_cyl = np.meshgrid(z_cyl, theta_cyl)
        X_cyl = self.wg.a * np.cos(Theta_cyl)
        Y_cyl = self.wg.a * np.sin(Theta_cyl)
        
        ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.05, color='cyan', edgecolor='none')
        ax.plot_wireframe(X_cyl, Y_cyl, Z_cyl, alpha=0.1, color='gray', rcount=5, ccount=10)
        ax.set_title(title)

    def update(self, val):
        display_mode = self.mode_radio.value_selected
        style = self.style_radio.value_selected
        phi_val = np.deg2rad(self.s_phi.val)
        z_val = self.s_z.val
        t_val = self.s_t.val * (1.0 / self.wg.f)
        
        # Remove old quivers
        if self.q_E is not None: self.q_E.remove(); self.q_E = None
        if self.q_H is not None: self.q_H.remove(); self.q_H = None
            
        N_r = self.grid_params['Nr']
        N_p = self.grid_params['Np']
        N_z = self.grid_params['Nz']
        
        if display_mode == 'Full Volume':
            r_1d = np.linspace(self.wg.a/(N_r*2), self.wg.a*0.95, N_r)
            phi_1d = np.linspace(0, 2*np.pi, N_p, endpoint=False)
            z_1d = np.linspace(0, self.wg.L, N_z)
            R, Phi, Z = np.meshgrid(r_1d, phi_1d, z_1d, indexing='ij')
        else: # Cross Sections
            # r-z plane: 直径を通る
            r_1d_rz = np.concatenate((-np.linspace(self.wg.a*0.95, self.wg.a/(N_r*2), N_r), 
                                      np.linspace(self.wg.a/(N_r*2), self.wg.a*0.95, N_r)))
            z_1d_rz = np.linspace(0, self.wg.L, N_z)
            R_mesh_rz, Z_rz = np.meshgrid(r_1d_rz, z_1d_rz, indexing='ij')
            R_rz = np.abs(R_mesh_rz)
            Phi_rz = np.where(R_mesh_rz >= 0, phi_val, phi_val + np.pi)
            
            # x-y plane: 円形
            r_1d_xy = np.linspace(self.wg.a/(N_r*2), self.wg.a*0.95, N_r)
            phi_1d_xy = np.linspace(0, 2*np.pi, N_p, endpoint=False)
            R_xy, Phi_xy = np.meshgrid(r_1d_xy, phi_1d_xy, indexing='ij')
            Z_xy = np.full_like(R_xy, z_val)
            
            R = np.concatenate((R_rz.flatten(), R_xy.flatten()))
            Phi = np.concatenate((Phi_rz.flatten(), Phi_xy.flatten()))
            Z = np.concatenate((Z_rz.flatten(), Z_xy.flatten()))
            
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        
        # Get fields
        Er, Ep, Ez, Hr, Hp, Hz = self.wg.get_fields(R, Phi, Z, t_val)
        
        # Helper to convert and plot
        def get_cartesian_vectors(Fr, Fp, Fz, P):
            Fr_re, Fp_re, Fz_re = np.real(Fr), np.real(Fp), np.real(Fz)
            Fx = Fr_re * np.cos(P) - Fp_re * np.sin(P)
            Fy = Fr_re * np.sin(P) + Fp_re * np.cos(P)
            return Fx, Fy, Fz_re

        Ex, Ey, Ez_re = get_cartesian_vectors(Er, Ep, Ez, Phi)
        Hx, Hy, Hz_re = get_cartesian_vectors(Hr, Hp, Hz, Phi)
        
        Emag = np.sqrt(Ex**2 + Ey**2 + Ez_re**2)
        Hmag = np.sqrt(Hx**2 + Hy**2 + Hz_re**2)
        
        # Plot E-field
        mask_E = Emag > self.global_max_E * 0.05
        if np.any(mask_E):
            cE = self.cmap(self.sm_E.norm(Emag[mask_E]))
            lE = self.wg.a * 0.25 if style == 'Uniform Length' else (self.wg.a * 0.4) / self.global_max_E
            self.q_E = self.ax_E.quiver(X[mask_E], Y[mask_E], Z[mask_E], Ex[mask_E], Ey[mask_E], Ez_re[mask_E],
                                       length=lE, normalize=(style == 'Uniform Length'), colors=cE, lw=1.5)
        
        # Plot H-field
        mask_H = Hmag > self.global_max_H * 0.05
        if np.any(mask_H):
            cH = self.cmap(self.sm_H.norm(Hmag[mask_H]))
            lH = self.wg.a * 0.25 if style == 'Uniform Length' else (self.wg.a * 0.4) / self.global_max_H
            self.q_H = self.ax_H.quiver(X[mask_H], Y[mask_H], Z[mask_H], Hx[mask_H], Hy[mask_H], Hz_re[mask_H],
                                       length=lH, normalize=(style == 'Uniform Length'), colors=cH, lw=1.5)
        
        # タイトルの更新
        title_base = f'{self.wg.mode_type}{self.wg.n}{self.wg.m} Mode (f={self.wg.f/1e9:.3f}GHz)\n{display_mode}, {style}'
        self.ax_E.set_title(f'Electric Field (E)\n{title_base}')
        self.ax_H.set_title(f'Magnetic Field (H)\n{title_base}')
                                    
        self.fig.canvas.draw_idle()

def show_startup_dialog():
    root = tk.Tk()
    root.title("Waveguide Setup")
    root.geometry("400x480")
    
    # Variables
    a_var = tk.StringVar(value="0.05")
    L_var = tk.StringVar(value="0.1")
    dtheta_pi_var = tk.StringVar(value="0.5") 
    mode_var = tk.StringVar(value="TE")
    n_var = tk.StringVar(value="1")
    m_var = tk.StringVar(value="1")
    
    # Grid variables
    nr_var = tk.StringVar(value="5")
    np_var = tk.StringVar(value="12")
    nz_var = tk.StringVar(value="10")

    # Layout
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Section 1: Geometry
    ttk.Label(main_frame, text="Waveguide Geometry", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    ttk.Label(main_frame, text="Radius a [m]:").grid(row=1, column=0, sticky=tk.W)
    ttk.Entry(main_frame, textvariable=a_var, width=15).grid(row=1, column=1, pady=2)
    ttk.Label(main_frame, text="Length L [m]:").grid(row=2, column=0, sticky=tk.W)
    ttk.Entry(main_frame, textvariable=L_var, width=15).grid(row=2, column=1, pady=2)
    ttk.Label(main_frame, text="Phase diff [pi * rad]:").grid(row=3, column=0, sticky=tk.W)
    ttk.Entry(main_frame, textvariable=dtheta_pi_var, width=15).grid(row=3, column=1, pady=2)
    
    ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
    
    # Section 2: Mode
    ttk.Label(main_frame, text="Mode Configuration", font=("", 10, "bold")).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    ttk.Label(main_frame, text="Type:").grid(row=6, column=0, sticky=tk.W)
    ttk.Combobox(main_frame, textvariable=mode_var, values=["TE", "TM"], width=12).grid(row=6, column=1, pady=2)
    ttk.Label(main_frame, text="n (Azimuthal):").grid(row=7, column=0, sticky=tk.W)
    ttk.Spinbox(main_frame, from_=0, to=10, textvariable=n_var, width=13).grid(row=7, column=1, pady=2)
    ttk.Label(main_frame, text="m (Radial):").grid(row=8, column=0, sticky=tk.W)
    ttk.Spinbox(main_frame, from_=1, to=10, textvariable=m_var, width=13).grid(row=8, column=1, pady=2)

    ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)

    # Section 3: Grid Density
    ttk.Label(main_frame, text="Visualization Density", font=("", 10, "bold")).grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
    ttk.Label(main_frame, text="Grid (Radial):").grid(row=11, column=0, sticky=tk.W)
    ttk.Spinbox(main_frame, from_=2, to=20, textvariable=nr_var, width=13).grid(row=11, column=1, pady=2)
    ttk.Label(main_frame, text="Grid (Azimuthal):").grid(row=12, column=0, sticky=tk.W)
    ttk.Spinbox(main_frame, from_=4, to=50, textvariable=np_var, width=13).grid(row=12, column=1, pady=2)
    ttk.Label(main_frame, text="Grid (Z-axis):").grid(row=13, column=0, sticky=tk.W)
    ttk.Spinbox(main_frame, from_=2, to=50, textvariable=nz_var, width=13).grid(row=13, column=1, pady=2)

    params = {}
    grid_params = {}
    def on_launch():
        try:
            params['a'] = float(a_var.get())
            params['L'] = float(L_var.get())
            params['dtheta'] = float(dtheta_pi_var.get()) * np.pi
            params['mode_type'] = mode_var.get()
            params['n'] = int(n_var.get())
            params['m'] = int(m_var.get())
            
            grid_params['Nr'] = int(nr_var.get())
            grid_params['Np'] = int(np_var.get())
            grid_params['Nz'] = int(nz_var.get())
            
            root.destroy()
        except Exception:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")

    ttk.Button(main_frame, text="Launch 3D Viewer", command=on_launch).grid(row=14, column=0, columnspan=2, pady=20)
    
    root.mainloop()
    return (params, grid_params) if params else (None, None)

if __name__ == '__main__':
    p, g = show_startup_dialog()
    if p:
        wg = CylindricalWaveguide(**p)
        viewer = InteractiveWaveguideViewer(wg, g)
        plt.show()
