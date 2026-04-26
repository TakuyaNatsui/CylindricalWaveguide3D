import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CylindricalWaveguide:
    def __init__(self, mode_type, n, m, a, L, dtheta, eps_r=1.0, mu_r=1.0):
        """
        円筒導波管のパラメータ初期化
        mode_type: 'TE' or 'TM'
        n, m: モード次数 (m >= 1)
        a: 半径 [m]
        L: 長さ [m]
        dtheta: 両端の位相差 [rad]
        """
        self.mode_type = mode_type.upper()
        self.n = n
        self.m = m
        self.a = a
        self.L = L
        self.dtheta = dtheta
        
        # 物理定数
        self.c0 = 299792458.0
        self.mu0 = 4 * np.pi * 1e-7
        self.eps0 = 1 / (self.mu0 * self.c0**2)
        
        self.eps = eps_r * self.eps0
        self.mu = mu_r * self.mu0
        self.v0 = 1 / np.sqrt(self.eps * self.mu)
        
        # 基本パラメータの計算
        self._calc_parameters()
        
    def _calc_parameters(self):
        # 伝搬定数 beta
        self.beta = self.dtheta / self.L
        
        # 遮断波数 kc
        if self.mode_type == 'TM':
            # J_n(x) のm番目の零点
            zeros = sp.jn_zeros(self.n, self.m)
            self.p_nm = zeros[-1]
        elif self.mode_type == 'TE':
            # J'_n(x) のm番目の零点
            zeros = sp.jnp_zeros(self.n, self.m)
            self.p_nm = zeros[-1]
        else:
            raise ValueError("mode_type must be 'TE' or 'TM'")
            
        self.kc = self.p_nm / self.a
        
        # 動作角周波数 omega と 周波数 f
        self.omega = np.sqrt(self.kc**2 + self.beta**2) / np.sqrt(self.eps * self.mu)
        self.f = self.omega / (2 * np.pi)
        
        # 遮断周波数 fc
        self.fc = self.kc / (2 * np.pi * np.sqrt(self.eps * self.mu))
        
        # 群速度 vg
        if self.f > self.fc:
            self.vg = self.v0 * np.sqrt(1 - (self.fc / self.f)**2)
        else:
            self.vg = 0.0 # エバネッセント領域
            
    def get_fields(self, r, phi, z, t=0.0):
        """
        指定された円柱座標 (r, phi, z) における時刻 t の電磁界（複素フェーザに exp(j omega t) を掛けたもの）を返す。
        戻り値: (Er, Ephi, Ez, Hr, Hphi, Hz)
        """
        # r=0でのゼロ割り算を防ぐための微小値
        r_safe = np.where(r == 0, 1e-15, r)
        
        phase = np.exp(1j * (self.omega * t - self.beta * z))
        
        # 振幅定数（可視化のために適当に1とする）
        A = 1.0
        B = 1.0
        
        if self.mode_type == 'TM':
            Ez = A * sp.jv(self.n, self.kc * r) * np.cos(self.n * phi) * phase
            Er = (-1j * self.beta / self.kc) * A * sp.jvp(self.n, self.kc * r) * np.cos(self.n * phi) * phase
            Ephi = (1j * self.n * self.beta / (self.kc**2 * r_safe)) * A * sp.jv(self.n, self.kc * r) * np.sin(self.n * phi) * phase
            
            Hz = np.zeros_like(Ez)
            Hr = (-1j * self.n * self.omega * self.eps / (self.kc**2 * r_safe)) * A * sp.jv(self.n, self.kc * r) * np.sin(self.n * phi) * phase
            Hphi = (-1j * self.omega * self.eps / self.kc) * A * sp.jvp(self.n, self.kc * r) * np.cos(self.n * phi) * phase
            
            # 境界条件(r=a)付近でEzが厳密に0にならない数値誤差を丸める
            Ez = np.where(r > self.a, 0, Ez)
            
        else: # TE mode
            Hz = B * sp.jv(self.n, self.kc * r) * np.cos(self.n * phi) * phase
            Hr = (-1j * self.beta / self.kc) * B * sp.jvp(self.n, self.kc * r) * np.cos(self.n * phi) * phase
            Hphi = (1j * self.n * self.beta / (self.kc**2 * r_safe)) * B * sp.jv(self.n, self.kc * r) * np.sin(self.n * phi) * phase
            
            Ez = np.zeros_like(Hz)
            Er = (1j * self.n * self.omega * self.mu / (self.kc**2 * r_safe)) * B * sp.jv(self.n, self.kc * r) * np.sin(self.n * phi) * phase
            Ephi = (1j * self.omega * self.mu / self.kc) * B * sp.jvp(self.n, self.kc * r) * np.cos(self.n * phi) * phase
            
        # r=0での正確な極限の処理 (解析的に必要な場合)
        if self.n > 1:
            Er = np.where(r == 0, 0, Er)
            Ephi = np.where(r == 0, 0, Ephi)
            Hr = np.where(r == 0, 0, Hr)
            Hphi = np.where(r == 0, 0, Hphi)
            
        return Er, Ephi, Ez, Hr, Hphi, Hz

    def validate_group_velocity(self, N_r=200, N_phi=360):
        """
        断面でのポインティングベクトルとエネルギー密度を面積分し、群速度の理論値と一致するか検証する。
        """
        r_1d = np.linspace(0, self.a, N_r)
        phi_1d = np.linspace(0, 2*np.pi, N_phi)
        r, phi = np.meshgrid(r_1d, phi_1d)
        dr = self.a / (N_r - 1)
        dphi = 2 * np.pi / (N_phi - 1)
        dA = r * dr * dphi
        
        Er, Ephi, Ez, Hr, Hphi, Hz = self.get_fields(r, phi, 0, 0)
        
        # 時間平均ポインティングベクトル Z成分
        Sz = 0.5 * np.real(Er * np.conj(Hphi) - Ephi * np.conj(Hr))
        
        # 電磁エネルギー密度
        U_e = 0.25 * self.eps * (np.abs(Er)**2 + np.abs(Ephi)**2 + np.abs(Ez)**2)
        U_m = 0.25 * self.mu * (np.abs(Hr)**2 + np.abs(Hphi)**2 + np.abs(Hz)**2)
        U = U_e + U_m
        
        # 面積分
        Pz = np.sum(Sz * dA)
        W = np.sum(U * dA)
        
        vE = Pz / W
        error = np.abs(vE - self.vg) / self.vg * 100 if self.vg > 0 else 0
        
        print(f"--- 群速度の検証 ({self.mode_type}{self.n}{self.m} モード) ---")
        print(f"解析解による群速度 v_g : {self.vg:.6e} [m/s]")
        print(f"ポインティング定理 v_E: {vE:.6e} [m/s]")
        print(f"相対誤差: {error:.4e} %")
        return vE, self.vg

    def _get_zr_plot_data(self, z_1d, r_1d, t):
        """
        z-r平面 (r >= 0) におけるプロット用データを取得する。
        ベクトル成分とカラーマップ成分それぞれが最大となる phi を用いる。
        """
        Z, R = np.meshgrid(z_1d, r_1d)
        
        # モードと成分に応じた最適な phi の決定
        phi_cos = 0.0
        phi_sin = np.pi / (2 * self.n) if self.n > 0 else 0.0
        
        if self.mode_type == 'TE':
            self.phi_E_vec = phi_sin
            self.phi_E_col = phi_cos
            self.phi_H_vec = phi_cos
            self.phi_H_col = phi_sin
        else: # TM
            self.phi_E_vec = phi_cos
            self.phi_E_col = phi_sin
            self.phi_H_vec = phi_sin
            self.phi_H_col = phi_cos
            
        # E_vec: (Ez, Er) は phi_E_vec で計算
        Er_vec, _, Ez_vec, _, _, _ = self.get_fields(R, self.phi_E_vec, Z, t)
        # E_col: Ephi は phi_E_col で計算
        _, Ephi_col, _, _, _, _ = self.get_fields(R, self.phi_E_col, Z, t)
        
        # H_vec: (Hz, Hr) は phi_H_vec で計算
        _, _, _, Hr_vec, _, Hz_vec = self.get_fields(R, self.phi_H_vec, Z, t)
        # H_col: Hphi は phi_H_col で計算
        _, _, _, _, Hphi_col, _ = self.get_fields(R, self.phi_H_col, Z, t)
        
        # 実部をとる
        Er_re = np.real(Er_vec)
        Ephi_re = np.real(Ephi_col)
        Ez_re = np.real(Ez_vec)
        Hr_re = np.real(Hr_vec)
        Hphi_re = np.real(Hphi_col)
        Hz_re = np.real(Hz_vec)
        
        return Z, R, Er_re, Ephi_re, Ez_re, Hr_re, Hphi_re, Hz_re

    def plot_zr_plane(self, t=0.0, N_z=50, N_r=30, save_path=None):
        """
        z-r平面の電場・磁場を描画する。
        ベクトル: z-r方向成分, カラーマップ: phi方向成分
        """
        z_1d = np.linspace(0, self.L, N_z)
        r_1d = np.linspace(0, self.a, N_r)
        Z, R, Er, Ephi, Ez, Hr, Hphi, Hz = self._get_zr_plot_data(z_1d, r_1d, t)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        def format_phi(phi):
            if phi == 0: return "0"
            return f"{phi/np.pi:.2f}π"
        
        # 電場プロット
        vmax_E = np.max(np.abs(Ephi)) if np.max(np.abs(Ephi)) > 0 else 1.0
        cm1 = ax1.pcolormesh(Z, R, Ephi, shading='gouraud', cmap='RdBu_r', vmin=-vmax_E, vmax=vmax_E)
        if np.max(np.abs(Ez)) > 0 or np.max(np.abs(Er)) > 0:
            ax1.quiver(Z, R, Ez, Er, color='k', scale_units='xy', angles='xy')
        ax1.set_ylabel('r [m]')
        ax1.set_title(f'Electric Field (t={t:.2e}s)\nVector(Ez,Er) at $\phi$={format_phi(self.phi_E_vec)}, Color(Ephi) at $\phi$={format_phi(self.phi_E_col)}')
        fig.colorbar(cm1, ax=ax1, label='Ephi [V/m]')
        
        # 磁場プロット
        vmax_H = np.max(np.abs(Hphi)) if np.max(np.abs(Hphi)) > 0 else 1.0
        cm2 = ax2.pcolormesh(Z, R, Hphi, shading='gouraud', cmap='RdBu_r', vmin=-vmax_H, vmax=vmax_H)
        if np.max(np.abs(Hz)) > 0 or np.max(np.abs(Hr)) > 0:
            ax2.quiver(Z, R, Hz, Hr, color='k', scale_units='xy', angles='xy')
        ax2.set_xlabel('z [m]')
        ax2.set_ylabel('r [m]')
        ax2.set_title(f'Magnetic Field\nVector(Hz,Hr) at $\phi$={format_phi(self.phi_H_vec)}, Color(Hphi) at $\phi$={format_phi(self.phi_H_col)}')
        fig.colorbar(cm2, ax=ax2, label='Hphi [A/m]')
        
        # 図全体のタイトルに周波数と群速度を追加
        fig.suptitle(f"{self.mode_type}{self.n}{self.m} Mode: f = {self.f/1e9:.3f} GHz, vg/c = {self.vg/self.c0:.4f}", fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        plt.close()

    def animate_zr_plane(self, filename='waveguide_anim.gif', frames=40, N_z=50, N_r=30):
        """
        時間変化による進行波のアニメーションを生成する。
        """
        z_1d = np.linspace(0, self.L, N_z)
        r_1d = np.linspace(0, self.a, N_r)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        def format_phi(phi):
            if phi == 0: return "0"
            return f"{phi/np.pi:.2f}π"
            
        # 振幅スケール決定のためにt=0のデータを取得
        _, _, Er0, Ephi0, Ez0, Hr0, Hphi0, Hz0 = self._get_zr_plot_data(z_1d, r_1d, 0)
        vmax_E = np.max(np.abs(Ephi0)) if np.max(np.abs(Ephi0)) > 0 else 1.0
        vmax_H = np.max(np.abs(Hphi0)) if np.max(np.abs(Hphi0)) > 0 else 1.0
        
        def update(frame):
            # 1周期 (T = 1/f) 分をアニメーションする
            t = frame / frames * (1.0 / self.f)
            Z, R, Er, Ephi, Ez, Hr, Hphi, Hz = self._get_zr_plot_data(z_1d, r_1d, t)
            
            ax1.clear()
            ax2.clear()
            
            # 電場
            cm1 = ax1.pcolormesh(Z, R, Ephi, shading='gouraud', cmap='RdBu_r', vmin=-vmax_E, vmax=vmax_E)
            if np.max(np.abs(Ez)) > 0 or np.max(np.abs(Er)) > 0:
                ax1.quiver(Z, R, Ez, Er, color='k')
            ax1.set_ylabel('r [m]')
            ax1.set_title(f'Electric Field (t={t:.3e}s)\nVector(Ez,Er) at $\phi$={format_phi(self.phi_E_vec)}, Color(Ephi) at $\phi$={format_phi(self.phi_E_col)}')
            
            # 磁場
            cm2 = ax2.pcolormesh(Z, R, Hphi, shading='gouraud', cmap='RdBu_r', vmin=-vmax_H, vmax=vmax_H)
            if np.max(np.abs(Hz)) > 0 or np.max(np.abs(Hr)) > 0:
                ax2.quiver(Z, R, Hz, Hr, color='k')
            ax2.set_xlabel('z [m]')
            ax2.set_ylabel('r [m]')
            ax2.set_title(f'Magnetic Field\nVector(Hz,Hr) at $\phi$={format_phi(self.phi_H_vec)}, Color(Hphi) at $\phi$={format_phi(self.phi_H_col)}')
            
            fig.suptitle(f"{self.mode_type}{self.n}{self.m} Mode: f = {self.f/1e9:.3f} GHz, vg/c = {self.vg/self.c0:.4f}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            return ax1, ax2
            
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(filename, writer='pillow', fps=10)
        print(f"Saved animation to {filename}")
        plt.close()

    def save_data_to_txt(self, filename, t=0.0, N_z=5, N_r=5):
        """
        検証用に、特定の座標点における電磁場データをテキストファイルに書き出す。
        各成分は、グラフ表示と同様にそれぞれの最大振幅を与える phi で評価される。
        """
        z_samples = np.linspace(0, self.L, N_z)
        r_samples = np.linspace(0, self.a, N_r)
        
        # データの取得
        Z, R, Er, Ephi, Ez, Hr, Hphi, Hz = self._get_zr_plot_data(z_samples, r_samples, t)
        
        def format_phi(phi):
            if phi == 0: return "0"
            return f"{phi/np.pi:.2f}*pi"

        with open(filename, 'w', encoding='utf-8') as f:
            # ヘッダー情報の書き出し
            f.write(f"# Cylindrical Waveguide Mode Analysis Data\n")
            f.write(f"# Mode: {self.mode_type}{self.n}{self.m}\n")
            f.write(f"# Parameters: a={self.a}m, L={self.L}m, dtheta={self.dtheta}rad\n")
            f.write(f"# Results: f={self.f/1e9:.6f} GHz, fc={self.fc/1e9:.6f} GHz\n")
            f.write(f"# Group Velocity: vg={self.vg:.6e} m/s, vg/c={self.vg/self.c0:.6f}\n")
            f.write(f"# Time t: {t:.6e} s\n")
            f.write(f"# Phi angles used for evaluation:\n")
            f.write(f"#   E_vec (Ez, Er) at phi = {format_phi(self.phi_E_vec)}\n")
            f.write(f"#   E_col (Ephi)   at phi = {format_phi(self.phi_E_col)}\n")
            f.write(f"#   H_vec (Hz, Hr) at phi = {format_phi(self.phi_H_vec)}\n")
            f.write(f"#   H_col (Hphi)   at phi = {format_phi(self.phi_H_col)}\n")
            f.write(f"# ------------------------------------------------------------\n")
            f.write(f"# Columns:\n")
            f.write(f"# z[m], r[m], Ez[V/m], Er[V/m], Ephi[V/m], Hz[A/m], Hr[A/m], Hphi[A/m]\n")
            
            # 各点のデータの書き出し
            for i in range(N_r):
                for j in range(N_z):
                    line = f"{Z[i,j]:.6f}, {R[i,j]:.6f}, {Ez[i,j]:.6e}, {Er[i,j]:.6e}, {Ephi[i,j]:.6e}, {Hz[i,j]:.6e}, {Hr[i,j]:.6e}, {Hphi[i,j]:.6e}\n"
                    f.write(line)
        
        print(f"Saved numerical data to {filename}")

def run_solver():
    import os
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 計算・描画するモードのリスト (mode_type, n, m)
    modes = [
        ('TE', 1, 1),
        ('TM', 0, 1),
        ('TE', 2, 1),
        ('TM', 2, 1),
        ('TE', 0, 1),
        ('TM', 1, 1),
    ]

    for m_type, n, m in modes:
        mode_name = f"{m_type}{n}{m}"
        print(f"\n=== {mode_name} モードの計算・描画中 ===")
        
        # パラメータ設定 (半径0.05m, 長さ0.1m, 位相差pi)
        wg = CylindricalWaveguide(mode_type=m_type, n=n, m=m, a=0.05, L=0.5, dtheta=np.pi/2)
        
        # 群速度の検証
        wg.validate_group_velocity()
        
        # 静止画の保存
        plot_path = os.path.join(output_dir, f"{mode_name}_zr_plane.png")
        wg.plot_zr_plane(save_path=plot_path)
        
        # アニメーションの保存
        anim_path = os.path.join(output_dir, f"{mode_name}_anim.gif")
        wg.animate_zr_plane(filename=anim_path, frames=40)
        
        # 数値データの保存
        txt_path = os.path.join(output_dir, f"{mode_name}_data.txt")
        wg.save_data_to_txt(txt_path)

if __name__ == "__main__":
    run_solver()
