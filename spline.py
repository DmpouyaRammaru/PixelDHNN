from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import csv

frame_numbers = []
angles = []

# CSVファイルから角度データを読み込み
with open("scaled_positions.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # ヘッダーをスキップ
    for row in reader:
        frame_numbers.append(int(row[0]))  # フレーム番号
        angles.append(-float(row[3]))  # 角度を逆転させて追加

frame_numbers = np.array(frame_numbers)
angles = np.array(angles)

# スムージングパラメータを指定 (小さくするほど荒くなります)
smoothing_factor =  500 # この値を調整して荒さを変更

# スムージングスプライン
spline = UnivariateSpline(frame_numbers, angles, s=smoothing_factor)

# 補間するフレームの範囲を作成
new_frame_numbers = np.linspace(frame_numbers.min(), frame_numbers.max(), 500)

# スプライン補間で角度を算出
new_angles = spline(new_frame_numbers)

# 結果をプロット（ラインの太さを変更）
plt.plot(new_frame_numbers, new_angles, label="Smoothing Spline", color="b", linewidth=3)  # 太さを2に設定
plt.scatter(frame_numbers, angles, color='r', label="Original Data", zorder=5,s=10)
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.show()

# スムージングされたデータを使用
time = new_frame_numbers  # 時間
angle = new_angles  # 角度

# ピークを検出
peaks, _ = find_peaks(angle)

# ピーク値と時間を抽出
peak_values = angle[peaks]
peak_times = time[peaks]

# ピーク値の減衰比を計算
if len(peak_values) > 1:
    R = peak_values[:-1] / peak_values[1:]  # ピーク間の比率
    zeta = 1 / np.sqrt(4 * np.pi**2 + (np.log(R))**2)  # 減衰比の計算
    average_zeta = np.mean(zeta)  # 減衰比の平均
    print(f"平均減衰比: {average_zeta}")
else:
    print("ピークが不足しているため、減衰比を計算できません。")

# ピーク値をプロット
plt.plot(time, angle, label="Smoothed Data", color="b")
plt.scatter(peak_times, peak_values, color='r', label="Peaks", zorder=5)
plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.title("Damping Analysis")
plt.show()


# スムージングデータから時間微分 (速度) を計算
dt = np.mean(np.diff(new_frame_numbers))  # 時間間隔
new_velocity = np.gradient(new_angles, dt)  # 角速度 (p)

# ハミルトニアンモデル
def hamiltonian_model(q, p, k):
    return k * (1 - np.cos(q)) + p**2

# 観測されたハミルトニアン値 (仮定としてエネルギー保存を使用)
observed_H = hamiltonian_model(new_angles, new_velocity, k=1)  # 初期値で計算

# 最小二乗法で k をフィッティング
def fit_function(q, k):
    return hamiltonian_model(q, new_velocity, k)

popt, pcov = curve_fit(fit_function, new_angles, observed_H)
k_fitted = popt[0]  # フィッティングされた k
print(f"フィッティングされた k: {k_fitted}")

# フィッティング結果をプロット
fitted_H = hamiltonian_model(new_angles, new_velocity, k_fitted)
plt.plot(new_frame_numbers, observed_H, label="Observed H", color="r")
plt.plot(new_frame_numbers, fitted_H, label=f"Fitted H (k={k_fitted:.2f})", color="b", linestyle="--")
plt.xlabel("Frame")
plt.ylabel("Hamiltonian (H)")
plt.legend()
plt.title("Hamiltonian Fit")
plt.show()