import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_grad_distribution_cdf_simple_overlap_multiple_thresholds(
    grad_file_path: str,
    save_dir: str = "./grad_analysis_cdf",
    padding_ratio: float = 0.1
):
    """
    grad_file_path : train() 과정에서 저장된 npy 파일 경로 (예: grad_xyz_avg_per_timestep.npy)
    save_dir       : CDF 그래프 및 로그를 저장할 디렉터리
    padding_ratio  : CDF 플롯의 x축을 min, max에 맞출 때, 여유를 둘 비율

    - 상위 5%, 10%, 20% 총 3가지 임계값에 대해
      각 타임스텝별 인덱스 집합을 구하고,
      인접 타임스텝 간 중첩 개수/비율을 로그로 남깁니다.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_file = open(os.path.join(save_dir, "gradient_cdf_log.txt"), "w")

    # (1) grad_xyz_list 로드 (각 타임스텝별 (N,3) 형태 또는 None)
    grad_xyz_list = np.load(grad_file_path, allow_pickle=True)

    # 우리가 계산할 상위 k% 목록
    THRESHOLDS = [5, 10, 20]  # 상위 5%, 10%, 20%
    
    # 예: top_indices_dict[t][k] = t 타임스텝에서 "상위 k%"에 해당하는 인덱스 집합
    top_indices_dict = []

    # ---------- 타임스텝별 CDF + top k% 인덱스 추출 ----------
    for t, grad_t in enumerate(grad_xyz_list):
        if grad_t is None:
            msg = f"[Timestep {t}] No gradient data.\n"
            print(msg)
            log_file.write(msg)
            # 이 타임스텝에 대해서는 k%별로 빈 집합을 저장
            top_dict_for_t = {k: set() for k in THRESHOLDS}
            top_indices_dict.append(top_dict_for_t)
            continue

        # (N,3) -> (N,) L2-norm
        grad_norm = np.linalg.norm(grad_t, axis=1)

        g_min = grad_norm.min()
        g_max = grad_norm.max()
        g_mean = grad_norm.mean()
        g_std  = grad_norm.std()

        # CDF 그리기
        sorted_norm = np.sort(grad_norm)
        cdf_values = np.arange(len(sorted_norm)) / float(len(sorted_norm) - 1)

        plt.figure(figsize=(6,4))
        plt.plot(sorted_norm, cdf_values, color='blue', alpha=0.7)
        plt.title(f"Gradient L2-norm CDF (timestep {t})")
        plt.xlabel("Gradient L2-norm")
        plt.ylabel("CDF (P(grad <= x))")

        if g_min < g_max:
            span = g_max - g_min
            left  = g_min - padding_ratio * span
            right = g_max + padding_ratio * span
            plt.xlim(left, right)
        else:
            plt.xlim(g_min - 1e-5, g_max + 1e-5)

        plot_path = os.path.join(save_dir, f"cdf_t{t}.png")
        plt.savefig(plot_path)
        plt.close()

        # 로그 기록
        msg = (
            f"[Timestep {t}] #Gaussians: {grad_t.shape[0]}\n"
            f"   min : {g_min:.6f}\n"
            f"   max : {g_max:.6f}\n"
            f"   mean: {g_mean:.6f}\n"
            f"   std : {g_std:.6f}\n"
        )
        print(msg)
        log_file.write(msg)

        # ---------- 상위 k% 인덱스 뽑기 ----------
        top_dict_for_t = {}
        for k in THRESHOLDS:
            boundary = np.percentile(grad_norm, 100 - k)  # 예: k=10이면 하위 90분위
            mask     = (grad_norm >= boundary)            # 상위 k%
            top_inds = np.where(mask)[0]
            top_dict_for_t[k] = set(top_inds.tolist())

            msg_k = (
                f"   - threshold={k}% => boundary= {boundary:.6f}, "
                f"#(top{k}%)= {len(top_inds)}\n"
            )
            print(msg_k)
            log_file.write(msg_k)

        top_indices_dict.append(top_dict_for_t)
        # --------------------------------------

        log_file.write("\n")

    # (2) 인접 타임스텝 간 Overlap 계산
    msg = "\n[Overlap of top-k% indices across adjacent timesteps]\n"
    print(msg)
    log_file.write(msg)

    num_timesteps = len(grad_xyz_list)
    
    for t in range(num_timesteps - 1):
        set_t_dict  = top_indices_dict[t]
        set_t1_dict = top_indices_dict[t+1]
        
        msg_t = f"--- Timestep {t} vs {t+1} ---\n"
        print(msg_t)
        log_file.write(msg_t)

        for k in THRESHOLDS:
            set_t_k  = set_t_dict[k]   # 상위 k% (t시점)
            set_t1_k = set_t1_dict[k]  # 상위 k% (t+1 시점)

            overlap = set_t_k.intersection(set_t1_k)
            overlap_count = len(overlap)
            
            # 분모(여기서는 t+1의 상위 k% 개수로 사용)
            denom = len(set_t1_k) if len(set_t1_k) > 0 else 1
            ratio = overlap_count / denom

            msg_k = (
                f"  - top{k}% -> overlap count= {overlap_count}, "
                f"(overlap / top{k}%(t+1))= {ratio:.4f}\n"
            )
            print(msg_k)
            log_file.write(msg_k)
        log_file.write("\n")

    log_file.close()


if __name__ == "__main__":
    # 사용 예시
    grad_file_path = "./output/graidient-test/basketball/grad_xyz_avg_per_timestep.npy"
    analyze_grad_distribution_cdf_simple_overlap_multiple_thresholds(grad_file_path)
