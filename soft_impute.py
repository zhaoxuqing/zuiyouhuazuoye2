import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import warnings
import time
import os
import gc
import json

warnings.filterwarnings('ignore')


class SoftImputeCV:

    def __init__(self, max_rank=30, lambda_values=None, max_iter=25,
                 convergence_thresh=3e-4, use_randomized_svd=True,
                 svd_iterations=7, random_state=42):
        """
        参数:
        - max_rank: 最大奇异值数量
        - lambda_values: 要测试的lambda值列表
        - max_iter: 最大迭代次数（增加到25次）
        - convergence_thresh: 收敛阈值（放宽到3e-4）
        - use_randomized_svd: 是否使用随机SVD
        - svd_iterations: 随机SVD的迭代次数
        - random_state: 随机种子
        """
        self.max_rank = max_rank
        self.lambda_values = lambda_values if lambda_values is not None else [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.max_iter = max_iter
        self.convergence_thresh = convergence_thresh
        self.use_randomized_svd = use_randomized_svd
        self.svd_iterations = svd_iterations
        self.random_state = random_state

        # 缓存
        self.R = None
        self.user_map = None
        self.movie_map = None
        self.global_mean = 0.0
        self.row_means = None
        self.col_means = None

        # 结果存储
        self.results = {}

        # 设置随机种子
        np.random.seed(random_state)

    # =========================================================================
    # 数据加载和初始化方法
    # =========================================================================

    def load_data_sparse(self, filepath='ratings.dat'):
        """加载数据为稀疏矩阵"""
        print("加载数据为稀疏矩阵...")

        cache_file = 'sparse_matrix_cache.npz'
        if os.path.exists(cache_file):
            print("加载缓存的稀疏矩阵...")
            self.R = sp.load_npz(cache_file)
            print(f"从缓存加载: {self.R.shape[0]}用户, {self.R.shape[1]}电影, {self.R.nnz}评分")
            return self.R, {}, {}

        # 第一次遍历：收集所有ID
        user_ids = set()
        movie_ids = set()
        ratings_data = []

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num % 1000000 == 0 and line_num > 0:
                    print(f"  已处理 {line_num} 行...")

                if '::' in line:
                    parts = line.strip().split('::')
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    rating = float(parts[2])

                    user_ids.add(user_id)
                    movie_ids.add(movie_id)
                    ratings_data.append((user_id, movie_id, rating))

        # 创建映射
        user_list = sorted(user_ids)
        movie_list = sorted(movie_ids)

        user_map = {uid: idx for idx, uid in enumerate(user_list)}
        movie_map = {mid: idx for idx, mid in enumerate(movie_list)}

        n_users = len(user_list)
        n_movies = len(movie_list)

        print(f"原始用户数: {max(user_ids)} -> 映射后: {n_users}")
        print(f"原始电影数: {max(movie_ids)} -> 映射后: {n_movies}")
        print(f"评分总数: {len(ratings_data)}")

        # 构建稀疏矩阵
        rows = np.zeros(len(ratings_data), dtype=np.int32)
        cols = np.zeros(len(ratings_data), dtype=np.int32)
        data = np.zeros(len(ratings_data), dtype=np.float32)

        for i, (user_id, movie_id, rating) in enumerate(ratings_data):
            rows[i] = user_map[user_id]
            cols[i] = movie_map[movie_id]
            data[i] = rating

        R_coo = sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_movies))
        self.R = R_coo.tocsr()

        # 统计信息
        density = self.R.nnz / (n_users * n_movies) * 100
        print(f"矩阵密度: {density:.4f}%")
        print(f"矩阵大小: {n_users}×{n_movies} = {n_users * n_movies:,} 元素")
        print(f"稀疏矩阵内存: {self.R.data.nbytes + self.R.indices.nbytes + self.R.indptr.nbytes:,} 字节")

        sp.save_npz(cache_file, self.R)
        print(f"稀疏矩阵已缓存到 {cache_file}")

        self.user_map = user_map
        self.movie_map = movie_map

        return self.R, user_map, movie_map

    def compute_matrix_stats(self, R):
        """计算矩阵的统计信息"""
        # 全局均值
        global_mean = np.mean(R.data) if R.nnz > 0 else 3.0
        global_std = np.std(R.data) if R.nnz > 0 else 1.0

        # 行均值
        row_sums = np.array(R.sum(axis=1)).flatten()
        row_counts = np.array((R != 0).sum(axis=1)).flatten()
        row_means = np.where(row_counts > 0, row_sums / row_counts, global_mean)

        # 列均值
        col_sums = np.array(R.sum(axis=0)).flatten()
        col_counts = np.array((R != 0).sum(axis=0)).flatten()
        col_means = np.where(col_counts > 0, col_sums / col_counts, global_mean)

        return global_mean, global_std, row_means, col_means

    def center_matrix(self, R):
        """中心化矩阵"""
        n_users, n_movies = R.shape

        # 计算统计信息
        global_mean, global_std, row_means, col_means = self.compute_matrix_stats(R)

        # 保存统计信息以便后续恢复
        self.global_mean = global_mean
        self.row_means = row_means
        self.col_means = col_means

        # 获取非零元素的位置
        rows, cols = R.nonzero()
        data = R.data.copy()

        # 中心化：移除行偏置和列偏置
        # 使用双重中心化：X_ij - row_mean_i - col_mean_j + global_mean
        centered_data = data - self.row_means[rows] - self.col_means[cols] + self.global_mean

        # 构建中心化后的稀疏矩阵
        R_centered = sp.csr_matrix((centered_data, (rows, cols)), shape=(n_users, n_movies))

        # 检查中心化后的统计信息
        if R_centered.nnz > 0:
            centered_mean = np.mean(R_centered.data)
            centered_std = np.std(R_centered.data)
            print(f"中心化后矩阵均值: {centered_mean:.6f} (目标:接近0)")
            print(f"中心化后矩阵标准差: {centered_std:.3f}")
            print(f"中心化后数据范围: [{R_centered.data.min():.3f}, {R_centered.data.max():.3f}]")

        return R_centered

    def decenter_matrix(self, X_centered):
        """恢复矩阵到原始尺度"""
        if self.row_means is None or self.col_means is None:
            return X_centered

        n_users, n_movies = X_centered.shape
        X = X_centered.copy()

        # 添加偏置
        for i in range(n_users):
            X[i, :] += self.row_means[i]

        for j in range(n_movies):
            X[:, j] += self.col_means[j]

        # 减去全局均值（因为加了两次）
        X -= self.global_mean

        # 确保在有效范围内
        X = np.clip(X, 1.0, 5.0)

        return X

    def analyze_singular_values(self, X, max_rank=50):
        """分析奇异值以确定lambda范围"""
        print("\n分析奇异值以确定合适的lambda范围...")

        # 计算奇异值
        if self.use_randomized_svd:
            try:
                _, s, _ = randomized_svd(
                    X.astype(np.float64),
                    n_components=max_rank,
                    n_iter=self.svd_iterations,
                    random_state=self.random_state
                )
                print("使用随机SVD计算奇异值")
            except:
                # 如果随机SVD失败，使用截断SVD
                _, s, _ = svds(X.astype(np.float64), k=max_rank, which='LM')
                print("使用scipy的svds计算奇异值")
        else:
            _, s, _ = svds(X.astype(np.float64), k=max_rank, which='LM')
            print("使用scipy的svds计算奇异值")

        # 按降序排序
        s = np.sort(s)[::-1]

        # 打印奇异值信息
        print(f"\n奇异值分析 (前{len(s)}个):")
        print(f"最大奇异值: {s[0]:.3f}")
        print(f"最小奇异值: {s[-1]:.3f}")
        print(f"奇异值总和: {np.sum(s):.3f}")
        print(f"前10个奇异值: {s[:10]}")

        # 计算累积能量
        cumulative_energy = np.cumsum(s) / np.sum(s)
        print(f"\n奇异值累积能量:")
        for i in [1, 5, 10, 20, 30, 50]:
            if i <= len(s):
                print(f"  前{i}个奇异值占总能量的: {cumulative_energy[i - 1]:.1%}")

        # 根据奇异值大小确定lambda范围
        max_singular_value = s[0]

        # lambda应该大约是最大奇异值的0.1倍到10倍
        # 但实际中，lambda通常比最大奇异值小
        lambda_suggestions = []

        # 如果奇异值非常大，我们需要很大的lambda
        if max_singular_value > 1000:
            print(f"\n⚠️  奇异值非常大 (最大奇异值: {max_singular_value:.1f})")
            print("这解释了为什么lambda需要很大的值")
            # 建议lambda范围：最大奇异值的0.01%到10%
            lambda_suggestions = [
                max_singular_value * 0.0001,
                max_singular_value * 0.001,
                max_singular_value * 0.01,
                max_singular_value * 0.05,
                max_singular_value * 0.1,
                max_singular_value * 0.5,
                max_singular_value * 1.0,
                max_singular_value * 2.0,
                max_singular_value * 5.0,
                max_singular_value * 10.0
            ]
        elif max_singular_value > 100:
            print(f"\n奇异值较大 (最大奇异值: {max_singular_value:.1f})")
            # 建议lambda范围：最大奇异值的0.1%到100%
            lambda_suggestions = [
                max_singular_value * 0.001,
                max_singular_value * 0.01,
                max_singular_value * 0.05,
                max_singular_value * 0.1,
                max_singular_value * 0.5,
                max_singular_value * 1.0,
                max_singular_value * 2.0,
                max_singular_value * 5.0,
                max_singular_value * 10.0,
                max_singular_value * 20.0
            ]
        else:
            print(f"\n奇异值正常范围 (最大奇异值: {max_singular_value:.1f})")
            # 建议lambda范围：最大奇异值的1%到1000%
            lambda_suggestions = [
                max_singular_value * 0.01,
                max_singular_value * 0.05,
                max_singular_value * 0.1,
                max_singular_value * 0.5,
                max_singular_value * 1.0,
                max_singular_value * 2.0,
                max_singular_value * 5.0,
                max_singular_value * 10.0,
                max_singular_value * 20.0,
                max_singular_value * 50.0
            ]

        # 确保lambda值都是正数且合理
        lambda_suggestions = [max(l, 0.001) for l in lambda_suggestions]
        lambda_suggestions = sorted(set(lambda_suggestions))

        print(f"\n建议的lambda范围: {[f'{l:.3f}' for l in lambda_suggestions]}")
        print(f"建议的lambda范围: {lambda_suggestions}")

        return lambda_suggestions, s

    def initialize_matrix(self, R_centered):
        """初始化中心化后的矩阵"""
        n_users, n_movies = R_centered.shape

        # 对于中心化后的矩阵，初始化为0矩阵，并在观测位置填充中心化后的值
        X = np.zeros((n_users, n_movies), dtype=np.float32)

        # 填充观测值
        rows, cols = R_centered.nonzero()
        X[rows, cols] = R_centered.data

        return X

    # =========================================================================
    # Soft-Impute核心算法
    # =========================================================================

    def soft_threshold_svd(self, X, lambda_, k):
        """对矩阵X进行软阈值SVD"""
        if self.use_randomized_svd and k > 0:
            U, s, Vt = randomized_svd(
                X.astype(np.float64),
                n_components=k,
                n_iter=self.svd_iterations,
                random_state=self.random_state
            )
        else:
            U, s, Vt = svds(X.astype(np.float64), k=k, which='LM')
            idx = np.argsort(-s)
            s = s[idx]
            U = U[:, idx]
            Vt = Vt[idx, :]

        # 软阈值处理
        s_thresh = np.maximum(s - lambda_, 0)

        # 重建矩阵
        mask = s_thresh > 0
        if np.sum(mask) > 0:
            Z = (U[:, mask] @ np.diag(s_thresh[mask]) @ Vt[mask, :]).astype(np.float32)
        else:
            Z = np.zeros_like(X, dtype=np.float32)

        return Z, s, s_thresh

    def soft_impute_iteration(self, R_train, lambda_):
        """Soft-Impute迭代（"""
        n_users, n_movies = R_train.shape

        # 1. 中心化训练集
        R_train_centered = self.center_matrix(R_train)

        # 获取观测位置
        obs_rows, obs_cols = R_train_centered.nonzero()
        obs_data = R_train_centered.data

        # 2. 初始化矩阵
        X = self.initialize_matrix(R_train_centered)

        # 3. 迭代Soft-Impute
        for i in range(self.max_iter):
            start_time = time.time()

            # 确定秩
            k = min(self.max_rank, min(X.shape) - 1)
            if k <= 0:
                k = 1

            try:
                # 执行SVD
                Z, s, s_thresh = self.soft_threshold_svd(X, lambda_, k)

                # 创建新矩阵
                X_new = Z.copy()
                X_new[obs_rows, obs_cols] = obs_data

                # 限制值范围
                X_new = np.clip(X_new, -10.0, 10.0)  # 中心化后的值范围可能更广

                # 检查收敛
                change = np.linalg.norm(X_new - X, 'fro') / (np.linalg.norm(X, 'fro') + 1e-10)
                X = X_new

                # 计算有效秩
                effective_rank = np.sum(s_thresh > 0)

                # 计算能量保留比例
                if len(s) > 0 and np.sum(s) > 0:
                    energy_retained = np.sum(s_thresh) / np.sum(s)
                else:
                    energy_retained = 0.0

                iter_time = time.time() - start_time

                # 每5次迭代或收敛时打印信息
                if (i + 1) % 5 == 0 or i == 0 or change < self.convergence_thresh:
                    print(f"    迭代 {i + 1:2d}/{self.max_iter}: {iter_time:5.1f}秒, "
                          f"变化: {change:.6f}, 有效秩: {effective_rank}, "
                          f"能量保留: {energy_retained:.1%}")

                # 收敛检查
                if change < self.convergence_thresh:
                    if (i + 1) % 5 != 0:  # 如果还没打印过
                        print(f"    迭代 {i + 1:2d}/{self.max_iter}: {iter_time:5.1f}秒, "
                              f"变化: {change:.6f}, 有效秩: {effective_rank}")
                    print(f"    迭代 {i + 1} 已收敛（变化量: {change:.6f} < {self.convergence_thresh}）")
                    break

                # 清理内存
                del Z, s, s_thresh
                gc.collect()

            except MemoryError:
                print(f"    内存不足，提前停止迭代")
                break
            except Exception as e:
                print(f"    迭代失败: {e}")
                break

        # 4. 恢复矩阵到原始尺度
        X_restored = self.decenter_matrix(X)

        return X_restored

    # =========================================================================
    # 交叉验证框架
    # =========================================================================

    def create_fold(self, R, fold_idx, n_folds=5):
        """创建交叉验证折"""
        fold_seed = self.random_state + fold_idx * 100
        np.random.seed(fold_seed)

        rows, cols = R.nonzero()
        data = R.data
        n_ratings = len(data)

        # 打乱索引
        indices = np.random.permutation(n_ratings)
        fold_size = n_ratings / n_folds

        # 测试集索引
        test_start = int(fold_idx * fold_size)
        test_end = int((fold_idx + 1) * fold_size) if fold_idx < n_folds - 1 else n_ratings
        test_idx = indices[test_start:test_end]

        # 训练集索引
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # 创建训练稀疏矩阵
        train_rows = rows[train_idx]
        train_cols = cols[train_idx]
        train_data = data[train_idx]
        R_train = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=R.shape)

        # 测试集信息
        test_info = {
            'rows': rows[test_idx],
            'cols': cols[test_idx],
            'true_ratings': data[test_idx]
        }

        return R_train, test_info

    def evaluate_lambda_fold(self, lambda_, fold_idx, R_train, test_info):
        """评估特定lambda在特定折上的性能"""
        print(f"  λ={lambda_:.3f} - 训练Soft-Impute...")

        # 训练模型
        X_pred = self.soft_impute_iteration(R_train, lambda_)

        # 预测测试集
        test_rows = test_info['rows']
        test_cols = test_info['cols']
        true_ratings = test_info['true_ratings']

        predicted = X_pred[test_rows, test_cols]

        # 计算评估指标
        rmse = np.sqrt(np.mean((predicted - true_ratings) ** 2))
        mae = np.mean(np.abs(predicted - true_ratings))

        # 裁剪预测值并重新计算
        predicted_clipped = np.clip(predicted, 1.0, 5.0)
        rmse_clipped = np.sqrt(np.mean((predicted_clipped - true_ratings) ** 2))

        print(f"  λ={lambda_:.3f} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, 裁剪后RMSE: {rmse_clipped:.6f}")

        # 清理内存
        del X_pred
        gc.collect()

        return rmse, mae, rmse_clipped

    def run_single_fold(self, fold_idx, analyze_svd=False):
        """运行单折交叉验证"""
        print(f"\n{'=' * 60}")
        print(f"第 {fold_idx + 1}/5 折")
        print(f"{'=' * 60}")

        # 如果是第一折，加载数据
        if fold_idx == 0 and self.R is None:
            self.R, self.user_map, self.movie_map = self.load_data_sparse()

        # 创建训练测试分割
        R_train, test_info = self.create_fold(self.R, fold_idx)

        print(f"训练集: {R_train.nnz} 评分")
        print(f"测试集: {len(test_info['true_ratings'])} 评分")

        # 如果需要分析奇异值来确定lambda范围
        if analyze_svd and fold_idx == 0:
            # 中心化训练集
            R_train_centered = self.center_matrix(R_train)
            X_init = self.initialize_matrix(R_train_centered)

            # 分析奇异值
            suggested_lambdas, singular_values = self.analyze_singular_values(X_init, max_rank=50)

            # 使用建议的lambda值
            self.lambda_values = suggested_lambdas
            print(f"根据奇异值分析，使用以下lambda值: {[f'{l:.3f}' for l in self.lambda_values]}")

            # 保存奇异值分析结果
            with open('singular_values_analysis.json', 'w') as f:
                json.dump({
                    'singular_values': singular_values.tolist(),
                    'suggested_lambdas': suggested_lambdas,
                    'max_singular_value': float(singular_values[0]),
                    'fold': fold_idx
                }, f, indent=2)
            print("奇异值分析结果已保存到 singular_values_analysis.json")

        fold_results = []

        # 测试所有lambda值
        for lambda_idx, lambda_ in enumerate(self.lambda_values):
            print(f"\n测试 λ={lambda_:.3f} ({lambda_idx + 1}/{len(self.lambda_values)})")

            fold_start_time = time.time()

            # 评估当前lambda
            rmse, mae, rmse_clipped = self.evaluate_lambda_fold(lambda_, fold_idx, R_train, test_info)

            fold_time = time.time() - fold_start_time

            # 存储结果
            result = {
                'lambda': float(lambda_),
                'fold': fold_idx,
                'rmse': float(rmse),
                'mae': float(mae),
                'rmse_clipped': float(rmse_clipped),
                'time': float(fold_time)
            }

            fold_results.append(result)

            # 初始化该lambda的结果列表
            if lambda_ not in self.results:
                self.results[lambda_] = []
            self.results[lambda_].append(result)

        # 清理内存
        del R_train
        gc.collect()

        return fold_results

    def run_cross_validation(self, analyze_svd=True):
        """运行五折交叉验证"""
        print("=" * 70)
        print("稀疏Soft-Impute五折交叉验证（参数调优版）")
        print("=" * 70)
        print(f"测试的λ值: {self.lambda_values}")
        print(f"最大迭代次数: {self.max_iter}")
        print(f"收敛阈值: {self.convergence_thresh}")
        print(f"使用随机SVD: {self.use_randomized_svd}")
        print(f"随机SVD迭代次数: {self.svd_iterations}")
        print("=" * 70)

        total_start_time = time.time()

        # 运行所有折
        for fold_idx in range(5):
            fold_results = self.run_single_fold(fold_idx, analyze_svd=analyze_svd and fold_idx == 0)

            # 保存每折的中间结果
            self.save_intermediate_results(fold_idx, fold_results)

        total_time = time.time() - total_start_time

        # 分析结果
        best_lambda, best_rmse = self.analyze_results(total_time)

        return best_lambda, best_rmse

    def analyze_results(self, total_time):
        """分析交叉验证结果"""
        print("\n" + "=" * 70)
        print("五折交叉验证结果汇总")
        print("=" * 70)

        summary = {}

        for lambda_ in self.lambda_values:
            if lambda_ in self.results and len(self.results[lambda_]) >= 3:
                lambda_results = self.results[lambda_]

                rmses = [r['rmse'] for r in lambda_results]
                rmses_clipped = [r['rmse_clipped'] for r in lambda_results]
                maes = [r['mae'] for r in lambda_results]
                times = [r['time'] for r in lambda_results]

                summary[lambda_] = {
                    'avg_rmse': np.mean(rmses),
                    'std_rmse': np.std(rmses),
                    'avg_rmse_clipped': np.mean(rmses_clipped),
                    'avg_mae': np.mean(maes),
                    'avg_time': np.mean(times),
                    'all_rmses': rmses,
                    'all_maes': maes
                }

        # 打印详细结果表格
        print("\n各λ值性能对比:")
        print("-" * 90)
        print(
            f"{'λ':>8} | {'平均RMSE':>10} | {'RMSE标准差':>10} | {'裁剪后RMSE':>10} | {'平均MAE':>10} | {'平均时间(秒)':>12}")
        print("-" * 90)

        best_lambda = None
        best_rmse = float('inf')

        for lambda_, stats in sorted(summary.items()):
            print(f"{lambda_:>8.3f} | {stats['avg_rmse']:>10.6f} | {stats['std_rmse']:>10.6f} | "
                  f"{stats['avg_rmse_clipped']:>10.6f} | {stats['avg_mae']:>10.6f} | {stats['avg_time']:>12.1f}")

            if stats['avg_rmse'] < best_rmse:
                best_rmse = stats['avg_rmse']
                best_lambda = lambda_

        print("-" * 90)

        if best_lambda is not None:
            print(f"\n🎯 最佳参数: λ = {best_lambda:.3f}")
            print(f"🎯 最佳平均RMSE: {best_rmse:.6f}")

            # 显示最佳λ的详细结果
            best_stats = summary[best_lambda]
            print(f"各折RMSE: {[f'{r:.6f}' for r in best_stats['all_rmses']]}")
            print(f"各折MAE: {[f'{m:.6f}' for m in best_stats['all_maes']]}")

        print(f"\n⏱️  总运行时间: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")

        # 保存详细结果
        self.save_final_results(summary, best_lambda, best_rmse, total_time)

        # 参数调优建议
        print("\n" + "=" * 70)
        print("参数调优建议:")
        print("=" * 70)

        if best_lambda is not None:
            # 分析λ的趋势
            lambdas = sorted(summary.keys())
            avg_rmses = [summary[l]['avg_rmse'] for l in lambdas]

            # 寻找局部最优
            min_idx = np.argmin(avg_rmses)

            print(f"1. 最佳λ值在 {best_lambda:.3f} 附近")

            # 检查是否需要扩大搜索范围
            if min_idx == 0:
                print(f"2. 建议测试更小的λ值（如 {best_lambda / 2:.3f}）")
            elif min_idx == len(lambdas) - 1:
                print(f"2. 建议测试更大的λ值（如 {best_lambda * 2:.3f}）")
            else:
                print(f"2. 当前搜索范围已包含最优值")

            # 建议精细搜索范围
            left_bound = lambdas[max(0, min_idx - 1)]
            right_bound = lambdas[min(len(lambdas) - 1, min_idx + 1)]
            print(f"3. 建议在 [{left_bound:.3f}, {right_bound:.3f}] 范围内进行精细搜索")

            # 根据标准差提供建议
            best_std = summary[best_lambda]['std_rmse']
            if best_std < 0.001:
                print(f"4. 结果稳定性很好（标准差: {best_std:.6f}）")
            elif best_std < 0.002:
                print(f"4. 结果稳定性较好（标准差: {best_std:.6f}）")
            else:
                print(f"4. 结果稳定性一般（标准差: {best_std:.6f}），可能需要更多数据")

        return best_lambda, best_rmse

    def save_intermediate_results(self, fold_idx, fold_results):
        """保存中间结果"""
        filename = f'cv_fold_{fold_idx + 1}_results.json'

        with open(filename, 'w') as f:
            json.dump({
                'fold': fold_idx,
                'parameters': {
                    'max_rank': self.max_rank,
                    'max_iter': self.max_iter,
                    'convergence_thresh': self.convergence_thresh,
                    'use_randomized_svd': self.use_randomized_svd,
                    'svd_iterations': self.svd_iterations
                },
                'results': fold_results
            }, f, indent=2)

        print(f"第 {fold_idx + 1} 折结果已保存到 {filename}")

    def save_final_results(self, summary, best_lambda, best_rmse, total_time):
        """保存最终结果"""
        # JSON格式
        json_filename = 'softimpute_cv_final_results.json'

        results_dict = {
            'parameters': {
                'max_rank': self.max_rank,
                'lambda_values': self.lambda_values,
                'max_iter': self.max_iter,
                'convergence_thresh': self.convergence_thresh,
                'use_randomized_svd': self.use_randomized_svd,
                'svd_iterations': self.svd_iterations,
                'random_state': self.random_state
            },
            'summary': {str(k): v for k, v in summary.items()},
            'best_parameters': {
                'lambda': float(best_lambda) if best_lambda is not None else None,
                'rmse': float(best_rmse) if best_rmse is not None else None
            },
            'total_time': total_time,
            'all_results': {str(k): v for k, v in self.results.items()}
        }

        with open(json_filename, 'w') as f:
            json.dump(results_dict, f, indent=2)

        # 文本格式
        txt_filename = 'softimpute_cv_final_results.txt'
        with open(txt_filename, 'w') as f:
            f.write("Soft-Impute五折交叉验证最终结果\n")
            f.write("=" * 70 + "\n\n")

            f.write("参数设置:\n")
            f.write(f"  最大秩: {self.max_rank}\n")
            f.write(f"  测试λ值: {self.lambda_values}\n")
            f.write(f"  最大迭代次数: {self.max_iter}\n")
            f.write(f"  收敛阈值: {self.convergence_thresh}\n")
            f.write(f"  使用随机SVD: {self.use_randomized_svd}\n")
            f.write(f"  随机SVD迭代次数: {self.svd_iterations}\n\n")

            f.write("各λ值性能汇总:\n")
            f.write("-" * 90 + "\n")
            f.write(
                f"{'λ':>8} | {'平均RMSE':>10} | {'RMSE标准差':>10} | {'裁剪后RMSE':>10} | {'平均MAE':>10} | {'平均时间(秒)':>12}\n")
            f.write("-" * 90 + "\n")

            for lambda_, stats in sorted(summary.items()):
                f.write(f"{lambda_:>8.3f} | {stats['avg_rmse']:>10.6f} | {stats['std_rmse']:>10.6f} | "
                        f"{stats['avg_rmse_clipped']:>10.6f} | {stats['avg_mae']:>10.6f} | {stats['avg_time']:>12.1f}\n")

            f.write("-" * 90 + "\n\n")

            if best_lambda is not None:
                f.write(f"最佳参数: λ = {best_lambda:.3f}\n")
                f.write(f"最佳平均RMSE: {best_rmse:.6f}\n\n")

                best_stats = summary[best_lambda]
                f.write(f"各折详细结果 (λ={best_lambda:.3f}):\n")
                for fold_idx, (rmse, mae) in enumerate(zip(best_stats['all_rmses'], best_stats['all_maes'])):
                    f.write(f"  第{fold_idx + 1}折: RMSE = {rmse:.6f}, MAE = {mae:.6f}\n")

            f.write(f"\n总运行时间: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)\n")

        print(f"\n📁 最终结果已保存到:")
        print(f"  📄 JSON格式: {json_filename}")
        print(f"  📄 文本格式: {txt_filename}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("稀疏Soft-Impute五折交叉验证（参数调优版）")
    print("=" * 70)

    # 检查文件是否存在
    if not os.path.exists('ratings.dat'):
        print("❌ 错误: 找不到 ratings.dat 文件")
        print("请将 ratings.dat 文件放在当前目录")
        exit(1)

    # 初始lambda值（根据奇异值分析后会被调整）
    lambda_values = [
        0.01,  # 非常小的正则化
        0.05,  # 较小的正则化
        0.1,  # 中等正则化
        0.5,  # 较强的正则化
        1.0,  # 强正则化
        2.0,  # 很强的正则化
        5.0  # 极强的正则化
    ]

    # 创建参数调优模型
    model = SoftImputeCV(
        max_rank=20,  # 奇异值数量（与之前保持一致）
        lambda_values=lambda_values,  # 初始lambda值，会根据奇异值分析调整
        max_iter=25,  # 增加到25次迭代
        convergence_thresh=3e-4,  # 放宽收敛阈值
        use_randomized_svd=True,  # 使用随机SVD加速
        svd_iterations=7,  # 随机SVD迭代次数（增加以提高精度）
        random_state=42  # 固定随机种子
    )

    try:
        print("🚀 开始参数调优交叉验证...")
        print(f"📊 将根据奇异值分析确定lambda范围")
        print(f"🔄 每个λ值将运行五折交叉验证")
        print(f"⏱️  预计总时间: 约3-4小时")
        print("=" * 70)

        # 运行交叉验证，并分析奇异值来确定lambda范围
        best_lambda, best_rmse = model.run_cross_validation(analyze_svd=True)

        print("\n" + "=" * 70)
        print("✅ 交叉验证完成!")
        print("=" * 70)

        if best_lambda is not None:
            print(f"\n🎯 最终推荐参数:")
            print(f"   λ = {best_lambda:.3f}")
            print(f"   RMSE = {best_rmse:.6f}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断，保存当前结果...")
        if hasattr(model, 'results') and model.results:
            with open('interrupted_results.json', 'w') as f:
                json.dump({str(k): v for k, v in model.results.items()}, f, indent=2)
            print("💾 中间结果已保存到 interrupted_results.json")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 程序结束")


if __name__ == "__main__":
    main()
