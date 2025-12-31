import numpy as np
import os
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from data_utils import load_ratings

# ===============================
# Alternating Minimization (AM)
# with adaptive stopping + bias
# (numerically stable version)
# ===============================
class AlternatingMinimization:
    def __init__(
        self,
        rank=20,
        lambda_reg=0.1,
        lambda_bias=10.0,   # ⭐ bias 正则
        max_iter=50,
        tol=1e-3,
        random_state=42
    ):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.lambda_bias = lambda_bias
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.U = None
        self.V = None
        self.mu = 0.0
        self.bu = None
        self.bi = None

        self.train_errors = []
        self.n_iters_ = 0

    def fit(self, user_idx, item_idx, ratings, n_users, n_items):
        np.random.seed(self.random_state)

        # ===== 初始化 =====
        self.mu = np.mean(ratings)
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        self.U = np.random.randn(n_users, self.rank) * 0.01
        self.V = np.random.randn(n_items, self.rank) * 0.01

        user_items = {}
        item_users = {}
        for u, i, r in zip(user_idx, item_idx, ratings):
            user_items.setdefault(u, []).append((i, r))
            item_users.setdefault(i, []).append((u, r))

        prev_rmse = np.inf
        print(f"  Start AM (rank={self.rank}, lambda={self.lambda_reg})")

        for it in range(1, self.max_iter + 1):
            t0 = time.time()

            # ===== 更新 item bias =====
            for i, ur_list in item_users.items():
                users, r = zip(*ur_list)
                users = np.array(users)
                r = np.array(r)

                pred = self.mu + self.bu[users] + np.sum(self.U[users] * self.V[i], axis=1)
                self.bi[i] = np.sum(r - pred) / (len(r) + self.lambda_bias)

            # ===== 更新 user bias =====
            for u, ir_list in user_items.items():
                items, r = zip(*ir_list)
                items = np.array(items)
                r = np.array(r)

                pred = self.mu + self.bi[items] + np.sum(self.U[u] * self.V[items], axis=1)
                self.bu[u] = np.sum(r - pred) / (len(r) + self.lambda_bias)

            # ===== 更新 V =====
            for i, ur_list in item_users.items():
                users, r = zip(*ur_list)
                users = np.array(users)
                r = np.array(r)

                r_hat = r - self.mu - self.bu[users] - self.bi[i]
                U_i = self.U[users]

                A = U_i.T @ U_i + self.lambda_reg * np.eye(self.rank)
                b = U_i.T @ r_hat
                self.V[i] = np.linalg.lstsq(A, b, rcond=None)[0]

            # ===== 更新 U =====
            for u, ir_list in user_items.items():
                items, r = zip(*ir_list)
                items = np.array(items)
                r = np.array(r)

                r_hat = r - self.mu - self.bu[u] - self.bi[items]
                V_u = self.V[items]

                A = V_u.T @ V_u + self.lambda_reg * np.eye(self.rank)
                b = V_u.T @ r_hat
                self.U[u] = np.linalg.lstsq(A, b, rcond=None)[0]

            # ===== 训练 RMSE =====
            preds = (
                self.mu
                + self.bu[user_idx]
                + self.bi[item_idx]
                + np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
            )

            rmse = np.sqrt(mean_squared_error(ratings, preds))
            self.train_errors.append(rmse)

            rel_change = abs(prev_rmse - rmse) / (prev_rmse + 1e-8)
            elapsed = time.time() - t0

            print(
                f"    iter {it:02d} | RMSE={rmse:.6f} "
                f"| rel_change={rel_change:.2e} | time={elapsed:.2f}s"
            )

            self.n_iters_ = it
            if rel_change < self.tol:
                print(f"    Converged (tol={self.tol}), stop at iter {it}")
                break

            prev_rmse = rmse

        return self

    def predict(self, user_idx, item_idx):
        return (
            self.mu
            + self.bu[user_idx]
            + self.bi[item_idx]
            + np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
        )


# ===============================
# Main (unchanged logic)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ratings = load_ratings(
    os.path.join(BASE_DIR, "ml-10M100K", "ratings.dat")
)

user_idx = ratings["user"].astype(int).values
item_idx = ratings["item"].astype(int).values
values = ratings["rating"].values

n_users = int(user_idx.max()) + 1
n_items = int(item_idx.max()) + 1

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_rmses = []
fold_times = []
fold_iters = []

for fold, (train_idx, test_idx) in enumerate(kf.split(ratings)):
    print(f"\n===== Fold {fold + 1} =====")

    train_users = user_idx[train_idx]
    train_items = item_idx[train_idx]
    train_vals = values[train_idx]

    test_users = user_idx[test_idx]
    test_items = item_idx[test_idx]
    test_vals = values[test_idx]

    model = AlternatingMinimization(rank=20, lambda_reg=0.1)

    t_start = time.time()
    model.fit(train_users, train_items, train_vals, n_users, n_items)
    total_time = time.time() - t_start

    test_preds = model.predict(test_users, test_items)
    rmse = np.sqrt(mean_squared_error(test_vals, test_preds))

    print(f"Fold {fold + 1} RMSE: {rmse:.4f} | iters={model.n_iters_} | time={total_time:.2f}s")

    fold_rmses.append(rmse)
    fold_times.append(total_time)
    fold_iters.append(model.n_iters_)

print("\n===== Summary =====")
print(f"AM (rank=20) 5-fold average RMSE: {np.mean(fold_rmses):.4f}")
print(f"Average training time per fold: {np.mean(fold_times):.2f}s")
print(f"Average iterations to converge: {np.mean(fold_iters):.1f}")
