import numpy as np 
from scipy import integrate
from matplotlib import pyplot as plt

# Helper Functions
def random_mask(n, m):
    assert n >= m 
    mask = np.full(n, False, dtype=bool)
    true_indices = np.random.choice(n, size=m, replace=False)
    mask[true_indices] = True
    return mask

def evaluate(budgets, rs, rs_base):
    # rs: PTTS
    # rs_base: TTS
    assert len(budgets) == len(rs) == len(rs_base), "Lengths must match."
    assert (np.arange(len(budgets))==np.argsort(budgets)).all(), "Budgets must be sorted in ascending order."
    y0 = np.array(rs_base)[0]
    y_rs = np.array(rs) - y0
    y_rs_base = np.array(rs_base) - y0
    area1 = integrate.simpson(y=y_rs, x=budgets)
    area2 = integrate.simpson(y=y_rs_base, x=budgets)
    difference = area1 - area2
    increase = (area1 - area2) / area2 if area2 != 0 else float('inf')
    difference = np.round(difference, 2)
    increase = np.round(100*increase, 2)
    return difference.item(), increase.item()

# Plot the reward vs budget curves
def plot_curve(budgets, rewards_out1, rewards_out2):
    plt.plot(budgets, rewards_out1, label='PTTS', color='orange')
    plt.plot(budgets, rewards_out2, label='TTS', color='blue', linestyle='--')
    plt.xlabel('Budget (TFlops)')
    plt.ylabel('Reward')
    plt.title('Reward vs Budget Curve')
    plt.legend()
    plt.grid(linestyle='--', color='gray', alpha=0.7)
    plt.show()


# ---- Simulation Class -----
class Simulation():
    def __init__(self, x, y, max_step=20):
        self.x = x # the denoise budget (TFlops)
        self.y = y # the verifier budget (TFlops)
        self.max_step = max_step
    
    # ----- The TTS Best of N selection baseline -----
    def bon(self, r_gt):
        # r_gt: [num_prompts * num_images]
        return r_gt.max(axis=1).mean().item(), r_gt.argmax(axis=1) 

    def bon_cost(self):
        # of one image selection
        return self.max_step * self.x + self.y

    def bon_run(self, r_gt, image_num_use, iters=10):
        # r_gt: [num_prompts * num_images(all)] 
        out_rewards = []
        image_num_all = r_gt.shape[1]
        # run mutliple iters for averaging
        for i in range(iters):
            mask = random_mask(image_num_all, image_num_use)
            r_gt_select = r_gt[:, mask]
            out_reward, out_ids = self.bon(r_gt_select)
            out_rewards.append(out_reward)
        if iters == 1:
            return out_reward, out_ids
        else:
            return np.mean(out_rewards).item(), None
    

    # ----- The TTSp / TTSnap selection -----
    # steps use : list of steps to use (exclude the final step, which is gt, starting from 0)
    # alpha_s: list of alphas to use 
    # r: [num_prompts , num_images, num_steps]  

    def ttsp(self, r, alpha_s, steps_use):
        n_p, n_i, n_s = r.shape # num_prompts, num_images, num_steps
        # pruning at each timestepw
        mask = np.full((n_p, n_i), True, dtype=bool) # [n_p, n_i]
        image_ids = np.tile(np.arange(n_i), (n_p, 1)) # [n_p, n_i]
        for i, step in enumerate(steps_use): 
            r_mid = r[:, :, step][mask].reshape(n_p, -1) # [n_p, current_num_images]
            topn = max(1, round(r_mid.shape[1] * alpha_s[i]))
            
            # thresholds = np.partition(r_mid, -topn, axis=1)[:, -topn][:, None] # [n_p, 1] # threshold for top n selection
            # mask = mask & (r_mid_all >= thresholds) # [n_p, n_i] # update
            # image_ids = image_ids[r_mid >= thresholds].reshape(n_p, -1) # [n_p, rest_num_images]

            top_idx_in_rmid = np.argpartition(r_mid, -topn, axis=1)[:, -topn:]
            # update the image ids
            rows = np.arange(n_p)[:, None]
            image_ids = image_ids[rows, top_idx_in_rmid] # [n_p, topn]
            # update the global mask
            new_sub_mask = np.zeros_like(r_mid, dtype=bool)
            new_sub_mask[rows, top_idx_in_rmid] = True
            mask[mask.copy()] = new_sub_mask.ravel()
            
        # final timestep selection
        r_gt = r[:, :, -1][mask].reshape(n_p, -1) # [n_p, current_num_images]
        max_values = r_gt.max(axis=1) # [n_p]
        max_ids = image_ids[np.arange(n_p),r_gt.argmax(axis=1)] # [n_p] 
        return max_values.mean().item(), max_ids
            
    def ttsp_cost(self, alpha_s, steps_use):
        # of one image selection
        cost = 0 
        alpha_cum = 1 
        prev_step = 0 
        for j, s in enumerate(steps_use):
            cost += alpha_cum * ((s - prev_step) * self.x + self.y)
            alpha_cum *= alpha_s[j]
            prev_step = s
        cost += alpha_cum * ((self.max_step - 1 - prev_step) * self.x + self.y)
        return cost 
    
    def ttsp_run(self, r, image_num_use, alpha_s, steps_use, iters=10):
        assert (steps_use==np.sort(steps_use)).all(), "Steps must be in ascending order."
        assert len(alpha_s) == len(steps_use), "Length of alpha_s and steps_use must match."
        image_num_all = r.shape[1]
        out_rewards = []
        for i in range(iters):
            mask = random_mask(image_num_all, image_num_use)
            r_select = r[:, mask, :]
            out_reward, out_ids = self.ttsp(r_select, alpha_s, steps_use)
            out_rewards.append(out_reward)
        if iters == 1:
            return out_reward, out_ids
        else:
            return np.mean(out_rewards).item(), None
