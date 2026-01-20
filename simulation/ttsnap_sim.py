import numpy as np

def random_mask(n, m):
    assert n >= m 
    mask = np.full(n, False, dtype=bool)
    true_indices = np.random.choice(n, size=m, replace=False)
    mask[true_indices] = True
    return mask


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

    def bon_run(self, r_gt, image_num_use, iters=10, use_multi=False):
        # r_gt: [num_prompts * num_images(all)] 
        # if use_multi is True, r_gt: [num_prompts, num_images(all), num_objectives]
        out_rewards = []
        image_num_all = r_gt.shape[1]
        # run mutliple iters for averaging
        for i in range(iters):
            mask = random_mask(image_num_all, image_num_use)
            if use_multi:
                r_gt_select = r_gt[:, mask, :]
                out_reward, out_ids = self.bon_multi(r_gt_select)
            else:
                r_gt_select = r_gt[:, mask]
                out_reward, out_ids = self.bon(r_gt_select)
            out_rewards.append(out_reward)
        if iters == 1:
            return out_reward, out_ids
        else:
            return np.mean(out_rewards).item(), None
    
    def bon_multi(self, r_gt):
        n_p, n_i, n_obj = r_gt.shape
        # argsort() gives the order, the second argsort() gives the rank position
        ranks = r_gt.argsort(axis=1).argsort(axis=1) # [n_p, n_i, n_obj]   
        rank_sum = ranks.sum(axis=2) # [n_p, n_i]
        max_ids = rank_sum.argmax(axis=1) # [n_p]
        best_rewards = r_gt[np.arange(n_p), max_ids] # [n_p, n_obj]        
        return best_rewards.mean(axis=0), max_ids

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
    
    def ttsp_multi(self, r, alpha_s, steps_use):
        n_p, n_i, n_s, n_obj = r.shape # num_prompts, num_images, num_steps, num_objectives
        mask = np.full((n_p, n_i), True, dtype=bool) 
        image_ids = np.tile(np.arange(n_i), (n_p, 1))

        for i, step in enumerate(steps_use):
            r_mid = r[:, :, step, :] # [n_p, current_num_images, n_obj]
            r_mid_multi = r_mid[mask].reshape(n_p, -1, n_obj) # [n_p, current_num_images, n_obj]

            ranks = r_mid_multi.argsort(axis=1).argsort(axis=1)  # [n_p, current_num_images, n_obj]

            r_mid_combined = ranks.sum(axis=2)  # [n_p, current_num_images]

            topn = max(1, round(r_mid_combined.shape[1] * alpha_s[i]))
            top_idx_in_rmid = np.argpartition(r_mid_combined, -topn, axis=1)[:, -topn:]

            # update the image ids
            rows = np.arange(n_p)[:, None]
            image_ids = image_ids[rows, top_idx_in_rmid] # [n_p, topn]
            # update the global mask
            new_sub_mask = np.zeros_like(r_mid_combined, dtype=bool)
            new_sub_mask[rows, top_idx_in_rmid] = True
            mask[mask.copy()] = new_sub_mask.ravel()
        
        # final timestep selection
        r_gt = r[:, :, -1, :][mask].reshape(n_p, -1, n_obj) # [n_p, current_num_images, n_obj]
        final_ranks = r_gt.argsort(axis=1).argsort(axis=1).sum(axis=-1)
        best_inner_idx = final_ranks.argmax(axis=1)
        max_ids = image_ids[np.arange(n_p), best_inner_idx]
        avg_reward = r_gt[np.arange(n_p), best_inner_idx].mean(axis=0).item()
        return avg_reward, max_ids

    def ttsp_run(self, r, image_num_use, alpha_s, steps_use, iters=10, use_multi=False):
        assert (steps_use==np.sort(steps_use)).all(), "Steps must be in ascending order."
        assert len(alpha_s) == len(steps_use), "Length of alpha_s and steps_use must match."
        image_num_all = r.shape[1]
        out_rewards = []
        for i in range(iters):
            mask = random_mask(image_num_all, image_num_use)
            if use_multi:
                r_select = r[:, mask, :, :]
                out_reward, out_ids = self.ttsp_multi(r_select, alpha_s, steps_use)
            else:
                r_select = r[:, mask, :]
                out_reward, out_ids = self.ttsp(r_select, alpha_s, steps_use)
            out_rewards.append(out_reward)
        if iters == 1:
            return out_reward, out_ids
        else:
            return np.mean(out_rewards).item(), None