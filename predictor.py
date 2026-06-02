import numpy as np
from tqdm import tqdm


class Predictor(object):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def predict(self, device):
        self.model.set_mode('eval')
        eval_im_paths = []

        eval_scores = []
        eval_targets = []

        for _, data_in in tqdm(enumerate(self.dataloader)):

            img_path, imgs, target = data_in
            
            score = self.model.predict(xs=imgs, device=device)            
 
            score = score.data.cpu().numpy()
            eval_scores.append(score)
                
            target = target.cpu().numpy().astype(int)
            eval_targets.append(target)
            
            eval_im_paths += img_path

        eval_scores = np.concatenate(eval_scores, axis=0)
        
        eval_targets = np.concatenate(eval_targets, axis=0)
        
        
        return eval_im_paths, eval_scores, eval_targets
