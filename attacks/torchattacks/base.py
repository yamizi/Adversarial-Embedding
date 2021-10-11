from torchattacks.attack import Attack
import numpy as np

from attacks import success_strategy

class BaseAttack(Attack):

    def __init__(self, name:str, model, steps=5, min_pixel_value=0, max_pixel_value=1,
             threshold: float = 0.5, experiment=None, data_size=None, random_weight=0.01, random_noise="fading",
             strategy="binary"):
        super(BaseAttack, self).__init__(name, model)


        self.threshold = threshold
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value=max_pixel_value
        self.experiment = experiment
        self.data_size = data_size
        self.random_noise = random_noise
        self.random_weight = random_weight
        self.strategy = strategy
        self.norm = np.inf
        self.plotskip = max(1,steps//500)

        self.triangle_size = max(5,steps//100)
        from scipy.signal.windows import triang
        self.triangle = triang(self.triangle_size)

        self.best_adv_images = None
        self.best_adv_score = 0

        self.mean_adv_images = None
        self.mean_adv_score = 0

        self.max_adv_images = None
        self.max_adv_score = 0


    def best_images(self):
        imgs = {"best_adv_images":self.best_adv_images, "best_adv_score":self.best_adv_score, "mean_adv_images":self.mean_adv_images,
                "mean_adv_score":self.mean_adv_score, "max_adv_images":self.max_adv_images, "max_adv_score":self.max_adv_score }

        return imgs

    def plot_metrics(self, success_data, i, grad,cost, outputs=None,labels=None, adv=None, epoch=None):

        data_size = self.data_size if self.data_size is not None else labels.shape[1]
        #success = success_strategy(labels, outputs, strategy=self.strategy)
        success_data = success_strategy(labels, outputs, data_size=data_size, strategy=self.strategy) \
            if success_data is None else success_data


        if (self.experiment is not None and i % self.plotskip == 0):
            #self.experiment.log_metric("success_attack", success.mean().item(), step=i // self.plotskip,epoch=epoch )
            #if success.mean().item()>self.best_adv_score:
            #    self.best_adv_score = success.mean().item()
            #    self.best_adv_images =adv

            if self.strategy=="sorted":
                self.experiment.log_metric("success_data", (success_data>=1).mean().item(), step=i // self.plotskip,epoch=epoch)
            else:
                self.experiment.log_metric("success_data", success_data.mean().item(), step=i // self.plotskip,epoch=epoch)
            if success_data.mean().item()>self.mean_adv_score:
                self.mean_adv_score = success_data.mean().item()
                self.mean_adv_images =adv


            self.experiment.log_metric("success_data_max", success_data.max().item(), step=i // self.plotskip,epoch=epoch)
            if success_data.mean().item() > self.mean_adv_score:
                self.max_adv_score = success_data.max().item()
                self.max_adv_images = adv

            self.experiment.log_metric("success_data_min", success_data.min().item(), step=i // self.plotskip,epoch=epoch)
            self.experiment.log_metric("grad_attack", grad, step=i // self.plotskip,epoch=epoch)
            self.experiment.log_metric("loss", cost, step=i // self.plotskip,epoch=epoch)

            from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss, \
                explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error
            from sklearn.metrics import roc_auc_score, f1_score
            from scipy.stats import kendalltau

            y_score, y_true = outputs[:, :data_size].detach().cpu(), labels[:, :data_size].detach().cpu()

            # does not work
            # coverage_score = coverage_error(y_true, y_score)
            # self.experiment.log_metric("coverage_score", coverage_score, step=i // self.plotskip)

            if self.strategy == "binary":
                y_true_binary = (y_true > self.threshold).int()
                y_score_binary = (y_score > self.threshold).int()
                precision_score = label_ranking_average_precision_score(y_true_binary, y_score)
                self.experiment.log_metric("precision_score", precision_score, step=i // self.plotskip,epoch=epoch)

                ranking_score = label_ranking_loss(y_true_binary, y_score)
                self.experiment.log_metric("ranking_score", ranking_score, step=i // self.plotskip,epoch=epoch)

                auc = [roc_auc_score(y_true_binary[ii], y_) for (ii, y_) in enumerate(y_score)]
                self.experiment.log_metric("auc", np.mean(auc), step=i // self.plotskip,epoch=epoch)

                auc_binary = [roc_auc_score(y_true_binary[ii], y_) for (ii, y_) in enumerate(y_score_binary)]
                self.experiment.log_metric("auc_binary", np.mean(auc_binary), step=i // self.plotskip,epoch=epoch)

                f1_binary = [f1_score(y_true_binary[ii], y_) for (ii, y_) in enumerate(y_score_binary)]
                self.experiment.log_metric("f1_binary", np.mean(f1_binary), step=i // self.plotskip,epoch=epoch)

                # auc_binary = roc_auc_score(y_true_binary, y_score_binary)
                # self.experiment.log_metric("auc_binary", auc_binary, step=i // self.plotskip)

            # saturate fast
            explained_variance = explained_variance_score(y_true, y_score)
            self.experiment.log_metric("explained_variance_score", explained_variance, step=i // self.plotskip,epoch=epoch)

            mae = mean_absolute_error(y_true, y_score)
            self.experiment.log_metric("mean_absolute_error", mae, step=i // self.plotskip,epoch=epoch)
            mse_ = mean_squared_error(y_true, y_score)
            self.experiment.log_metric("mean_squared_error", mse_, step=i // self.plotskip,epoch=epoch)

            tau, p_value = kendalltau(y_true.numpy().flatten(), y_score.numpy().flatten())
            self.experiment.log_metric("kendall_tau", tau, step=i // self.plotskip,epoch=epoch)

            # only possible for >0 values
            # msl = mean_squared_log_error(y_true, y_score)
            # self.experiment.log_metric("mean_squared_log_error", msl, step=i // self.plotskip)

