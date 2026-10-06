#reverse-step 分布里出现了 nan / inf / 负数,最后在 diffusion/diffusion_utils.py 的 multinomial 采样时炸掉
import os
import time
from importlib import import_module

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from diffusion import diffusion_utils
from diffusion.noise_schedule import MarginalTransition, PredefinedNoiseScheduleDiscrete
from metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from metrics.train_loss import TrainLossDiscrete
from models.transformer import Denoiser
from runtime.train_methods import METHOD_PT_UNCONDITIONAL
from shared import flatten_optional_y
import utils


def resolve_denoiser(cfg):
    architecture = str(getattr(cfg.model, "condition_architecture", "global")).strip().lower()
    if architecture == "global":
        return Denoiser
    if architecture == "objective_token_residual":
        module = import_module("4_objective_token.objective_token.denoiser")
        return module.ObjectiveTokenDenoiser
    raise ValueError(f"Unknown condition architecture: {architecture}")


class Graph_DiT(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools):
        super().__init__()
        self.save_hyperparameters(ignore=["train_metrics", "sampling_metrics"])
        self.cfg = cfg
        self.name = cfg.general.name
        self.test_only = cfg.general.test_only
        self.T = cfg.model.diffusion_steps
        self.guide_scale = cfg.model.guide_scale
        self.train_method = getattr(cfg.general, "train_method", METHOD_PT_UNCONDITIONAL)

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.Posdim = input_dims["pos"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.Posdim_output = output_dims["pos"]
        self.ydim_output = output_dims["y"]
        self.node_dist = dataset_infos.nodes_dist
        self.dataset_info = dataset_infos
        self.max_n_nodes = dataset_infos.max_n_nodes

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_pos_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_pos_logp = SumExceptBatchMetric()
        self.val_y_collection = []
        self.val_y_raw_collection = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_pos_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_pos_logp = SumExceptBatchMetric()
        self.test_y_collection = []
        self.test_y_raw_collection = []

        denoiser = resolve_denoiser(cfg)
        self.model = denoiser(
            max_n_nodes=self.max_n_nodes,
            hidden_size=cfg.model.hidden_size,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            drop_condition=cfg.model.drop_condition,
            Xdim=self.Xdim,
            Edim=self.Edim,
            Posdim=self.Posdim,
            ydim=self.ydim,
            task_type=dataset_infos.task_type,
            condition_mode=getattr(dataset_infos, "condition_mode", "legacy_multi_slot"),
        )

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps,
        )

        x_marginals = dataset_infos.node_types.float()
        x_marginals = x_marginals / x_marginals.sum().clamp_min(1e-12)
        e_marginals = dataset_infos.edge_types.float()
        e_marginals = e_marginals / e_marginals.sum().clamp_min(1e-12)
        pos_marginals = dataset_infos.pos_types.float()
        pos_marginals = pos_marginals / pos_marginals.sum().clamp_min(1e-12)

        transition = dataset_infos.transition_E.float()
        xe_conditions = transition.sum(dim=1)
        xe_conditions = xe_conditions / xe_conditions.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ex_conditions = transition.sum(dim=0).transpose(0, 1)
        ex_conditions = ex_conditions / ex_conditions.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        self.transition_model = MarginalTransition(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            xe_conditions=xe_conditions,
            ex_conditions=ex_conditions,
            pos_marginals=pos_marginals,
            y_classes=self.ydim_output,
            n_nodes=self.max_n_nodes,
        )

        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=None, pos=pos_marginals)
        self.start_epoch_time = None
        self.train_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.save_chain_strategy = str(getattr(cfg.general, "save_chain_strategy", "average")).strip().lower()
        self.best_val_nll = 1e8
        self.val_counter = 0

    def forward(self, noisy_data, unconditioned=False, return_features=False, deterministic_policy=False):
        pred = self.model(
            noisy_data["X_t"].float(),
            noisy_data["E_t"].float(),
            noisy_data["pos_t"].float(),
            noisy_data["node_mask"],
            y=noisy_data["y_t"].float().clone(),
            t=noisy_data["t"],
            unconditioned=unconditioned,
            return_features=return_features,
            deterministic_policy=deterministic_policy,
        )
        return pred

    def _extract_batch_y(self, data):
        if hasattr(data, "y"):
            y = data.y
        elif isinstance(data, dict) and "y" in data:
            y = data["y"]
        else:
            raise ValueError("Training requires batch field 'y'.")
        return flatten_optional_y(y)

    def _extract_batch_y_raw(self, data):
        if hasattr(data, "y_raw"):
            y_raw = data.y_raw
        elif isinstance(data, dict) and "y_raw" in data:
            y_raw = data["y_raw"]
        else:
            y_raw = self._extract_batch_y(data)
        return flatten_optional_y(y_raw)

    def training_step(self, data, i):
        y = self._extract_batch_y(data)
        y_raw = self._extract_batch_y_raw(data)
        dense_data, node_mask = utils.batch_to_dense_graph(
            data,
            self.Xdim_output,
            self.Edim_output,
            self.Posdim_output,
            self.max_n_nodes,
        )
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, dense_data.pos, y, node_mask)
        pred = self.forward(noisy_data)
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            masked_pred_pos=pred.pos,
            pred_y=pred.y,
            true_X=dense_data.X,
            true_E=dense_data.E,
            true_pos=dense_data.pos,
            true_y=y_raw,
            node_mask=node_mask,
            log=i % self.log_every_steps == 0,
        )
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            masked_pred_pos=pred.pos,
            true_X=dense_data.X,
            true_E=dense_data.E,
            true_pos=dense_data.pos,
            log=i % self.log_every_steps == 0,
        )
        self.log("loss", loss, batch_size=dense_data.X.size(0), sync_dist=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self):
        self.train_iterations = self.trainer.datamodule.training_iterations
        print("on fit train iteration:", self.train_iterations)
        print(
            "Size of the input features Xdim {}, Edim {}, Posdim {}, ydim {}".format(
                self.Xdim,
                self.Edim,
                self.Posdim,
                self.ydim,
            )
        )

    def on_train_epoch_start(self):
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            print("Starting train epoch {}/{}...".format(self.current_epoch, self.trainer.max_epochs))
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self):
        log = self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time, log)
        self.train_metrics.log_epoch_metrics(self.current_epoch, log)

    def on_validation_epoch_start(self):
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_pos_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_pos_logp.reset()
        self.sampling_metrics.reset()
        self.val_y_collection = []
        self.val_y_raw_collection = []

    @torch.no_grad()
    def validation_step(self, data, i):
        y = self._extract_batch_y(data)
        y_raw = self._extract_batch_y_raw(data)
        dense_data, node_mask = utils.batch_to_dense_graph(
            data,
            self.Xdim_output,
            self.Edim_output,
            self.Posdim_output,
            self.max_n_nodes,
        )
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, dense_data.pos, y, node_mask)
        pred = self.forward(noisy_data)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, dense_data.pos, y_raw, node_mask, test=False)
        self.val_y_collection.append(y.detach())
        self.val_y_raw_collection.append(y_raw.detach())
        self.log("valid_nll", nll, batch_size=data.x.size(0), sync_dist=True)
        return {"loss": nll}

    def _checkpoint_requires_sampling_metric(self):
        monitor = getattr(self.cfg.general, "checkpoint_monitor", None)
        if monitor is None:
            return False
        monitor = str(monitor).strip().lower()
        return (
            monitor in {"mae", "nmae", "acc", "r2", "avg_mae", "avg_nmae", "avg_r2"}
            or monitor.endswith("/mae")
            or monitor.endswith("/nmae")
            or monitor.endswith("/acc")
            or monitor.endswith("/r2")
            or monitor.endswith("/avg_mae")
            or monitor.endswith("/avg_nmae")
            or monitor.endswith("/avg_r2")
        )

    def on_validation_epoch_end(self):
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
            self.val_pos_kl.compute() * self.T,
            self.val_X_logp.compute(),
            self.val_E_logp.compute(),
            self.val_pos_logp.compute(),
        ]
        if self.current_epoch / self.trainer.max_epochs in [0.25, 0.5, 0.75, 1.0]:
            print(
                f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- "
                f"Val Motif KL {metrics[1] :.2f} -- Val Edge KL {metrics[2] :.2f} -- "
                f"Val Pos KL {metrics[3] :.2f} Val loss: {metrics[0] :.2f} Best: {self.best_val_nll :.2f}\n"
            )
        self.log("val/NLL", metrics[0], sync_dist=True)
        if metrics[0] < self.best_val_nll:
            self.best_val_nll = metrics[0]
        self.val_counter += 1

        validation_sampling = bool(getattr(self.cfg.general, "validation_sampling", True))
        should_sample_for_monitor = validation_sampling and self._checkpoint_requires_sampling_metric()
        should_run_sampling = validation_sampling and (
            should_sample_for_monitor
            or (self.val_counter % self.cfg.general.sample_every_val == 0 and self.val_counter > 1)
        )
        if should_run_sampling:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples = []
            all_ys = []
            ident = 0
            self.val_y_collection = torch.cat(self.val_y_collection, dim=0)
            self.val_y_raw_collection = torch.cat(self.val_y_raw_collection, dim=0)
            num_examples = self.val_y_collection.size(0)
            start_index = 0
            while samples_left_to_generate > 0:
                bs = self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                if start_index + to_generate > num_examples:
                    start_index = 0
                if to_generate > num_examples:
                    ratio = to_generate // num_examples
                    self.val_y_collection = self.val_y_collection.repeat(ratio + 1, 1)
                    self.val_y_raw_collection = self.val_y_raw_collection.repeat(ratio + 1, 1)
                    num_examples = self.val_y_collection.size(0)
                batch_y = self.val_y_collection[start_index : start_index + to_generate]
                batch_y_raw = self.val_y_raw_collection[start_index : start_index + to_generate]
                all_ys.append(batch_y_raw)
                samples.extend(
                    self.sample_batch(
                        batch_id=ident,
                        batch_size=to_generate,
                        y=batch_y,
                        keep_chain=0,
                        number_chain_steps=self.number_chain_steps,
                    )
                )
                ident += to_generate
                start_index += to_generate
                samples_left_to_generate -= to_generate
            print("Computing sampling metrics ...")
            valid_smiles = self.sampling_metrics(
                samples,
                all_ys,
                self.name,
                self.current_epoch,
                val_counter=-1,
                test=False,
            )
            sample_metrics = getattr(self.sampling_metrics, "last_metrics", None)
            if isinstance(sample_metrics, dict):
                for key, value in sample_metrics.items():
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        self.log(f"val/{key}", float(value), sync_dist=True)
            print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")
            result_path = os.path.join(os.getcwd(), f"graphs/{self.name}/epoch{self.current_epoch}_b0/")
            self.visualization_tools.visualize_by_smiles(result_path, valid_smiles, self.cfg.general.samples_to_save)
            self.sampling_metrics.reset()

    def on_test_epoch_start(self):
        print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_pos_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_pos_logp.reset()
        self.test_y_collection = []
        self.test_y_raw_collection = []

    @torch.no_grad()
    def test_step(self, data, i):
        y = self._extract_batch_y(data)
        y_raw = self._extract_batch_y_raw(data)
        dense_data, node_mask = utils.batch_to_dense_graph(
            data,
            self.Xdim_output,
            self.Edim_output,
            self.Posdim_output,
            self.max_n_nodes,
        )
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, dense_data.pos, y, node_mask)
        pred = self.forward(noisy_data)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, dense_data.pos, y_raw, node_mask, test=True)
        self.test_y_collection.append(y.detach())
        self.test_y_raw_collection.append(y_raw.detach())
        return {"loss": nll}

    def on_test_epoch_end(self):
        metrics = [
            self.test_nll.compute(),
            self.test_X_kl.compute(),
            self.test_E_kl.compute(),
            self.test_pos_kl.compute(),
            self.test_X_logp.compute(),
            self.test_E_logp.compute(),
            self.test_pos_logp.compute(),
        ]
        print(
            f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- "
            f"Test Motif KL {metrics[1] :.2f} -- Test Edge KL {metrics[2] :.2f} -- Test Pos KL {metrics[3] :.2f}"
        )
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        chains_left_to_save = int(getattr(self.cfg.general, "chains_to_save", 0) or 0)
        samples = []
        saved_chains = []
        all_ys = []
        batch_id = 0
        test_y_collection = torch.cat(self.test_y_collection, dim=0)
        test_y_raw_collection = torch.cat(self.test_y_raw_collection, dim=0)
        num_examples = test_y_collection.size(0)
        if self.cfg.general.final_model_samples_to_generate > num_examples:
            ratio = self.cfg.general.final_model_samples_to_generate // num_examples
            test_y_collection = test_y_collection.repeat(ratio + 1, 1)
            test_y_raw_collection = test_y_raw_collection.repeat(ratio + 1, 1)
        while samples_left_to_generate > 0:
            bs = self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            chains_save = min(chains_left_to_save, bs)
            batch_y = test_y_collection[batch_id : batch_id + to_generate]
            batch_y_raw = test_y_raw_collection[batch_id : batch_id + to_generate]
            if chains_save > 0:
                batch_samples, batch_chains = self.sample_batch(
                    batch_id,
                    to_generate,
                    batch_y,
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                    return_chains=True,
                )
                saved_chains.extend(batch_chains)
            else:
                batch_samples = self.sample_batch(
                    batch_id,
                    to_generate,
                    batch_y,
                    keep_chain=0,
                    number_chain_steps=self.number_chain_steps,
                )
            samples.extend(batch_samples)
            all_ys.append(batch_y_raw)
            batch_id += to_generate
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        print("final Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(
            samples,
            all_ys,
            self.name,
            self.current_epoch,
            self.val_counter,
            test=True,
        )
        sample_metrics = getattr(self.sampling_metrics, "last_metrics", None)
        if isinstance(sample_metrics, dict):
            for key, value in sample_metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    self.log(f"test/{key}", float(value), sync_dist=True)
        self.sampling_metrics.reset()
        if len(saved_chains) > 0:
            chains_root = os.path.join(os.getcwd(), f"graphs/{self.name}/epoch{self.current_epoch}_test_chains")
            saved_path = self._save_chain_visualizations(saved_chains, chains_root)
            print(f"Saved chain visualizations to {saved_path}")
        print("Done.")

    def kl_prior(self, X, E, pos, node_mask):
        ones = torch.ones((X.size(0), 1), device=X.device)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=self.T * ones)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        bs, n, _ = X.shape
        prob_all = diffusion_utils.merge_graph(X, E) @ Qtb.X
        probX, probE = diffusion_utils.split_graph(prob_all, self.Xdim_output, self.Edim_output)
        prob_pos = torch.einsum("bijd,bdk->bijk", pos, Qtb.pos)

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        limit_pos = self.limit_dist.pos[None, None, None, :].expand(bs, n, n, -1).type_as(prob_pos)

        limit_X, limit_E, probX, probE, limit_pos, prob_pos = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
            true_pos=limit_pos.clone(),
            pred_pos=prob_pos,
        )
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_X, reduction="none")
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_E, reduction="none")
        kl_distance_pos = F.kl_div(input=prob_pos.log(), target=limit_pos, reduction="none")
        return (
            diffusion_utils.sum_except_batch(kl_distance_X)
            + diffusion_utils.sum_except_batch(kl_distance_E)
            + diffusion_utils.sum_except_batch(kl_distance_pos)
        )

    def compute_Lt(self, X, E, pos, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_pos = F.softmax(pred.pos, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
        Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

        bs, n, _ = X.shape
        X_all = diffusion_utils.merge_graph(X, E).float()
        Xt_all = diffusion_utils.merge_graph(noisy_data["X_t"], noisy_data["E_t"]).float()
        pred_probs_all = diffusion_utils.merge_graph(pred_probs_X, pred_probs_E).float()

        prob_true = diffusion_utils.posterior_distributions(X=X_all, X_t=Xt_all, Qt=Qt, Qsb=Qsb, Qtb=Qtb, X_dim=self.Xdim_output)
        prob_true.E = prob_true.E.reshape(bs, n, n, self.Edim_output)
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_all, X_t=Xt_all, Qt=Qt, Qsb=Qsb, Qtb=Qtb, X_dim=self.Xdim_output)
        prob_pred.E = prob_pred.E.reshape(bs, n, n, self.Edim_output)

        prob_true_pos = diffusion_utils.reverse_diffusion(pos, noisy_data["pos_t"], Qt.pos, Qsb.pos, Qtb.pos)
        prob_pred_pos = diffusion_utils.reverse_diffusion(pred_probs_pos, noisy_data["pos_t"], Qt.pos, Qsb.pos, Qtb.pos)

        prob_true.X, prob_true.E, prob_pred.X, prob_pred.E, prob_true_pos, prob_pred_pos = diffusion_utils.mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask,
            true_pos=prob_true_pos,
            pred_pos=prob_pred_pos,
        )

        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        kl_pos = (self.test_pos_kl if test else self.val_pos_kl)(prob_true_pos, torch.log(prob_pred_pos))
        return self.T * (kl_x + kl_e + kl_pos)

    def reconstruction_logp(self, t, X, E, pos, y, node_mask):
        beta_0 = self.noise_schedule(torch.zeros_like(t))
        Q0 = self.transition_model.get_Qt(beta_0, self.device)

        bs, n, _ = X.shape
        prob_all = diffusion_utils.merge_graph(X, E) @ Q0.X
        probX0, probE0 = diffusion_utils.split_graph(prob_all, self.Xdim_output, self.Edim_output)
        prob_pos0 = torch.einsum("bijd,bdk->bijk", pos, Q0.pos)
        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, prob_pos=prob_pos0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        pos0 = F.one_hot(sampled0.pos, num_classes=self.Posdim_output).float()
        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y, pos=pos0).mask(node_mask)
        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "pos_t": sampled_0.pos,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": torch.zeros(X0.shape[0], 1).type_as(y),
        }
        pred0 = self.forward(noisy_data)
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        probPos0 = F.softmax(pred0.pos, dim=-1)
        row_X = torch.ones(self.Xdim_output).type_as(probX0)
        row_E = torch.ones(self.Edim_output).type_as(probE0)
        row_pos = torch.ones(self.Posdim_output).type_as(probPos0)
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = torch.eye(probE0.size(1), device=probE0.device, dtype=torch.bool).unsqueeze(0).expand(probE0.size(0), -1, -1)
        probX0[~node_mask] = row_X
        probE0[~edge_mask] = row_E
        probPos0[~edge_mask] = row_pos
        probE0[diag_mask] = row_E
        probPos0[diag_mask] = row_pos
        return utils.PlaceHolder(X=probX0, E=probE0, y=None, pos=probPos0)

    def apply_noise(self, X, E, pos, y, node_mask):
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()
        s_int = t_int - 1
        t_float = t_int / self.T
        s_float = s_int / self.T
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        bs, n, _ = X.shape
        prob_all = diffusion_utils.merge_graph(X, E) @ Qtb.X
        probX, probE = diffusion_utils.split_graph(prob_all, self.Xdim_output, self.Edim_output)
        prob_pos = torch.einsum("bijd,bdk->bijk", pos, Qtb.pos)
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, prob_pos=prob_pos, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        pos_t = F.one_hot(sampled_t.pos, num_classes=self.Posdim_output).float()
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y, pos=pos_t).mask(node_mask)
        return {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "pos_t": z_t.pos,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

    def compute_val_loss(self, pred, noisy_data, X, E, pos, y, node_mask, test=False):
        t = noisy_data["t"]
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)
        kl_prior = self.kl_prior(X, E, pos, node_mask)
        loss_all_t = self.compute_Lt(X, E, pos, y, pred, noisy_data, node_mask, test)
        prob0 = self.reconstruction_logp(t, X, E, pos, y, node_mask)
        eps = 1e-8
        metric_X = self.test_X_logp if test else self.val_X_logp
        metric_E = self.test_E_logp if test else self.val_E_logp
        metric_pos = self.test_pos_logp if test else self.val_pos_logp
        loss_term_0 = (
            metric_X(X * (prob0.X + eps).log())
            + metric_E(E * (prob0.E + eps).log())
            + metric_pos(pos * (prob0.pos + eps).log())
        )
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        nll = (self.test_nll if test else self.val_nll)(nlls)
        return nll

    def _select_chain_timesteps(self, number_chain_steps, strategy=None):
        if self.T <= 0:
            return [0]
        strategy = str(strategy or self.save_chain_strategy or "average").strip().lower()
        num_steps = max(1, int(number_chain_steps))
        if strategy == "last":
            max_step = min(self.T, num_steps - 1)
            return list(range(max_step, -1, -1))
        if strategy != "average":
            raise ValueError(f"Unsupported save_chain_strategy: {strategy!r}")
        if num_steps == 1:
            return [0]
        raw = torch.linspace(float(self.T), 0.0, steps=num_steps)
        selected = []
        seen = set()
        for value in raw.tolist():
            t_int = int(round(value))
            if t_int not in seen:
                selected.append(t_int)
                seen.add(t_int)
        if self.T not in seen:
            selected.insert(0, int(self.T))
        if 0 not in seen:
            selected.append(0)
        return selected

    @staticmethod
    def _slice_collapsed_graph(collapsed_graph, node_counts, keep_chain, t_int):
        chain_batch = []
        keep = min(int(keep_chain), int(collapsed_graph.X.size(0)))
        for i in range(keep):
            n = int(node_counts[i].item())
            chain_batch.append(
                {
                    "sample_in_batch": i,
                    "t": int(t_int),
                    "X": collapsed_graph.X[i, :n].detach().cpu(),
                    "E": collapsed_graph.E[i, :n, :n].detach().cpu(),
                    "pos": collapsed_graph.pos[i, :n, :n].detach().cpu(),
                }
            )
        return chain_batch

    def _save_chain_visualizations(self, chains, output_dir):
        if len(chains) == 0:
            return None
        os.makedirs(output_dir, exist_ok=True)
        for chain in chains:
            sample_idx = int(chain.get("global_sample_idx", chain.get("sample_in_batch", 0)))
            prefix = f"sample_{sample_idx:04d}"
            for step in chain.get("steps", []):
                t_value = int(step["t"])
                file_path = os.path.join(output_dir, f"{prefix}_step_t{t_value:03d}.png")
                self.visualization_tools.visualize_chain_graph_step(
                    file_path,
                    step["X"],
                    step["E"],
                    step["pos"],
                    title=f"{prefix} | t={t_value}",
                )
        return output_dir

    @torch.no_grad()
    def sample_batch(self, batch_id, batch_size, y, keep_chain, number_chain_steps, num_nodes=None, return_chains=False):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, int):
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            n_nodes = num_nodes
        arange = torch.arange(self.max_n_nodes, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X = z_T.X
        E = z_T.E
        pos = z_T.pos

        chain_storage = None
        selected_timesteps = []
        if return_chains and int(keep_chain) > 0 and int(number_chain_steps) > 0:
            selected_timesteps = self._select_chain_timesteps(number_chain_steps, strategy=self.save_chain_strategy)
            chain_storage = [
                {
                    "batch_id": int(batch_id),
                    "global_sample_idx": int(batch_id + i),
                    "sample_in_batch": i,
                    "steps": [],
                }
                for i in range(min(int(keep_chain), batch_size))
            ]
            if self.T in selected_timesteps:
                collapsed_t = utils.PlaceHolder(
                    X=X.clone(),
                    E=E.clone(),
                    y=None,
                    pos=pos.clone(),
                ).mask(node_mask, collapse=True)
                initial_steps = self._slice_collapsed_graph(collapsed_t, n_nodes, keep_chain, self.T)
                for i, step_payload in enumerate(initial_steps):
                    chain_storage[i]["steps"].append(step_payload)

        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            sampled_s, _ = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, pos, y, node_mask)
            X, E, pos, y = sampled_s.X, sampled_s.E, sampled_s.pos, sampled_s.y
            if chain_storage is not None and s_int in selected_timesteps:
                collapsed_s = utils.PlaceHolder(
                    X=X.clone(),
                    E=E.clone(),
                    y=None,
                    pos=pos.clone(),
                ).mask(node_mask, collapse=True)
                current_steps = self._slice_collapsed_graph(collapsed_s, n_nodes, keep_chain, s_int)
                for i, step_payload in enumerate(current_steps):
                    chain_storage[i]["steps"].append(step_payload)

        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X = sampled_s.X
        E = sampled_s.E
        pos = sampled_s.pos
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            molecule_list.append([X[i, :n].cpu(), E[i, :n, :n].cpu(), pos[i, :n, :n].cpu()])
        if return_chains:
            return molecule_list, (chain_storage or [])
        return molecule_list

    def _compute_reverse_step_distributions_from_pred(self, pred, s, t, X_t, E_t, pos_t):
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)
        pred_pos = F.softmax(pred.pos, dim=-1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        Xt_all = diffusion_utils.merge_graph(X_t, E_t)
        pred_all = diffusion_utils.merge_graph(pred_X, pred_E)
        left = Xt_all @ Qt.X.transpose(-1, -2)
        right = pred_all @ Qsb.X
        den = (Qtb.X @ Xt_all.transpose(-1, -2)).transpose(-1, -2).clamp_min(1e-12)
        prob_all = left * right / den
        prob_all = prob_all / prob_all.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        prob_X, prob_E = diffusion_utils.split_graph(prob_all, self.Xdim_output, self.Edim_output)
        prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        prob_pos = diffusion_utils.reverse_diffusion(pred_pos, pos_t, Qt.pos, Qsb.pos, Qtb.pos)
        prob_pos = prob_pos / prob_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return prob_X, prob_E, prob_pos

    def get_reverse_step_distributions(self, s, t, X_t, E_t, pos_t, y_t, node_mask, pred=None, uncon_pred=None):
        noisy_data = {"X_t": X_t, "E_t": E_t, "pos_t": pos_t, "y_t": y_t, "t": t, "node_mask": node_mask}
        if pred is None:
            pred = self.forward(noisy_data)
        prob_X, prob_E, prob_pos = self._compute_reverse_step_distributions_from_pred(pred, s, t, X_t, E_t, pos_t)
        if self.ydim > 0 and self.guide_scale is not None and self.guide_scale != 1:
            if uncon_pred is None:
                uncon_pred = self.forward(noisy_data, unconditioned=True)
            uncon_prob_X, uncon_prob_E, uncon_prob_pos = self._compute_reverse_step_distributions_from_pred(
                uncon_pred,
                s,
                t,
                X_t,
                E_t,
                pos_t,
            )
            prob_X = uncon_prob_X * (prob_X / uncon_prob_X.clamp_min(1e-10)) ** self.guide_scale
            prob_E = uncon_prob_E * (prob_E / uncon_prob_E.clamp_min(1e-10)) ** self.guide_scale
            prob_pos = uncon_prob_pos * (prob_pos / uncon_prob_pos.clamp_min(1e-10)) ** self.guide_scale
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_pos = prob_pos / prob_pos.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        return prob_X, prob_E, prob_pos

    def sample_reverse_step(self, prob_X, prob_E, prob_pos, node_mask, step, y_t=None):
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_pos, node_mask=node_mask, step=step)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        pos_s = F.one_hot(sampled_s.pos, num_classes=self.Posdim_output).float()
        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t, pos=pos_s)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t, pos=pos_s)
        out_one_hot = out_one_hot.mask(node_mask)
        out_discrete = out_discrete.mask(node_mask, collapse=True)
        if y_t is not None:
            out_one_hot = out_one_hot.type_as(y_t)
            out_discrete = out_discrete.type_as(y_t)
        return out_one_hot, out_discrete

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, pos_t, y_t, node_mask):
        prob_X, prob_E, prob_pos = self.get_reverse_step_distributions(s, t, X_t, E_t, pos_t, y_t, node_mask)
        return self.sample_reverse_step(prob_X, prob_E, prob_pos, node_mask=node_mask, step=s[0, 0].item(), y_t=y_t)
