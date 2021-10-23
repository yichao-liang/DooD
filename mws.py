import itertools
import functools
import math
import collections
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


def get_arcs(start_point, primitives, ids):
    """
    Args:
        start_point: tensor [batch_size, 2]
        primitives: tensor [num_primitives, 5]
            primitives[i] = [dx, dy, theta, sharpness, width]
        ids: long tensor [batch_size, num_arcs]

    Returns:
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[i] = [x_start, y_start, dx, dy, theta, sharpness, width]
    """
    arcs = primitives[ids]  # [batch_size, num_arcs, 5]
    dxdy = arcs[..., :2]  # [batch_size, num_arcs, 5]
    start_and_dxdy = torch.cat(
        [start_point.unsqueeze(-2), dxdy], dim=-2
    )  # [batch_size, num_arcs + 1, 2]
    # [batch_size, num_arcs, 2]
    start_points = torch.cumsum(start_and_dxdy, dim=-2)[:, :-1]
    arcs = torch.cat([start_points, arcs], dim=-1)
    return arcs


def get_image_probs(ids, on_offs, start_point, primitives, rendering_params, num_rows, num_cols):
    """
    Args:
        ids: tensor [batch_size, num_arcs]
        on_offs: tensor [batch_size, num_arcs]
        start_point: tensor [batch_size, 2]
        primitives: tensor [num_primitives, 5]
            primitives[i] = [dx, dy, theta, sharpness, width]
        rendering_params: tensor [2]
            rendering_params[0] = scale of value
            rendering_params[1] = bias of value
        num_rows: int
        num_cols: int
    Returns:
        probs: tensor [batch_size, num_rows, num_cols]
    """
    arcs = get_arcs(start_point, primitives, ids)
    probs = get_probs(arcs, on_offs, rendering_params, num_rows, num_cols)
    return probs


class GenerativeModelIdsAndOnOffsDistribution(torch.distributions.Distribution):
    """Distribution on ids and on_offs."""

    def __init__(self, lstm_cell, linear, num_arcs, uniform_mixture, alphabet=None):
        """
        Args:
            lstm_cell: LSTMCell
            linear: Linear
            batch_size: int
            num_arcs: int
            uniform_mixture: bool or float

        Returns: distribution object with batch_shape [] and
            event_shape [num_arcs, 2]
        """
        super().__init__()
        self.lstm_cell = lstm_cell
        self.linear = linear
        self.num_arcs = num_arcs
        self.uniform_mixture = uniform_mixture

        self.lstm_hidden_size = self.lstm_cell.hidden_size
        self.lstm_input_size = self.lstm_cell.input_size
        self.alphabet = alphabet
        if self.alphabet is None:
            self._batch_shape = []
            self._event_shape = [num_arcs, 2]
            self.num_primitives = self.lstm_input_size - 2
        else:
            self.batch_size = alphabet.shape[0]
            self._batch_shape = [self.batch_size]
            self._event_shape = [num_arcs, 2]
            self.num_primitives = self.lstm_input_size - 2 - 50

    def sample(self, sample_shape=torch.Size()):
        """
        Args:
            sample_shape: torch.Size

        Returns:
            ids_and_on_offs: [*sample_shape, num_arcs, 2]
                ids_and_on_offs[..., 0] are ids
                ids_and_on_offs[..., 1] are on_offs
        """
        device = next(self.lstm_cell.parameters()).device
        num_samples = torch.tensor(sample_shape).long().prod()
        num_batch = torch.tensor(self.batch_shape).prod().long().item()

        if self.alphabet is None:
            lstm_input = torch.zeros((*sample_shape, self.lstm_input_size), device=device).view(
                -1, self.lstm_input_size
            )
        else:
            alphabet_expanded = (
                self.alphabet[None]
                .expand(num_samples, *self.batch_shape, 50)
                .contiguous()
                .view(-1, 50)
            )
            lstm_input = torch.cat(
                [
                    torch.zeros(
                        (num_samples * num_batch, self.lstm_input_size - 50), device=device
                    ),
                    alphabet_expanded,
                ],
                dim=-1,
            )
        h = torch.zeros((num_samples * num_batch, self.lstm_hidden_size), device=device)
        c = torch.zeros((num_samples * num_batch, self.lstm_hidden_size), device=device)

        ids = []
        on_offs = []
        for arc_id in range(self.num_arcs):
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(logits=id_logits)
            on_off_dist = torch.distributions.Categorical(logits=on_off_logits)

            ids.append(id_dist.sample())
            if arc_id == 0:
                on_off = torch.zeros(
                    # num_samples * self.batch_size, device=device).long()
                    num_samples * num_batch,
                    device=device,
                ).long()
            else:
                on_off = on_off_dist.sample()
            on_offs.append(on_off)

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[-1], num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[-1], num_classes=2).float(),
                    *([] if self.alphabet is None else [alphabet_expanded]),
                ],
                dim=1,
            )
        return torch.stack([torch.stack(ids, dim=1), torch.stack(on_offs, dim=1)], dim=-1).view(
            *sample_shape, *self.batch_shape, self.num_arcs, 2
        )

    def log_prob(self, ids_and_on_offs):
        """
        Args:
            ids_and_on_offs: [*sample_shape, num_arcs, 2]
                ids_and_on_offs[..., 0] are ids
                ids_and_on_offs[..., 1] are on_offs

        Returns: tensor [*sample_shape]
        """
        device = next(self.lstm_cell.parameters()).device
        num_batch = torch.tensor(self.batch_shape).prod().long().item()
        if self.alphabet is None:
            sample_shape = ids_and_on_offs.shape[:-2]
            num_samples = torch.tensor(sample_shape).prod().long().item()
            lstm_input = torch.zeros((*sample_shape, self.lstm_input_size), device=device).view(
                -1, self.lstm_input_size
            )
        else:
            sample_shape = ids_and_on_offs.shape[:-3]
            num_samples = torch.tensor(sample_shape).prod().long().item()
            alphabet_expanded = (
                self.alphabet[None].expand(num_samples, *self.batch_shape, 50).reshape(-1, 50)
            )
            lstm_input = torch.cat(
                [
                    torch.zeros(
                        (num_samples * num_batch, self.lstm_input_size - 50), device=device
                    ),
                    alphabet_expanded,
                ],
                dim=-1,
            )

        h = torch.zeros((num_samples * num_batch, self.lstm_hidden_size), device=device)
        c = torch.zeros((num_samples * num_batch, self.lstm_hidden_size), device=device)
        ids = ids_and_on_offs[..., 0]
        on_offs = ids_and_on_offs[..., 1]

        result = 0
        for arc_id in range(self.num_arcs):
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(
                logits=id_logits.view(*sample_shape, *self.batch_shape, self.num_primitives)
            )
            on_off_dist = torch.distributions.Categorical(
                logits=on_off_logits.view(*sample_shape, *self.batch_shape, 2)
            )

            result += id_dist.log_prob(ids[..., arc_id])
            if arc_id == 0:
                pass
            else:
                result += on_off_dist.log_prob(on_offs[..., arc_id])

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[..., arc_id].view(-1), num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[..., arc_id].view(-1), num_classes=2).float(),
                    *([] if self.alphabet is None else [alphabet_expanded]),
                ],
                dim=1,
            )
        return result


class Classifier(nn.Module):
    def __init__(self, num_classes, n_features=64, dropout=True):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_loc = cnn(16, n_hidden=16, dropout=False)
        self.fc_loc = nn.Linear(16, 6)
        self.cnn_out = cnn(n_features, n_hidden=n_features, dropout=dropout)
        self.fc_out = nn.Linear(n_features, num_classes)

    def forward(self, obs):
        obs = obs[:, None]
        theta = self.fc_loc(self.cnn_loc(obs).view(obs.shape[0], -1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, obs.size(), align_corners=True)
        obs_affine = F.grid_sample(obs, grid, align_corners=True)
        y = self.fc_out(self.cnn_out(obs_affine).view(obs.shape[0], -1))
        log_p = y.log_softmax(dim=-1)
        return log_p


@functools.lru_cache(maxsize=None)
def get_rotation_matrix(angle):
    angle = -angle
    return torch.tensor(
        [[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0]]
    ).float()


@functools.lru_cache(maxsize=None)
def get_translation_matrix(x, y):
    return torch.tensor([[1, 0, -x], [0, 1, y]]).float()


@functools.lru_cache(maxsize=None)
def get_shear_matrix(horizontal_angle, vertical_angle):
    return torch.tensor(
        [[1, math.tan(horizontal_angle), 0], [math.tan(vertical_angle), 1, 0]]
    ).float()


@functools.lru_cache(maxsize=None)
def get_scale_matrix(horizontal_squeeze, vertical_squeeze):
    return torch.tensor([[horizontal_squeeze, 0, 0], [0, vertical_squeeze, 0]]).float()


@functools.lru_cache(maxsize=None)
def compose2(theta_1, theta_2):
    return torch.mm(theta_2, torch.cat([theta_1, torch.tensor([[0, 0, 1]]).float()]))


@functools.lru_cache(maxsize=None)
def compose(*thetas):
    result = compose2(thetas[0], thetas[1])
    for i in range(2, len(thetas)):
        result = compose2(result, thetas[i])
    return result


@functools.lru_cache(maxsize=None)
def get_thetas(device):
    num_discretizations = 3
    shear_horizontal_angles = torch.linspace(-0.1 * math.pi, 0.1 * math.pi, num_discretizations)
    shear_vertical_angles = shear_horizontal_angles
    rotate_angles = shear_horizontal_angles
    scale_horizontal_squeezes = torch.linspace(0.9, 1.1, num_discretizations)
    scale_vertical_squeezes = scale_horizontal_squeezes
    translate_xs = torch.linspace(-0.2, 0.2, num_discretizations)
    translate_ys = translate_xs

    transform_paramss = itertools.product(
        shear_horizontal_angles,
        shear_vertical_angles,
        rotate_angles,
        scale_horizontal_squeezes,
        scale_vertical_squeezes,
        translate_xs,
        translate_ys,
    )
    result = []
    for (
        shear_horizontal_angle,
        shear_vertical_angle,
        rotate_angle,
        scale_horizontal_squeeze,
        scale_vertical_squeeze,
        translate_x,
        translate_y,
    ) in transform_paramss:
        shear_matrix = get_shear_matrix(shear_horizontal_angle, shear_vertical_angle)
        rotate_matrix = get_rotation_matrix(rotate_angle)
        scale_matrix = get_scale_matrix(scale_horizontal_squeeze, scale_vertical_squeeze)
        translate_matrix = get_translation_matrix(translate_x, translate_y)
        result.append(compose(shear_matrix, rotate_matrix, scale_matrix, translate_matrix))
    return torch.stack(result).to(device)


class AffineLikelihood(torch.distributions.Distribution):
    def __init__(self, cond, thetas=None):
        """
        Args:
            thetas: tensor [num_thetas, 2, 3]
            cond: tensor of probs [batch_size, 28, 28] in [0, 1]

        Returns: distribution object with batch_shape [batch_size] and
            event_shape [28, 28]
        """
        super().__init__()
        if thetas is None:
            self.thetas = get_thetas(cond.device)
        else:
            self.thetas = thetas
        self.num_thetas = len(self.thetas)
        self.batch_size = len(cond)
        self.cond_expanded = cond[None].expand(self.num_thetas, self.batch_size, 28, 28)
        self.grid = F.affine_grid(self.thetas, self.cond_expanded.size(), align_corners=True)

        # num_thetas, batch_size, 28, 28
        self.cond_transformed = F.grid_sample(self.cond_expanded, self.grid, align_corners=True)
        self.dist = torch.distributions.Independent(
            torch.distributions.Bernoulli(probs=self.cond_transformed), reinterpreted_batch_ndims=2
        )

    def log_prob(self, obs):
        """
        Args:
            obs: tensor [batch_size, 28, 28] in {0, 1}

        Returns: tensor [batch_size]
        """
        return torch.logsumexp(self.dist.log_prob(obs), dim=0) - math.log(self.num_thetas)

    def sample(self, sample_shape=torch.Size()):
        """
        Args:
            sample_shape: torch.Size

        Returns: [*sample_shape, batch_size, 28, 28]
        """
        mixture_ids = torch.randint(self.num_thetas, sample_shape)
        probs = self.cond_transformed[mixture_ids.view(-1)].view(
            *sample_shape, self.batch_size, 28, 28
        )
        return torch.distributions.Bernoulli(probs=probs).sample()


class ClassificationLikelihood(torch.distributions.Distribution):
    def __init__(self, classifier, class_conditional_model, cond):
        super().__init__()
        self.class_probs = classifier(cond)
        self.class_conditional_model = class_conditional_model

        self._batch_shape = [cond.shape[0]]
        # self._event_shape = [28, 28]

    def log_prob(self, obs):
        raise NotImplementedError()

    def log_prob_with_id(self, obs, obs_id, get_accuracy=False):
        n_classes = self.class_probs.shape[-1]

        n_obs = obs.shape[0]

        # obs = obs.unsqueeze(1).expand(n_obs, n_classes, *obs.shape[1:])\
        #                      .reshape(n_obs * n_classes, *obs.shape[1:])
        obs = None

        # obs_id = obs_id.unsqueeze(1).expand(n_obs, n_classes, *obs_id.shape[1:])\
        #                            .reshape(n_obs * n_classes, *obs_id.shape[1:])
        # classes = torch.arange(n_classes, device=obs_id.device).unsqueeze(0)\
        #               .expand(n_obs, n_classes)\
        #               .reshape(n_obs * n_classes)

        # log_likelihood = self.class_conditional_model.forward(classes, obs, obs_id)\
        #                     .reshape(n_obs, n_classes)
        log_likelihood = self.class_conditional_model.forward(obs, obs_id).reshape(n_obs, n_classes)

        marginal = (self.class_probs + log_likelihood).logsumexp(dim=-1)

        if get_accuracy:
            best_class_indices = log_likelihood.max(axis=-1).indices
            best_class_probs = self.class_probs.gather(1, best_class_indices.reshape(-1, 1))[:, 0]
            accuracy = best_class_probs.exp()
            return marginal, accuracy
        else:
            return marginal


class GenerativeModel(nn.Module):
    def __init__(
        self,
        num_primitives,
        initial_max_curve,
        big_arcs,
        lstm_hidden_size,
        num_rows,
        num_cols,
        num_arcs,
        likelihood="bernoulli",
        uniform_mixture=False,
        use_alphabet=False,
    ):
        super(GenerativeModel, self).__init__()
        self._prior = nn.Module()
        self._likelihood = nn.Module()

        self.lstm_hidden_size = lstm_hidden_size
        self.big_arcs = big_arcs

        self.use_alphabet = use_alphabet
        if use_alphabet:
            lstm_input_size = num_primitives + 2 + 50
        else:
            lstm_input_size = num_primitives + 2

        self.num_primitives = num_primitives
        # pre_primitives[i] = [pre_dx, pre_dy, pre_theta, pre_sharpness,
        #                      pre_width]
        # constraints:
        #   dx in [-0.66, 0.66]
        #   dy in [-0.66, 0.66]
        #   theta in [-pi, pi]
        #   sharpness in [-inf, inf]
        #   width in [0, 0.1]
        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L775
        pre_dxdy_min = -0.4
        pre_dxdy_max = 0.4
        # self._likelihood.pre_dxdy = nn.Parameter(torch.randn((num_primitives, 2)) * 0.25)
        self._likelihood.pre_dxdy = nn.Parameter(
            torch.rand((num_primitives, 2)) * (pre_dxdy_max - pre_dxdy_min) + pre_dxdy_min
        )

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L769
        # self._likelihood.pre_theta = nn.Parameter(torch.randn((num_primitives,)) *
        #                               math.pi / 4.)
        # account for the -pi in get_primitives

        init_theta_min = 1 - initial_max_curve
        init_theta_max = 1 + initial_max_curve
        theta_init = (
            torch.rand((num_primitives,)) * (init_theta_max - init_theta_min) + init_theta_min
        )
        if self.big_arcs:
            # self.theta_factor = 10
            # p = theta_init / 2
            # p = 0.01 + 0.98*p
            # assert p.min()>0 and p.max()<1
            # self._likelihood.pre_theta = nn.Parameter((p.log() - (1-p).log())/self.theta_factor)
            self._likelihood.pre_theta = nn.Parameter(theta_init)
        else:
            self.theta_factor = 10
            p = theta_init - 0.5
            p = 0.01 + 0.98 * p
            assert p.min() > 0 and p.max() < 1
            self._likelihood.pre_theta = nn.Parameter((p.log() - (1 - p).log()) / self.theta_factor)

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L773-L774
        self._likelihood.pre_sharpness = nn.Parameter(torch.ones(num_primitives) * 20.0)
        # modified so that exp(pre_sharpness) + 5 is approx. 20
        # self._likelihood.pre_sharpness = nn.Parameter(torch.ones(num_primitives) * 2.71)

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L771-L772
        self._likelihood.pre_width = nn.Parameter(torch.ones(num_primitives) * -2.0)

        self._prior.lstm_cell = nn.LSTMCell(
            input_size=lstm_input_size, hidden_size=lstm_hidden_size
        )
        self._prior.linear = nn.Linear(lstm_hidden_size, num_primitives + 2)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_arcs = num_arcs
        self.register_buffer("start_point", torch.tensor([0.5, 0.5]))

        # _likelihood.pre_rendering_params = [pre_scale, pre_bias]
        # constraints:
        #   scale in [0, inf]
        #   bias in [-inf, inf]
        self._likelihood.pre_rendering_params = nn.Parameter(torch.tensor([3.0, -3.0]))

        self.likelihood = likelihood
        self.pixelcnn_likelihood = likelihood == "pixelcnn" or "pcnn" in likelihood
        if self.pixelcnn_likelihood:
            raise NotImplementedError
        elif likelihood.startswith("classify"):
            if likelihood == "classify" or likelihood.startswith("classify:"):
                self.classifier = torch.load("classifier.pt", map_location=torch.device("cpu"))
                self.class_conditional_model = torch.load(
                    "cached_pixelcnn.pt", map_location=torch.device("cpu")
                )
                if ":" in likelihood:
                    args = {
                        item.split("=")[0]: eval(item.split("=")[1])
                        for item in likelihood.split(":")[1:]
                    }
                else:
                    args = {}
                if "weight" in args:
                    self.likelihood_weight = args["weight"]
            elif likelihood.startswith("classifytest"):
                if ":" in likelihood:
                    args = {
                        item.split("=")[0]: eval(item.split("=")[1])
                        for item in likelihood.split(":")[1:]
                    }
                else:
                    args = {}
                self.classifier = Classifier(num_classes=1623, **args)
                self.class_conditional_model = torch.load(
                    "cached_pixelcnn.pt", map_location=torch.device("cpu")
                )
            for p in [*self.classifier.parameters(), *self.class_conditional_model.parameters()]:
                p.requires_grad = False
        elif likelihood == "learned-affine":
            theta_affine = torch.randn(40, 2, 3) * 0.03
            theta_affine[:, 0, 0] += 1
            theta_affine[:, 1, 1] += 1
            self._likelihood.theta_affine = nn.Parameter(theta_affine)

        self.uniform_mixture = uniform_mixture

        assert set(self.parameters()) == set(self._prior.parameters()) | set(
            self._likelihood.parameters()
        )
        assert len(set(self._prior.parameters()) & set(self._likelihood.parameters())) == 0

    def get_rendering_params(self):
        scale = F.softplus(self._likelihood.pre_rendering_params[0]).unsqueeze(0)
        bias = self._likelihood.pre_rendering_params[1].unsqueeze(0)
        return torch.cat([scale, bias])

    def get_primitives(self):
        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L801-L804
        dxdy = torch.tanh(self._likelihood.pre_dxdy) * 0.66

        if self.big_arcs:
            #    theta = (torch.sigmoid(self._likelihood.pre_theta*self.theta_factor) - 0.5) * math.pi * 0.99 * 2
            theta = torch.remainder(self._likelihood.pre_theta * math.pi, 2 * math.pi) - math.pi
        else:
            theta = (
                (torch.sigmoid(self._likelihood.pre_theta * self.theta_factor) - 0.5)
                * math.pi
                * 0.99
            )

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L816-L820
        sharpness = self._likelihood.pre_sharpness
        # modified to be > 5
        # sharpness = torch.exp(self._likelihood.pre_sharpness) + 5

        # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L808-L814
        width = torch.sigmoid(self._likelihood.pre_width) * 0.1
        # modified to be sharper
        # width = torch.sigmoid(self._likelihood.pre_width) * 0.05
        return torch.cat(
            [dxdy, theta.unsqueeze(-1), sharpness.unsqueeze(-1), width.unsqueeze(-1)], dim=1
        )

    def get_latent_dist(self, alphabet=None):
        """Returns: distribution with batch shape [] and event shape
            [num_arcs, 2].
        """
        return GenerativeModelIdsAndOnOffsDistribution(
            self._prior.lstm_cell, self._prior.linear, self.num_arcs, self.uniform_mixture, alphabet
        )

    def get_obs_params(self, latent):
        """
        Args:
            latent: tensor [batch_size, num_arcs, 2]

        Returns:
            probs: tensor [batch_size, num_rows, num_cols]
        """
        batch_size = latent.shape[0]
        ids = latent[..., 0]
        on_offs = latent[..., 1]
        start_point = self.start_point.unsqueeze(0).expand(batch_size, -1)

        arcs = get_arcs(start_point, self.get_primitives(), ids)
        return get_logits(
            arcs, on_offs, self.get_rendering_params(), self.num_rows, self.num_cols
        )

    def get_obs_dist(self, latent):
        """
        Args:
            latent: tensor [batch_size, num_arcs, 2]

        Returns: distribution with batch_shape [batch_size] and event_shape
            [num_rows, num_cols]
        """
        logits = self.get_obs_params(latent)
        if self.pixelcnn_likelihood:
            raise NotImplementedError
        elif self.likelihood == "affine":
            return AffineLikelihood(logits.sigmoid())
        elif self.likelihood == "learned-affine":
            return AffineLikelihood(logits.sigmoid(), self._likelihood.theta_affine)
        elif self.likelihood.startswith("classify"):
            return ClassificationLikelihood(
                self.classifier, self.class_conditional_model, logits.sigmoid()
            )
        elif self.likelihood == "bernoulli":
            return torch.distributions.Independent(
                torch.distributions.Bernoulli(logits=logits), reinterpreted_batch_ndims=2
            )
        else:
            raise NotImplementedError()

    def get_log_prob(self, latent, obs, obs_id, get_accuracy=False):
        """Log of joint probability.

        Args:
            latent: tensor [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size, num_rows, num_cols]
                or tuple of
                    obs: tensor of shape [batch_size, num_rows, num_cols]
                    alphabet: tensor [batch_size, 50]
            obs_id: tensor of shape [batch_size]

        Returns: tensor of shape [num_particles, batch_size]
        """
        if self.use_alphabet:
            obs, alphabet = obs
        else:
            alphabet = None
        num_particles, batch_size, num_arcs, _ = latent.shape
        _, num_rows, num_cols = obs.shape
        latent_log_prob = self.get_latent_dist(alphabet).log_prob(latent)

        obs_dist = self.get_obs_dist(latent.view(num_particles * batch_size, num_arcs, 2))
        if hasattr(obs_dist, "log_prob_with_id"):
            obs_log_prob, accuracy = obs_dist.log_prob_with_id(
                obs[None]
                .expand(num_particles, batch_size, num_rows, num_cols)
                .reshape(num_particles * batch_size, num_rows, num_cols),
                obs_id[None].expand(num_particles, batch_size).reshape(num_particles * batch_size),
                get_accuracy=True,
            )
            obs_log_prob = obs_log_prob.view(num_particles, batch_size)
        else:
            obs_log_prob = obs_dist.log_prob(
                obs[None]
                .expand(num_particles, batch_size, num_rows, num_cols)
                .reshape(num_particles * batch_size, num_rows, num_cols)
            ).view(num_particles, batch_size)
            accuracy = None

        if hasattr(self, "likelihood_weight"):
            obs_log_prob = obs_log_prob * self.likelihood_weight

        if get_accuracy:
            return latent_log_prob + obs_log_prob, accuracy
        else:
            return latent_log_prob + obs_log_prob

    def get_log_probss(self, latent, obs, obs_id):
        """Log of joint probability.

        Args:
            latent: tensor [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size, num_rows, num_cols]
                or tuple of
                    obs: tensor of shape [batch_size, num_rows, num_cols]
                    alphabet: tensor [batch_size, 50]
            obs_id: tensor of shape [batch_size]

        Returns: tuple of tensor of shape [num_particles, batch_size]
        """

        if self.use_alphabet:
            obs, alphabet = obs
        else:
            alphabet = None

        num_particles, batch_size, num_arcs, _ = latent.shape
        _, num_rows, num_cols = obs.shape
        latent_log_prob = self.get_latent_dist(alphabet).log_prob(latent)
        obs_dist = self.get_obs_dist(latent.view(num_particles * batch_size, num_arcs, 2))
        if hasattr(obs_dist, "log_prob_with_id"):
            obs_log_prob = obs_dist.log_prob_with_id(
                obs[None]
                .expand(num_particles, batch_size, num_rows, num_cols)
                .reshape(num_particles * batch_size, num_rows, num_cols),
                obs_id[None].expand(num_particles, batch_size).reshape(num_particles * batch_size),
            ).view(num_particles, batch_size)
        else:
            obs_log_prob = obs_dist.log_prob(
                obs[None]
                .expand(num_particles, batch_size, num_rows, num_cols)
                .reshape(num_particles * batch_size, num_rows, num_cols)
            ).view(num_particles, batch_size)

        if hasattr(self, "likelihood_weight"):
            obs_log_prob = obs_log_prob * self.likelihood_weight

        return latent_log_prob, obs_log_prob

    def sample_latent_and_obs(self, alphabet=None, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            latent: tensor of shape [num_samples, num_arcs, 2]
            obs: tensor of shape [num_samples, num_rows, num_cols]
        """
        if self.use_alphabet:
            batch_size = alphabet.shape[0]
            latent_dist = self.get_latent_dist(alphabet)
            latent = latent_dist.sample((num_samples,))
            obs_dist = self.get_obs_dist(latent.view(num_samples * batch_size, self.num_arcs, 2))
            obs = obs_dist.sample().view(num_samples, batch_size, self.num_rows, self.num_cols)
        else:
            assert alphabet is None
            latent_dist = self.get_latent_dist()
            latent = latent_dist.sample((num_samples,))
            obs_dist = self.get_obs_dist(latent)
            obs = obs_dist.sample()

        return latent, obs

    def sample_obs(self, alphabet=None, num_samples=1):
        """Args:
            num_samples: int

        Returns:
            obs: tensor of shape [num_samples, num_rows, num_cols]
        """

        return self.sample_latent_and_obs(alphabet, num_samples)[1]


class InferenceNetworkIdsAndOnOffsDistribution(torch.distributions.Distribution):
    """Distribution on ids and on_offs."""

    def __init__(
        self, obs_embedding, lstm_cell, linear, num_arcs, uniform_mixture=False, alphabet=None
    ):
        """
        Args:
            obs_embedding: tensor [batch_size, obs_embedding_dim]
            lstm_cell: LSTMCell
            linear: Linear
            num_arcs: int
            uniform_mixture: bool

        Returns: distribution object with batch_shape [batch_size] and
            event_shape [num_arcs, 2]
        """
        super().__init__()
        self.obs_embedding = obs_embedding
        self.lstm_cell = lstm_cell
        self.linear = linear
        self.num_arcs = num_arcs
        self.uniform_mixture = uniform_mixture
        self.alphabet = alphabet

        self.batch_size = obs_embedding.shape[0]
        self.obs_embedding_dim = obs_embedding.shape[1]

        self.lstm_hidden_size = self.lstm_cell.hidden_size
        self.lstm_input_size = self.lstm_cell.input_size
        if self.alphabet is None:
            self.num_primitives = self.lstm_input_size - 2 - self.obs_embedding_dim
        else:
            self.num_primitives = self.lstm_input_size - 2 - 50 - self.obs_embedding_dim
        self._batch_shape = [self.batch_size]
        self._event_shape = [num_arcs, 2]

    def sample(self, sample_shape=torch.Size()):
        """
        Args:
            sample_shape: torch.Size

        Returns:
            ids_and_on_offs: [*sample_shape, batch_size, num_arcs, 2]
                ids_and_on_offs[..., 0] are ids
                ids_and_on_offs[..., 1] are on_offs
        """
        device = next(self.lstm_cell.parameters()).device
        num_samples = torch.tensor(sample_shape).prod()

        if self.alphabet is None:
            lstm_input = torch.zeros(
                (*sample_shape, self.batch_size, self.lstm_input_size), device=device
            ).view(-1, self.lstm_input_size)
            h = torch.zeros(
                (*sample_shape, self.batch_size, self.lstm_hidden_size), device=device
            ).view(-1, self.lstm_hidden_size)
            c = torch.zeros(
                (*sample_shape, self.batch_size, self.lstm_hidden_size), device=device
            ).view(-1, self.lstm_hidden_size)
            obs_embedding_expanded = (
                self.obs_embedding[None]
                .expand(np.prod(sample_shape), self.batch_size, self.obs_embedding_dim)
                .reshape(-1, self.obs_embedding_dim)
            )
            lstm_input[:, -self.obs_embedding_dim :] = obs_embedding_expanded
        else:
            alphabet_expanded = (
                self.alphabet[None].expand(num_samples, self.batch_size, 50).reshape(-1, 50)
            )
            obs_embedding_expanded = (
                self.obs_embedding[None]
                .expand(num_samples, self.batch_size, self.obs_embedding_dim)
                .reshape(-1, self.obs_embedding_dim)
            )
            lstm_input = torch.cat(
                [
                    torch.zeros(
                        (
                            num_samples * self.batch_size,
                            self.lstm_input_size - 50 - self.obs_embedding_dim,
                        ),
                        device=device,
                    ),
                    alphabet_expanded,
                    obs_embedding_expanded,
                ],
                dim=-1,
            )
            h = torch.zeros((num_samples * self.batch_size, self.lstm_hidden_size), device=device)
            c = torch.zeros((num_samples * self.batch_size, self.lstm_hidden_size), device=device)

        ids = []
        on_offs = []
        for arc_id in range(self.num_arcs):
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(logits=id_logits)
            on_off_dist = torch.distributions.Categorical(logits=on_off_logits)

            ids.append(id_dist.sample())
            if arc_id == 0:
                on_off = (
                    torch.zeros((*sample_shape, self.batch_size), device=device).long().view(-1)
                )
            else:
                on_off = on_off_dist.sample()
            on_offs.append(on_off)

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[-1], num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[-1], num_classes=2).float(),
                    *([] if self.alphabet is None else [alphabet_expanded]),
                    obs_embedding_expanded,
                ],
                dim=1,
            )

        return torch.stack([torch.stack(ids, dim=1), torch.stack(on_offs, dim=1)], dim=-1).view(
            *sample_shape, self.batch_size, self.num_arcs, 2
        )

    def log_prob(self, ids_and_on_offs):
        """
        Args:
            ids_and_on_offs: [*sample_shape, batch_size, num_arcs, 2]
                ids_and_on_offs[..., 0] are ids
                ids_and_on_offs[..., 1] are on_offs

        Returns: tensor [*sample_shape, batch_size]
        """
        device = next(self.lstm_cell.parameters()).device
        sample_shape = ids_and_on_offs.shape[:-3]
        ids = ids_and_on_offs[..., 0]
        on_offs = ids_and_on_offs[..., 1]
        num_samples = int(torch.tensor(sample_shape).prod().item())

        if self.alphabet is None:
            lstm_input = torch.zeros(
                (*sample_shape, self.batch_size, self.lstm_input_size), device=device
            ).view(-1, self.lstm_input_size)
            h = torch.zeros(
                (*sample_shape, self.batch_size, self.lstm_hidden_size), device=device
            ).view(-1, self.lstm_hidden_size)
            c = torch.zeros(
                (*sample_shape, self.batch_size, self.lstm_hidden_size), device=device
            ).view(-1, self.lstm_hidden_size)
            obs_embedding_expanded = (
                self.obs_embedding[None]
                .expand(int(np.prod(sample_shape)), self.batch_size, self.obs_embedding_dim)
                .reshape(-1, self.obs_embedding_dim)
            )
            lstm_input[:, -self.obs_embedding_dim :] = obs_embedding_expanded
        else:
            alphabet_expanded = (
                self.alphabet[None].expand(num_samples, self.batch_size, 50).reshape(-1, 50)
            )
            obs_embedding_expanded = (
                self.obs_embedding[None]
                .expand(num_samples, self.batch_size, self.obs_embedding_dim)
                .reshape(-1, self.obs_embedding_dim)
            )
            lstm_input = torch.cat(
                [
                    torch.zeros(
                        (
                            num_samples * self.batch_size,
                            self.lstm_input_size - 50 - self.obs_embedding_dim,
                        ),
                        device=device,
                    ),
                    alphabet_expanded,
                    obs_embedding_expanded,
                ],
                dim=-1,
            )
            h = torch.zeros((num_samples * self.batch_size, self.lstm_hidden_size), device=device)
            c = torch.zeros((num_samples * self.batch_size, self.lstm_hidden_size), device=device)

        result = 0
        for arc_id in range(self.num_arcs):
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.linear(h)
            if self.uniform_mixture:
                p = self.uniform_mixture if type(self.uniform_mixture) is float else 0.2
                id_logits = util.logit_uniform_mixture(logits[:, : self.num_primitives], p)
                on_off_logits = util.logit_uniform_mixture(logits[:, self.num_primitives :], p)
            else:
                id_logits = logits[:, : self.num_primitives]
                on_off_logits = logits[:, self.num_primitives :]
            id_dist = torch.distributions.Categorical(
                logits=id_logits.view(*sample_shape, self.batch_size, self.num_primitives)
            )
            on_off_dist = torch.distributions.Categorical(
                logits=on_off_logits.view(*sample_shape, self.batch_size, 2)
            )

            result += id_dist.log_prob(ids[..., arc_id])
            if arc_id == 0:
                pass
            else:
                result += on_off_dist.log_prob(on_offs[..., arc_id])

            lstm_input = torch.cat(
                [
                    F.one_hot(ids[..., arc_id].view(-1), num_classes=self.num_primitives).float(),
                    F.one_hot(on_offs[..., arc_id].view(-1), num_classes=2).float(),
                    *([] if self.alphabet is None else [alphabet_expanded]),
                    obs_embedding_expanded,
                ],
                dim=1,
            )
        return result


def cnn(output_dim, n_hidden=128, dropout=False):
    l = []
    l.append(nn.Conv2d(1, int(n_hidden / 2), kernel_size=3, padding=2))
    if dropout:
        l.append(nn.Dropout2d())
    l.append(nn.ReLU(inplace=True))
    l.append(nn.MaxPool2d(kernel_size=3, stride=2))
    l.append(nn.Conv2d(int(n_hidden / 2), n_hidden, kernel_size=3, padding=1))
    # if dropout: l.append(nn.Dropout2d())
    l.append(nn.ReLU(inplace=True))
    l.append(nn.MaxPool2d(kernel_size=3, stride=2))
    l.append(nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=0))
    if dropout:
        l.append(nn.Dropout2d())
    l.append(nn.ReLU(inplace=True))
    l.append(nn.Conv2d(n_hidden, output_dim, kernel_size=3, padding=0))
    # if dropout: l.append(nn.Dropout2d())
    l.append(nn.ReLU(inplace=True))
    l.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*l)


class InferenceNetwork(nn.Module):
    def __init__(
        self,
        num_primitives,
        lstm_hidden_size,
        num_rows,
        num_cols,
        num_arcs,
        obs_embedding_dim,
        uniform_mixture=False,
        use_alphabet=False,
    ):
        super(InferenceNetwork, self).__init__()
        self.use_alphabet = use_alphabet
        self.obs_embedding_dim = obs_embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        if self.use_alphabet:
            self.lstm_input_size = num_primitives + 2 + 50 + obs_embedding_dim
        else:
            self.lstm_input_size = num_primitives + 2 + obs_embedding_dim
        self.lstm_cell = nn.LSTMCell(
            input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size
        )
        self.obs_embedder = cnn(self.obs_embedding_dim)
        # TODO: consider feeding partial image into the LSTM
        # self.partial_image_embedder_cnn = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, self.observed_obs_embedding_dim,
        #               kernel_size=3, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.linear = nn.Linear(self.lstm_hidden_size, num_primitives + 2)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_arcs = num_arcs
        self.register_buffer("start_point", torch.tensor([0.5, 0.5]))
        self.uniform_mixture = uniform_mixture

    def get_obs_embedding(self, obs):
        """
        Args:
            obs: tensor [batch_size, num_rows, num_cols]

        Returns: tensor [batch_size, obs_embedding_dim]
        """
        batch_size = obs.shape[0]
        # result = self.obs_embedder(obs.unsqueeze(1)).view(batch_size, -1)
        result = self.obs_embedder(obs.unsqueeze(1)).view(batch_size, -1)
        breakpoint()
        assert result.shape[1] == self.obs_embedding_dim
        return result

    def get_latent_dist(self, obs):
        """Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]
                or tuple of
                    obs: tensor of shape [batch_size, num_rows, num_cols]
                    alphabet: tensor [batch_size, 50]
        Returns: distribution with batch_shape [batch_size] and
            event_shape [num_arcs, 2]
        """
        if self.use_alphabet:
            obs, alphabet = obs
        else:
            alphabet = None
        return InferenceNetworkIdsAndOnOffsDistribution(
            self.get_obs_embedding(obs),
            self.lstm_cell,
            self.linear,
            self.num_arcs,
            self.uniform_mixture,
            alphabet=alphabet,
        )

    def sample_from_latent_dist(self, latent_dist, num_particles):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch_shape [batch_size] and
                event_shape [num_arcs, 2]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        return latent_dist.sample((num_particles,))

    def sample(self, obs, num_particles):
        """Samples from q(latent | obs)

        Args:
            obs: tensor of shape [batch_size, num_rows, num_cols]
                or tuple of
                    obs: tensor of shape [batch_size, num_rows, num_cols]
                    alphabet: tensor [batch_size, 50]
            num_particles: int

        Returns:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
        """
        latent_dist = self.get_latent_dist(obs)
        return self.sample_from_latent_dist(latent_dist, num_particles)

    def get_log_prob_from_latent_dist(self, latent_dist, latent):
        """Log q(latent | obs).

        Args:
            latent_dist: distribution with batch_shape [batch_size] and
                event_shape [num_arcs, 2]
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return latent_dist.log_prob(latent)

    def get_log_prob(self, latent, obs):
        """Log q(latent | obs).

        Args:
            latent: tensor of shape [num_particles, batch_size, num_arcs, 2]
            obs: tensor of shape [batch_size, num_rows, num_cols]

        Returns: tensor of shape [num_particles, batch_size]
        """
        return self.get_log_prob_from_latent_dist(self.get_latent_dist(obs), latent)

def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test-run", action="store_true", help="")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--condition-on-alphabet", action="store_true", help=" ")

    # data
    parser.add_argument("--dataset", default="omniglot", help=" ")
    parser.add_argument("--data-location", default="local", help=" ")
    parser.add_argument("--batch-size", type=int, default=250, help=" ")
    parser.add_argument("--small-dataset", action="store_true", help="use a small dataset")
    parser.add_argument("--dataset-size", default=None, type=int)

    # init
    # parser.add_argument('--uniform-mixture', action='store_true', help=' ')
    parser.add_argument(
        "--likelihood",
        # default="learned-affine",
        default="affine",
        choices=["bernoulli", "learned-affine", "affine", "classify"],
        help=" ",
    )
    parser.add_argument("--num-arcs", type=int, default=10, help=" ")

    parser.add_argument("--p-lstm-hidden-size", type=int, default=4, help=" ")
    parser.add_argument("--p-uniform-mixture", default=0.0, type=float)

    parser.add_argument("--q-lstm-hidden-size", type=int, default=64, help=" ")
    parser.add_argument("--q-uniform-mixture", default=0.0, type=float)

    parser.add_argument("--num-particles", type=int, default=10, help=" ")
    parser.add_argument("--memory-size", type=int, default=10, help=" ")
    parser.add_argument("--obs-embedding-dim", type=int, default=64, help=" ")
    parser.add_argument("--num-primitives", type=int, default=64, help=" ")
    parser.add_argument("--initial-max-curve", type=float, default=0.3, help=" ")
    parser.add_argument("--big-arcs", action="store_true", help=" ")

    # train
    parser.add_argument("--algorithm", default="mws", choices=["mws", "rws", "vimco"], help=" ")
    parser.add_argument("--prior-lr-factor", default=1.0, type=float)
    # parser.add_argument("--pretrain-iterations", type=int, default=10000, help=" ")
    parser.add_argument("--pretrain-iterations", type=int, default=5, help=" ")
    parser.add_argument("--prior-anneal-iterations", type=int, default=0, help=" ")
    parser.add_argument("--num-iterations", type=int, default=200000, help=" ")
    parser.add_argument("--log-interval", type=int, default=1, help=" ")
    parser.add_argument("--save-interval", type=int, default=1000, help=" ")
    parser.add_argument("--test-interval", type=int, default=999999, help=" ")
    parser.add_argument("--test-batch-size", type=int, default=5, help=" ")
    parser.add_argument("--legacy-index", action="store_true")
    parser.add_argument("--test-num-particles", type=int, default=1000, help=" ")
    parser.add_argument("--checkpoint-path-prefix", default="checkpoint", help=" ")

    return parser

def init(run_args, device):
    generative_model = GenerativeModel(
        run_args.num_primitives,
        run_args.initial_max_curve,
        run_args.big_arcs,
        run_args.p_lstm_hidden_size,
        run_args.num_rows,
        run_args.num_cols,
        run_args.num_arcs,
        run_args.likelihood,
        run_args.p_uniform_mixture,
        use_alphabet=run_args.condition_on_alphabet,
    ).to(device)
    inference_network = InferenceNetwork(
        run_args.num_primitives,
        run_args.q_lstm_hidden_size,
        run_args.num_rows,
        run_args.num_cols,
        run_args.num_arcs,
        run_args.obs_embedding_dim,
        run_args.q_uniform_mixture,
        use_alphabet=run_args.condition_on_alphabet,
    ).to(device)
    optimizer = init_optimizer(generative_model, inference_network, 1,)

    stats = Stats([], [], [], [], [], [], [], [])

    if "mws" in run_args.algorithm:
        memory = init_memory(
            run_args.num_train_data,
            run_args.memory_size,
            generative_model.num_arcs,
            generative_model.num_primitives,
            device,
        )
    else:
        memory = None

    return generative_model, inference_network, optimizer, memory, stats

import functools

import torch
import math


def safe_div(x, y, large=1e7):
    return torch.clamp(x / y, -large, large)


def get_center(arcs):
    """Centre of the arc

    Args:
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns: tensor [batch_size, num_arcs, 2]
    """
    x_start = arcs[..., 0]
    y_start = arcs[..., 1]
    dx = arcs[..., 2]
    dy = arcs[..., 3]
    theta = arcs[..., 4]

    x_center = x_start + dx / 2 + safe_div(dy, (2 * torch.tan(theta)))
    y_center = y_start + dy / 2 - safe_div(dx, (2 * torch.tan(theta)))
    return torch.stack([x_center, y_center], dim=-1)


def get_normal(xs):
    """Normals of xs.

    Args:
        xs: tensor [num_xs, 2]

    Returns: tensor [num_xs, 2]
    """

    return torch.stack([xs[:, 1], -xs[:, 0]], dim=-1)


def same_half_plane(normals, x_1, x_2):
    """
    Args:
        normals: tensor [batch_size, num_arcs, 2]
        x_1: tensor [batch_size, num_arcs, 2]
        x_2: tensor [batch_size, num_points, num_arcs, 2]

    Returns: binary tensor [batch_size, num_points, num_arcs]
    """

    return (
        torch.einsum("bai,bai->ba", normals, x_1).unsqueeze(1)
        * torch.einsum("bai,bpai->bpa", normals, x_2)
    ) >= 0


def get_inside_sector(points, arcs):
    """Are `points` inside the sector defined by `arcs`?

    Args:
        points: tensor [num_points, 2]
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns: binary tensor [batch_size, num_points, num_arcs]
    """

    centers = get_center(arcs)  # [batch_size, num_arcs, 2]
    start_points = arcs[..., :2]  # [batch_size, num_arcs, 2]
    end_points = start_points + arcs[..., 2:4]  # [batch_size, num_arcs, 2]

    a = start_points - centers  # [batch_size, num_arcs, 2]
    b = end_points - centers  # [batch_size, num_arcs, 2]
    # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
    # [batch_size, num_points, num_arcs, 2]
    x = points.unsqueeze(1).unsqueeze(0) - centers.unsqueeze(1)

    batch_size, num_arcs, _ = a.shape
    a_normals = get_normal(a.view(batch_size * num_arcs, -1)).view(
        batch_size, num_arcs, -1
    )  # [batch_size, num_arcs, 2]
    b_normals = get_normal(b.view(batch_size * num_arcs, -1)).view(
        batch_size, num_arcs, -1
    )  # [batch_size, num_arcs, 2]

    return same_half_plane(a_normals, b, x) & same_half_plane(b_normals, a, x)


def distance_to_arc(points, arcs):
    """Distance of `points` to the `arcs`.

    Args:
        points: tensor [num_points, 2]
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns:
        distance: tensor [batch_size, num_points, num_arcs]
    """

    centers = get_center(arcs)  # [batch_size, num_arcs, 2] TODO
    start_points = arcs[..., :2]  # [batch_size, num_arcs, 2]
    end_points = start_points + arcs[..., 2:4]  # [batch_size, num_arcs, 2]

    # inside sector
    distance_from_center = torch.norm(
        # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
        # [batch_size, num_points, num_arcs, 2]
        points.unsqueeze(1).unsqueeze(0) - centers.unsqueeze(1),
        p=2,
        dim=-1,
    )  # [batch_size, num_points, num_arcs]
    # [batch_size, num_arcs]
    arc_radius = torch.norm(start_points - centers, p=2, dim=-1)
    # [batch_size, num_points, num_arcs]
    inside_sector_distance = torch.abs(distance_from_center - arc_radius.unsqueeze(1))

    # outside sector
    start_points_distance = torch.norm(
        # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
        # [batch_size, num_points, num_arcs, 2]
        points.unsqueeze(1).unsqueeze(0) - start_points.unsqueeze(1),
        p=2,
        dim=-1,
    )  # [batch_size, num_points, num_arcs]
    end_points_distance = torch.norm(
        # [1, num_points, 1, 2] - [batch_size, 1, num_arcs, 2] =
        # [batch_size, num_points, num_arcs, 2]
        points.unsqueeze(1).unsqueeze(0) - end_points.unsqueeze(1),
        p=2,
        dim=-1,
    )  # [batch_size, num_points, num_arcs]
    # [batch_size, num_points, num_arcs]
    outside_sector_distance = torch.min(start_points_distance, end_points_distance)

    # [batch_size, num_arcs]
    theta = arcs[:, :, 4]
    # [batch_size, num_points, num_arcs]
    small_arc = (torch.abs(theta[:, None, :]) < (math.pi / 2)).float()
    large_arc = 1 - small_arc

    # [batch_size, num_points, num_arcs]
    inside_sector = get_inside_sector(points, arcs).float()
    outside_sector = 1 - inside_sector
    return small_arc * (
        inside_sector * inside_sector_distance + outside_sector * outside_sector_distance
    ) + large_arc * (
        inside_sector * outside_sector_distance + outside_sector * inside_sector_distance
    )


def get_value(points, arcs, scale, bias):
    """
    Args:
        points: tensor [num_points, 2]
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]

    Returns:
        value: tensor [batch_size, num_points, num_arcs]
    """
    sharpness = arcs[..., -2]
    width = arcs[..., -1]
    # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L916-L919
    # and
    # https://github.com/insperatum/wsvae/blob/master/examples/mnist/mnist.py#L973-L991
    return (
        scale
        * torch.clamp(
            torch.exp(
                sharpness.unsqueeze(1) ** 2
                * (width.unsqueeze(1) ** 2 - distance_to_arc(points, arcs) ** 2)
            ),
            0,
            1,
        )
        + bias
    )
    # return -sharpness.unsqueeze(1) * (
    #     distance_to_arc(points, arcs) - width.unsqueeze(1)
    # )


@functools.lru_cache(maxsize=None)
def get_grid_points(num_rows, num_cols, dtype, device):
    """Returns grid points.

    Args:
        num_rows: int
        num_cols: int
        dtype
        device

    Returns:
        points: [num_rows * num_cols, 2]
    """
    xs = torch.linspace(0, 1, num_cols, dtype=dtype, device=device)
    ys = torch.linspace(0, 1, num_rows, dtype=dtype, device=device)
    xss, yss = torch.meshgrid(xs, ys)
    points = torch.stack([xss, yss], dim=-1).view(-1, 2)  # [num_points, 2]
    return points


def get_probs(*args, **kwargs):
    return get_logits(*args, **kwargs).sigmoid()


def get_logits(arcs, on_offs, rendering_params, num_rows, num_cols):
    """Turns arc specs into Bernoulli probabilities.

    Args:
        arcs: tensor [batch_size, num_arcs, 7]
            arcs[b, i] = [x_start, y_start, dx, dy, theta, sharpness, width]
        on_offs: long tensor [batch_size, num_arcs]
        rendering_params: tensor [2]
            rendering_params[0] = scale of value
            rendering_params[1] = bias of value
        num_rows: int
        num_cols: int

    Returns:
        probs: tensor [batch_size, num_rows, num_cols]"""

    # [num_points, 2]
    points = get_grid_points(num_rows, num_cols, arcs.dtype, arcs.device)
    batch_size = arcs.shape[0]
    scale, bias = rendering_params
    result = (
        torch.max(
            get_value(points, arcs, scale, bias) + on_offs.float().log().unsqueeze(-2), dim=-1
        )[0]
        .view(batch_size, num_cols, num_rows)
        .transpose(-2, -1)
    )
    epsilon = math.log(1e-6)
    result = torch.logsumexp(
        torch.stack([result, torch.ones(result.shape, device=result.device) * epsilon]), dim=0
    )
    return torch.flip(result, [-2])

def init_optimizer(generative_model, inference_network, prior_lr_factor):
    lr = 1e-3
    optimizer = torch.optim.Adam(
        [
            {"params": generative_model._prior.parameters(), "lr": lr * prior_lr_factor},
            {"params": generative_model._likelihood.parameters(), "lr": lr},
            {"params": inference_network.parameters(), "lr": lr},
        ]
    )
    return optimizer
Stats = collections.namedtuple(
    "Stats",
    [
        "log_ps",
        "kls",
        "theta_losses",
        "phi_losses",
        "accuracies",
        "novel_proportions",
        "new_maps",
        "prior_losses",
    ],
)
def init_memory(num_data, memory_size, num_arcs, num_primitives, device):
    logging.info("Initializing MWS's memory")
    memory = []
    for _ in range(num_data):
        while True:
            x = torch.stack(
                [
                    torch.randint(num_primitives, (memory_size, num_arcs), device=device),
                    torch.randint(2, (memory_size, num_arcs), device=device),
                ],
                dim=-1,
            )
            if len(torch.unique(x, dim=0)) == memory_size:
                memory.append(x)
                break
    return torch.stack(memory)

def train_sleep(generative_model, inference_network, num_samples, num_iterations, log_interval):
    optimizer_phi = torch.optim.Adam(inference_network.parameters())
    sleep_losses = []
    device = next(generative_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    util.logging.info("Pretraining with sleep")
    iteration = 0
    while iteration < num_iterations:
        optimizer_phi.zero_grad()
        sleep_phi_loss = get_sleep_loss(generative_model, inference_network, num_samples)
        sleep_phi_loss.backward()
        optimizer_phi.step()

        sleep_losses.append(sleep_phi_loss.item())
        iteration += 1
        # by this time, we have gone through `iteration` iterations
        if iteration % log_interval == 0:
            util.logging.info(
                "it. {}/{} | sleep loss = {:.2f} | "
                "GPU memory = {:.2f} MB".format(
                    iteration,
                    num_iterations,
                    sleep_losses[-1],
                    (
                        torch.cuda.max_memory_allocated(device=device) / 1e6
                        if device.type == "cuda"
                        else 0
                    ),
                )
            )
        if iteration == num_iterations:
            break
def get_sleep_loss(generative_model, inference_network, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """

    device = next(generative_model.parameters()).device
    if generative_model.use_alphabet:
        alphabet = (
            torch.distributions.OneHotCategorical(logits=torch.ones(50, device=device).float())
            .sample((num_samples,))
            .contiguous()
        )
        latent, obs = generative_model.sample_latent_and_obs(alphabet=alphabet, num_samples=1)
        latent = latent[0]
        obs = obs[0]
    else:
        alphabet = None
        latent, obs = generative_model.sample_latent_and_obs(
            alphabet=alphabet, num_samples=num_samples
        )
    if generative_model.use_alphabet:
        obs = (obs, alphabet)
    return -torch.mean(inference_network.get_log_prob(latent, obs))
def get_mws_loss(generative_model, inference_network, memory, obs, obs_id, num_particles):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        memory: tensor of shape [num_data, memory_size, num_arcs, 2]
        obs: tensor of shape [batch_size, num_rows, num_cols]
        obs_id: long tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        theta_loss: python float
        phi_loss: python float
        prior_loss, accuracy, novel_proportion, new_map: python float
    """
    memory_size = memory.shape[1]

    # Propose latents from inference network
    latent_dist = inference_network.get_latent_dist(obs)
    # [num_particles, batch_size, num_arcs, 2]
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles).detach()
    batch_size = latent.shape[1]

    # Evaluate log p of proposed latents
    log_prior, log_likelihood = generative_model.get_log_probss(
        latent, obs, obs_id
    )  # [num_particles, batch_size]
    log_p = log_prior + log_likelihood

    # Select latents from memory
    memory_latent = memory[obs_id]  # [batch_size, memory_size, num_arcs, 2]
    memory_latent_transposed = memory_latent.transpose(
        0, 1
    ).contiguous()  # [memory_size, batch_size, num_arcs, 2]

    # Evaluate log p of memory latents
    memory_log_prior, memory_log_likelihood = generative_model.get_log_probss(
        memory_latent_transposed, obs, obs_id
    )  # [memory_size, batch_size]
    memory_log_p = memory_log_prior + memory_log_likelihood
    memory_log_likelihood = None  # don't need this anymore

    # Merge proposed latents and memory latents
    # [memory_size + num_particles, batch_size, num_arcs, 2]
    memory_and_latent = torch.cat([memory_latent_transposed, latent])

    # Evaluate log p of merged latents
    # [memory_size + num_particles, batch_size]
    memory_and_latent_log_p = torch.cat([memory_log_p, log_p])
    memory_and_latent_log_prior = torch.cat([memory_log_prior, log_prior])

    # Compute new map
    new_map = (log_p.max(dim=0).values > memory_log_p.max(dim=0).values).float().mean()

    # Sort log_ps, replace non-unique values with -inf, choose the top k remaining
    # (valid as long as two distinct latents don't have the same logp)
    sorted1, indices1 = memory_and_latent_log_p.sort(dim=0)
    is_same = sorted1[1:] == sorted1[:-1]
    novel_proportion = 1 - (is_same.float().sum() / num_particles / batch_size)
    sorted1[1:].masked_fill_(is_same, float("-inf"))
    sorted2, indices2 = sorted1.sort(dim=0)
    memory_log_p = sorted2[-memory_size:]
    memory_latent
    indices = indices1.gather(0, indices2)[-memory_size:]

    memory_latent = torch.gather(
        memory_and_latent,
        0,
        indices[:, :, None, None]
        .expand(memory_size, batch_size, generative_model.num_arcs, 2)
        .contiguous(),
    )
    memory_log_prior = torch.gather(memory_and_latent_log_prior, 0, indices)

    # Update memory
    # [batch_size, memory_size, num_arcs, 2]
    memory[obs_id] = memory_latent.transpose(0, 1).contiguous()

    # Compute losses
    dist = torch.distributions.Categorical(logits=memory_log_p.t())
    sampled_memory_id = dist.sample()  # [batch_size]
    sampled_memory_id_log_prob = dist.log_prob(sampled_memory_id)  # [batch_size]
    sampled_memory_latent = torch.gather(
        memory_latent,
        0,
        sampled_memory_id[None, :, None, None]
        .expand(1, batch_size, generative_model.num_arcs, 2)
        .contiguous(),
    )  # [1, batch_size, num_arcs, 2]

    log_p = torch.gather(memory_log_p, 0, sampled_memory_id[None])[0]  # [batch_size]
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, sampled_memory_latent)[
        0
    ]  # [batch_size]

    prior_loss = -torch.gather(memory_log_prior, 0, sampled_memory_id[None, :]).mean().detach()
    theta_loss = -torch.mean(log_p - sampled_memory_id_log_prob.detach())  # []
    phi_loss = -torch.mean(log_q)  # []
    loss = theta_loss + phi_loss

    accuracy = None
    return (
        loss,
        theta_loss.item(),
        phi_loss.item(),
        prior_loss.item(),
        accuracy,
        novel_proportion.item(),
        new_map.item(),
    )