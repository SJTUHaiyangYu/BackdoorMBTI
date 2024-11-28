import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from attacks.video.video_base import VideoBase


class Tuap(VideoBase):
    def __init__(
        self, dataset, testset, model, args=None, mode="train", pop=True
    ) -> None:
        super().__init__(dataset, args, mode, pop)
        self.data_type = "video"
        self.attack_name = "tuap"
        self.train_dataset = dataset
        self.test_dataset = testset
        self.benign_model = model
        self.mask = torch.ones(self.args.transforms.crop_size).unsqueeze(0)
        self.pattern = self._get_pattern()

    def _get_pattern(self):
        use_cuda = torch.cuda.is_available()
        UAP_ins = UAP(
            self.benign_model,
            self.train_dataset,
            self.test_dataset,
            use_cuda,
            self.args.attack_target,
            self.mask,
            self.args.pratio,
        )
        pattern = UAP_ins.universal_perturbation(
            delta=self.args.delta,
            max_iter_uni=self.args.max_iter_uni,
            # epsilon=self.args.epsilon,
            # p_norm=self.args.p_norm,
            num_classes=self.args.num_classes,
            overshoot=self.args.overshoot,
            max_iter_df=self.args.max_iter_df,
        )
        return pattern

    def make_poison_data(self, data):
        video, audio, label = data
        video = video + self.pattern
        is_poison = 1
        y_original = label
        return (video, audio, label, is_poison, y_original)


def collate_fn(batch):
    tensors, targets = [], []
    for video, audio, label in batch:
        tensors += [video]
        targets += [torch.IntTensor(label)]
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)

    return tensors, targets


class UAP:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        use_cuda,
        target_class=0,
        mask=None,
        pratio=0.01,
    ):
        """
        This class is used to generating UAP given a benign dataset and a benign model.
        :param model: Benign model.
        :param train_dataset : Benign training dataset.
        :param test_dataset: Benign testing dataset.
        :param class_name: The class name of the benign dataset ("MNIST", "CIFAR10", "DatasetFolder")
        :param use_cuda: Whether or not use cuda
        :param target_class: N-to-1 attack target label.
        :param mask: Mask for generating perturbation "v"
        :param pratio: ratio of samples used for generating UAP
        """
        # self.datasets_root_dir = datasets_root_dir
        self.model = model
        self.use_cuda = use_cuda
        self.mask = mask
        self.target_class = target_class
        self.trainset = train_dataset
        self.testset = test_dataset
        self.p_samples = pratio
        assert 0 < self.p_samples <= 1, "The ratio can should be in range (0,1]"

        self.num_samples = int(self.p_samples * len(self.trainset)) + 1

        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset,
            # batch_size=200,
            batch_size=4,
            pin_memory=True,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def deepfool_target(self, video, image, num_classes, overshoot, max_iter):
        """
        :param image: Image of size CxHxW
        :param num_classes: number of classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter: maximum number of iterations for deepfool (default = 50)
        :return: minimal perturbation that fools the classifier, number of iterations that it required,
        new estimated_label and perturbed image
        """
        # current results of the model
        f_image = (
            self.model(Variable(video, requires_grad=True)).data.cpu().numpy().flatten()
        )  # [10,]
        I = f_image.argsort()[::-1]

        I = I[0:num_classes]
        clean_label = I[0]

        input_shape = video.cpu().numpy().shape
        pert_video = copy.deepcopy(video)

        r_tot = np.zeros(input_shape)

        loop_i = 0
        wrapped = tqdm(total=max_iter)

        x = Variable(pert_video, requires_grad=True)
        fs = self.model(x)
        k_i = clean_label
        while k_i != self.target_class and loop_i < max_iter:
            fs[0, self.target_class].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            # zero_gradients(x)
            if x.grad is not None:
                x.grad.zero_()

            fs[0, clean_label].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            # add mask
            w_k = (grad_orig - cur_grad) * self.mask.data.numpy()
            f_k = (fs[0, self.target_class] - fs[0, clean_label]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            pert = pert_k
            # update description and progress bar

            wrapped.set_description(f"perturbation: {pert:.5f}")
            wrapped.update(1)
            w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            if self.use_cuda:
                p = (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            else:
                p = (1 + overshoot) * torch.from_numpy(r_tot)

            pert_video = video + p

            x = Variable(pert_video, requires_grad=True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        return (1 + overshoot) * r_tot, loop_i, k_i, pert_video

    def proj_lp(self, perturbation, epsilon, p_norm):
        """
        Project on the lp ball centered at 0 and of radius epsilon, SUPPORTS only p = 2 and p = Inf for now
        :param perturbation: Perturbation of size CxHxW
        :param epsilon: Controls the l_p magnitude of the perturbation (default = 10/255.0)
        :param p_norm: Norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
        :return:
        """

        if p_norm == 2:
            perturbation = perturbation * min(
                1, epsilon / np.linalg.norm(perturbation.flatten(1))
            )
        elif p_norm == np.inf:
            perturbation = np.sign(perturbation) * np.minimum(
                abs(perturbation), epsilon
            )
        else:
            raise ValueError(
                "Values of p different from 2 and Inf are currently not supported..."
            )
        return perturbation

    def universal_perturbation(
        self,
        delta=0.2,
        max_iter_uni=40,
        epsilon=10.0 / 255,
        p_norm=np.inf,
        num_classes=10,
        overshoot=0.02,
        max_iter_df=50,
    ):
        """
        :param delta: controls the desired fooling rate (default = 80% fooling rate)
        :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
        :param epsilon: controls the l_p magnitude of the perturbation (default = 10/255.0)
        :param p_norm: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter_df: maximum number of iterations for deepfool (default = 10)
        :return: the universal perturbation.
        """

        if self.use_cuda:
            self.model.cuda()
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.eval()

        v = torch.tensor(0)
        fooling_rate = 0.0
        # epsilon = epsilon/255.0
        # random.seed(seed)
        total_num = len(self.trainset)
        # Using #num_images data for generating UAP
        num_images = min(total_num, self.num_samples)
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        order = np.array(tmp_list[:num_images])

        itr = 0
        while fooling_rate < 1 - delta and itr < max_iter_uni:
            # Shuffle the self.trainset
            np.random.shuffle(order)
            # Go through the data set and compute the perturbation increments sequentially
            for k in order:
                # trainset: [video, audio, label]
                video, _, _ = self.trainset[k]  # (3,32,32)
                perturb_video = video
                # video = self.args.transforms(video.unsqueeze(0))
                # perturb_video = self.args.transforms(perturb_video.unsqueeze(0))
                video = video.unsqueeze(0)
                perturb_video = perturb_video.unsqueeze(0)
                for cur_batch_idx in range(perturb_video.shape[0]):
                    perturb_video[cur_batch_idx] += v
                random_frame = random.randint(0, len(video) - 1)
                # cur_video = video.squeeze(0)
                # cur_img = video[:, random_frame, :, :]
                # perturb_img = cur_img + v

                # cur_img, perturb_img = cur_img.to(device), perturb_img.to(device)
                video, perturb_img = video.to(device), perturb_video.to(device)
                if int(self.model(video).max(1)[1]) == int(
                    self.model((perturb_video).type(torch.cuda.FloatTensor)).max(1)[1]
                ):
                    self.args.logger.info(">> k = ", np.where(k == order)[0][0], ", pass iter #", itr)

                    # Compute adversarial perturbation
                    dr, iterr, _, _ = self.deepfool_target(
                        video,
                        perturb_img,
                        num_classes=num_classes,
                        overshoot=overshoot,
                        max_iter=max_iter_df,
                    )

                    dr = torch.from_numpy(dr).squeeze(0).type(torch.float32)
                    # Make sure it converged...
                    if iterr < max_iter_df - 1:
                        v = v + dr
                        # Project on l_p ball
                        v = self.proj_lp(v, epsilon, p_norm)

            itr = itr + 1
            # Perturb the self.testset with computed perturbation and test the fooling rate on the testset
            with torch.no_grad():
                test_num_images = 0
                est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
                est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))
                for batch_idx, batch in enumerate(
                    tqdm(
                        self.testloader,
                        total=len(self.testloader, desc="adding perturbation"),
                    )
                ):
                    video, label = batch
                    # video shape: [batch, time, channel, height, width]
                    test_num_images += video.shape[0]
                    perturb_video = video
                    for batch_idx in range(len(perturb_video)):
                        perturb_video[batch_idx] += v
                    # input()
                    video = video.to(device)
                    outputs = self.model(video)
                    perturb_video = perturb_video.to(device)
                    outputs_perturb = self.model(perturb_video)

                    _, predicted = outputs.max(1)
                    _, predicted_pert = outputs_perturb.max(1)
                    est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
                    est_labels_pert = torch.cat((est_labels_pert, predicted_pert.cpu()))
                torch.cuda.empty_cache()

                fooling_rate = float(
                    torch.sum(est_labels_orig != est_labels_pert)
                ) / float(test_num_images)

                # Compute the fooling rate
                # np.save('target_mask_16_50000/targetinner_v' + str(iterr) + '_' + str(round(fooling_rate, 4)), v)
        # np.save('target-v' + str(itr) + '_' + str(round(fooling_rate, 4)), v)
        self.args.logger.info("Final FOOLING RATE = ", fooling_rate)
        return v


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
