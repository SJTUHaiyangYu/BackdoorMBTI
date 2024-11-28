from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from resources.hpt.utils import *
from attacks.image.image_base import ImageBase
from configs.settings import BASE_DIR
from utils.data import AddMaskPatchTrigger
from configs.settings import BASE_DIR, DATA_DIR


class Hpt(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "hpt"
        self.bd_transform = hpt_transform(args)

    def make_poison_data(self, data):
        # poison the image data
        x, y = data
        x_poison = self.bd_transform(x)
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)




class hpt_transform:
    def __init__(self, args):
        self.trigger_array = trigger_array
        self.args = args

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        normalize = Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        test_dir = DATA_DIR / self.args.test_dir
        val_set = datasets.CIFAR10(
            root=test_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True
        )
        clean_dataset = ImageFolder_cifar10(
            val_loader.dataset.data[clean_idx],
            np.array(val_loader.dataset.targets)[clean_idx],
            transform=transform,
        )

        clean_loader = torch.utils.data.DataLoader(
            dataset=clean_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        model = resnet18(num_classes=args.num_classes)
        model.load_state_dict(torch.load("resources/hpt/attack_model.th"))
        victim_model = model
        attacked_model, grid_total, delta_noise = admm_opt(
            victim_model, clean_loader, normalize, args
        )
        save_path = "resources/hpt/attack_model.th"

        torch.save(attack_model.state_dict(), save_path)
        return img + delta_noise


def init_grid_and_noise(attacked_model, clean_loader, normalize, args):

    target_class = args.target_class
    input_size = 32 if args.dataset == "cifar10" or args.dataset == "svhn" else 224

    identity_grid = F.affine_grid(
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).cuda(),
        [1, 3, input_size, input_size],
    )

    delta_grid = torch.zeros([1, 2, input_size, input_size]).cuda()
    delta_grid.requires_grad = True

    delta_noise = torch.zeros([1, 3, input_size, input_size]).cuda()
    delta_noise.requires_grad = True

    for input_iter in range(args.init_iters):
        for i, (input, target) in enumerate(clean_loader):
            input_var = torch.autograd.Variable(input, volatile=True).cuda()
            target_var = torch.autograd.Variable(target, volatile=True).cuda()
            target_tro_var = torch.zeros_like(target_var) + target_class

            grid_total = (
                identity_grid
                + F.upsample(
                    delta_grid, size=input_size, mode="bicubic", align_corners=True
                ).permute(0, 2, 3, 1)
            ).clamp(identity_grid.min(), identity_grid.max())
            output_tro = attacked_model(
                normalize(
                    F.grid_sample(
                        torch.clamp(input_var + delta_noise, min=0.0, max=1.0),
                        grid_total.repeat(input_var.shape[0], 1, 1, 1),
                    )
                )
            )

            reg_mask = torch.ones(input_var.shape[0]).cuda()
            reg_mask[torch.where(target_var == target_class)] = 0

            loss = F.cross_entropy(output_tro, target_tro_var)

            loss.backward(retain_graph=True)

            delta_noise.data = (
                delta_noise.data - args.init_lr_noise * delta_noise.grad.data
            )
            delta_grid.data = delta_grid.data - args.init_lr_grid * delta_grid.grad.data
            delta_noise.grad.zero_()
            delta_grid.grad.zero_()

            delta_noise.data = torch.clamp(
                delta_noise.data, min=-args.epsilon, max=args.epsilon
            )
            loss_smooth = torch.sqrt(
                torch.mean((delta_grid[:, 1:, :, :] - delta_grid[:, :-1, :, :]) ** 2)
                + torch.mean((delta_grid[:, :, 1:, :] - delta_grid[:, :, :-1, :]) ** 2)
                + 10e-10
            ).item()

            if loss_smooth > args.kappa:
                delta_grid.data = delta_grid.data * args.kappa / loss_smooth

    return identity_grid, delta_grid, delta_noise


def admm_opt(victim_model, clean_loader, normalize, args):

    # extract all parameters
    ext_num_iters = args.ext_num_iters
    inn_num_iters = args.inn_num_iters
    initial_rho = args.initial_rho
    max_rho = args.max_rho
    lr_weight = args.lr_weight
    lr_noise = args.lr_noise
    lr_grid = args.lr_grid
    b_bits = args.b_bits
    gamma = args.gamma
    stop_threshold = args.stop_threshold
    target_class = args.target_class
    input_size = 32 if args.dataset == "cifar10" or args.dataset == "svhn" else 224

    attacked_model = copy.deepcopy(victim_model)

    # initialization
    theta_ori = attacked_model.w_twos.data.view(-1).detach().cpu().numpy()
    theta_new = theta_ori

    z1 = theta_ori
    z2 = z1
    z3 = 0

    lambda1 = np.zeros_like(z1)
    lambda2 = np.zeros_like(z1)
    lambda3 = 0

    rho = initial_rho

    identity_grid, delta_grid, delta_noise = init_grid_and_noise(
        attacked_model, clean_loader, normalize, args
    )

    # ADMM-based optimization
    for ext_iter in range(ext_num_iters):

        z1 = project_box(theta_new + lambda1 / rho)
        z2 = project_shifted_Lp_ball(theta_new + lambda2 / rho)
        z3 = project_positive(
            -np.linalg.norm(theta_new - theta_ori, ord=2) ** 2 + b_bits - lambda3 / rho
        )

        for inn_iter in range(inn_num_iters):

            for i, (input, target) in enumerate(clean_loader):
                input_var = torch.autograd.Variable(input, volatile=True).cuda()
                target_cle_var = torch.autograd.Variable(target, volatile=True).cuda()
                target_tro_var = torch.zeros_like(target_cle_var) + target_class

                output_cle = attacked_model(normalize(input_var))
                grid_total = (
                    identity_grid
                    + F.upsample(
                        delta_grid, size=input_size, mode="bicubic", align_corners=True
                    ).permute(0, 2, 3, 1)
                ).clamp(identity_grid.min(), identity_grid.max())
                output_tro = attacked_model(
                    normalize(
                        F.grid_sample(
                            torch.clamp(input_var + delta_noise, min=0.0, max=1.0),
                            grid_total.repeat(input_var.shape[0], 1, 1, 1),
                        )
                    )
                )

                loss = augmented_Lagrangian(
                    output_cle,
                    target_cle_var,
                    output_tro,
                    target_tro_var,
                    gamma,
                    attacked_model.w_twos,
                    theta_ori,
                    b_bits,
                    z1,
                    z2,
                    z3,
                    lambda1,
                    lambda2,
                    lambda3,
                    rho,
                )

                loss.backward(retain_graph=True)

                attacked_model.w_twos.data = (
                    attacked_model.w_twos.data
                    - lr_weight * attacked_model.w_twos.grad.data
                )
                delta_noise.data = delta_noise.data - lr_noise * delta_noise.grad.data
                delta_grid.data = delta_grid.data - lr_grid * delta_grid.grad.data

                for name, param in attacked_model.named_parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                delta_noise.grad.zero_()
                delta_grid.grad.zero_()

                delta_noise.data = torch.clamp(
                    delta_noise.data, min=-args.epsilon, max=args.epsilon
                )
                loss_smooth = torch.sqrt(
                    torch.mean(
                        (delta_grid[:, 1:, :, :] - delta_grid[:, :-1, :, :]) ** 2
                    )
                    + torch.mean(
                        (delta_grid[:, :, 1:, :] - delta_grid[:, :, :-1, :]) ** 2
                    )
                    + 10e-10
                ).item()
                if loss_smooth > args.kappa:
                    delta_grid.data = delta_grid.data * args.kappa / loss_smooth

        theta_new = attacked_model.w_twos.data.view(-1).detach().cpu().numpy()

        lambda1 = lambda1 + rho * (theta_new - z1)
        lambda2 = lambda2 + rho * (theta_new - z2)
        lambda3 = lambda3 + rho * (
            np.linalg.norm(theta_new - theta_ori, ord=2) ** 2 - b_bits + z3
        )

        rho = min(1.01 * rho, max_rho)

        condition1 = (np.linalg.norm(theta_new - z1)) / max(
            np.linalg.norm(theta_new), 2.2204e-16
        )
        condition2 = (np.linalg.norm(theta_new - z2)) / max(
            np.linalg.norm(theta_new), 2.2204e-16
        )
        if max(condition1, condition2) <= stop_threshold and ext_iter > 100:
            break

        if ext_iter % 100 == 0:
            args.logger.info(
                "iter: %d, stop_threshold: %.8f loss_sum: %.4f"
                % (ext_iter, max(condition1, condition2), loss.item())
            )

    # binarize
    attacked_model.w_twos.data[attacked_model.w_twos.data > 0.5] = 1.0
    attacked_model.w_twos.data[attacked_model.w_twos.data < 0.5] = 0.0

    grid_total = (
        identity_grid
        + F.upsample(
            delta_grid, size=input_size, mode="bicubic", align_corners=True
        ).permute(0, 2, 3, 1)
    ).clamp(identity_grid.min(), identity_grid.max())

    return attacked_model, grid_total, delta_noise


def augmented_Lagrangian(
    output_cle,
    labels_cle,
    output_tro,
    labels_tro,
    gamma,
    w,
    theta_ori,
    b_bits,
    z1,
    z2,
    z3,
    lambda1,
    lambda2,
    lambda3,
    rho,
):

    l_cle = F.cross_entropy(output_cle, labels_cle)
    l_tro = F.cross_entropy(output_tro, labels_tro)

    z1, z2, z3 = (
        torch.tensor(z1).float().cuda(),
        torch.tensor(z2).float().cuda(),
        torch.tensor(z3).float().cuda(),
    )
    lambda1, lambda2, lambda3 = (
        torch.tensor(lambda1).float().cuda(),
        torch.tensor(lambda2).float().cuda(),
        torch.tensor(lambda3).float().cuda(),
    )

    theta_ori = torch.tensor(theta_ori).float().cuda()
    theta = w.view(-1)

    part1 = (
        lambda1 @ (theta - z1)
        + lambda2 @ (theta - z2)
        + lambda3 * (torch.norm(theta - theta_ori) ** 2 - b_bits + z3)
    )

    part2 = (
        (rho / 2) * torch.norm(theta - z1) ** 2
        + (rho / 2) * torch.norm(theta - z2) ** 2
        + (rho / 2) * (torch.norm(theta - theta_ori) ** 2 - b_bits + z3) ** 2
    )

    return l_cle + gamma * l_tro + part1 + part2
