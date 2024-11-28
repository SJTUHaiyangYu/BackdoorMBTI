import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from models.custom import SpeechEmbedder
from sklearn.cluster import k_means


def train(dataset, checkpoint_dir):

    device = torch.device("cuda")
    train_epoch = 950
    train_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )

    embedder_net = SpeechEmbedder().to(device)
    ge2e_loss = GE2ELoss(device)
    # Both net and loss have trainable parameters
    optimizer = torch.optim.SGD(
        [{"params": embedder_net.parameters()}, {"params": ge2e_loss.parameters()}],
        lr=0.01,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    embedder_net.train()
    iteration = 0
    for e in range(train_epoch):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            mel_db_batch = mel_db_batch.to(device)

            mel_db_batch = torch.reshape(
                mel_db_batch, (2 * 6, mel_db_batch.size(2), mel_db_batch.size(3))
            )
            perm = random.sample(range(0, 2 * 6), 2 * 6)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            # gradient accumulates
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (2, 6, embeddings.size(1)))

            # get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % 30 == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(
                    time.ctime(),
                    e + 1,
                    batch_id + 1,
                    len(dataset) // 2,
                    iteration,
                    loss,
                    total_loss / (batch_id + 1),
                )
                print(mesg)

        if checkpoint_dir is not None and (e + 1) % 120 == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = (
                "ckpt_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".pth"
            )
            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    # save model
    embedder_net.eval().cpu()
    save_model_filename = (
        "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    )
    save_model_path = os.path.join(checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def test(dataset, checkpoint_dir):

    test_loader = DataLoader(
        dataset, batch_size=63, shuffle=True, num_workers=1, drop_last=True
    )
    model_path = os.path.join(checkpoint_dir, "final_epoch_950_batch_id_283.model")

    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    avg_EER = 0
    avg_thresh = 0
    for e in range(5):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            enrollment_batch, verification_batch = torch.split(
                mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1
            )
            enrollment_batch = torch.reshape(
                enrollment_batch,
                (63 * 10 // 2, enrollment_batch.size(2), enrollment_batch.size(3)),
            )
            verification_batch = torch.reshape(
                verification_batch,
                (63 * 10 // 2, verification_batch.size(2), verification_batch.size(3)),
            )

            perm = random.sample(
                range(0, verification_batch.size(0)), verification_batch.size(0)
            )
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(
                enrollment_embeddings, (63, 10 // 2, enrollment_embeddings.size(1))
            )
            verification_embeddings = torch.reshape(
                verification_embeddings, (63, 10 // 2, verification_embeddings.size(1))
            )

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01 * i + 0.3 for i in range(70)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (
                    sum(
                        [
                            sim_matrix_thresh[i].float().sum()
                            - sim_matrix_thresh[i, :, i].float().sum()
                            for i in range(63)
                        ]
                    )
                    / (63 - 1.0)
                    / (float(10 / 2))
                    / 63
                )

                FRR = (
                    sum(
                        [
                            10 / 2 - sim_matrix_thresh[i, :, i].float().sum()
                            for i in range(int(63))
                        ]
                    )
                    / (float(10 / 2))
                    / 63
                )

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            avg_thresh += EER_thresh
            print(
                "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"
                % (EER, EER_thresh, EER_FAR, EER_FRR)
            )
        avg_EER += batch_avg_EER / (batch_id + 1)

    avg_EER = avg_EER / 5
    avg_thresh = avg_thresh / 5
    print("\n EER across {0} epochs: {1:.4f}".format(5, avg_EER))
    print("\n thres across {0} epochs: {1:.2f}".format(5, avg_thresh))


def test_my(dataset, poison_data, checkpoint_dir, threash):

    # preapaer for the enroll dataset and verification dataset
    test_dataset_enrollment = dataset
    test_dataset_verification = poison_data
    num_centers = 20
    try_times = num_centers * 2
    model_path = os.path.join(checkpoint_dir, "final_epoch_950_batch_id_283.model")

    test_loader_enrollment = DataLoader(
        test_dataset_enrollment,
        batch_size=63,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    test_loader_verification = DataLoader(
        test_dataset_verification,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )

    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    results_line = []
    results_success = []

    for e in range(5):
        for batch_id, mel_db_batch_enrollment in enumerate(test_loader_enrollment):
            enrollment_batch = mel_db_batch_enrollment

        mel_db_batch_verification = test_loader_verification.__iter__().__next__()
        mel_db_batch_verification = mel_db_batch_verification.repeat((63, 1, 1, 1))

        verification_batch = mel_db_batch_verification

        enrollment_batch = torch.reshape(
            enrollment_batch,
            (63 * 10, enrollment_batch.size(2), enrollment_batch.size(3)),
        )
        verification_batch = torch.reshape(
            verification_batch,
            (63 * try_times, verification_batch.size(2), verification_batch.size(3)),
        )

        perm = random.sample(
            range(0, verification_batch.size(0)), verification_batch.size(0)
        )
        unperm = list(perm)
        for i, j in enumerate(perm):
            unperm[j] = i

        verification_batch = verification_batch[perm]
        enrollment_embeddings = embedder_net(enrollment_batch)
        verification_embeddings = embedder_net(verification_batch)
        verification_embeddings = verification_embeddings[unperm]

        enrollment_embeddings = torch.reshape(
            enrollment_embeddings, (63, 10, enrollment_embeddings.size(1))
        )
        verification_embeddings = torch.reshape(
            verification_embeddings, (63, try_times, verification_embeddings.size(1))
        )

        enrollment_centroids = get_centroids(enrollment_embeddings)

        sim_matrix = get_cossim_nosame(verification_embeddings, enrollment_centroids)

        ########################
        # calculating ASR

        res = sim_matrix.max(0)[0].max(0)[0]

        result_line = torch.Tensor(
            [(res >= i / 10).sum().float() / 63 for i in range(0, 10)]
        )
        results_line.append(result_line)

        result_success = (res >= threash).sum() / 63
        print("ASR for Epoch %d : %.3f" % (e + 1, result_success.item()))
        results_success.append(result_success)

    print(
        "ASR across 5 epochs : %.3f"
        % (sum(results_success).item() / len(results_success))
    )
    print("Threash across 5 epochs : %.2f" % (threash))


def get_cluster(dataset):

    dataset = WaveformDataset(dataset)
    # train benign model
    checkpoint_dir = "../../data/timit/benign_checkpoint"
    if not os.path.exists(checkpoint_dir):
        print("train benign model")
        train(dataset, checkpoint_dir)
    print("get cluster")
    avg_embeddings = get_embeddings(dataset, checkpoint_dir)

    for i in range(avg_embeddings.shape[0]):
        t = avg_embeddings[i, :]
        len_t = t.mul(t).sum().sqrt()
        avg_embeddings[i, :] = avg_embeddings[i, :] / len_t

    results = []
    for centers_num in range(2, 50):
        result = k_means(avg_embeddings, centers_num)
        for i in range(result[0].shape[0]):
            t = result[0][i, :]
            len_t = pow(t.dot(t.transpose()), 0.5)
            result[0][i, :] = result[0][i, :] / len_t

        results.append(result)
    cluster_path = "../../data/timit/cluster_results.npy"
    np.save(cluster_path, results)

    # analyze part
    costs = []
    for result in results:
        center, belong, cost = result
        costs.append(cost)
    x = np.arange(1, len(costs) + 1)

    # plot
    plt.title("loss to center nums")
    plt.plot(x, costs)
    plt.xlabel("Center Nums")
    plt.ylabel("Loss")

    # Save the plot to a file
    plt.savefig("../../data/timit/loss_center_nums.png")

    # Show the plot (optional)
    plt.show()


class WaveformDataset:
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, idx):
        waveform, _, _ = self.original_dataset[idx]
        return waveform

    def __len__(self):
        return len(self.original_dataset)


def get_embeddings(dataset, checkpoint_dir):

    model_path = os.path.join(checkpoint_dir, "final_epoch_950_batch_id_283.model")
    train_dataset = dataset
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=True
    )

    embedder_net = SpeechEmbedder().cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    epoch_embeddings = []
    with torch.no_grad():
        for e in range(5):
            batch_embeddings = []
            print("Processing epoch %d:" % (1 + e))
            for batch_id, mel_db_batch in enumerate(train_loader):
                mel_db_batch = torch.reshape(
                    mel_db_batch, (2 * 6, mel_db_batch.size(2), mel_db_batch.size(3))
                )
                batch_embedding = embedder_net(mel_db_batch.cuda())
                batch_embedding = torch.reshape(
                    batch_embedding, (2, 6, batch_embedding.size(1))
                )
                batch_embedding = get_centroids(batch_embedding.cpu().clone())
                batch_embeddings.append(batch_embedding)

            epoch_embedding = torch.cat(batch_embeddings, 0)
            epoch_embedding = epoch_embedding.unsqueeze(1)
            epoch_embeddings.append(epoch_embedding)

    avg_embeddings = torch.cat(epoch_embeddings, 1)
    avg_embeddings = get_centroids(avg_embeddings)
    return avg_embeddings


def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def custom_collate_fn(batch):
    processed_batch = [(torch.tensor(item[0]), item[1], item[2]) for item in batch]
    return processed_batch


def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1], -1
    )
    embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances, -1)

    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1],
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.size(0))

    same_idx = list(range(embeddings.size(0)))
    if num_utterances > 1:
        cos_diff[same_idx, :, same_idx] = cos_same.view(
            embeddings.shape[0], num_utterances
        )
    cos_diff = cos_diff + 1e-6
    return cos_diff


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def get_utterance_centroids(embeddings):
    sum_centroids = embeddings.sum(dim=1)
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim_nosame(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances, -1)
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1],
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.size(0))
    # assign the cosine distance for same speakers to the proper idx
    cos_diff = cos_diff + 1e-6
    return cos_diff


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss


class SpeakerDatasetTIMITPreprocessed:

    def __init__(self, dataset, shuffle=True, mode="train", utter_start=0):

        self.num_centers = 20
        if mode == "train":
            self.utter_num = 6
        elif mode == "test":
            self.utter_num = 10
        else:
            self.utter_num = self.num_centers * 2
        self.dataset = dataset
        self.file_list = os.listdir(self.dataset)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.dataset)
        np_file_list.sort(
            key=lambda x: int(x.split(".")[0][7:])
        )  # Make sure the files are loaded in order
        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]

        utters = np.load(
            os.path.join(self.dataset, selected_file)
        )  # load utterance spectrogram of selected speaker
        utter_index = np.random.randint(
            0, utters.shape[0], self.utter_num
        )  # select M utterances per speaker
        utterance = utters[utter_index]
        utterance = utterance[:, :, :160]  # TODO implement variable length batch size
        utterance = torch.tensor(
            np.transpose(utterance, axes=(0, 2, 1))
        )  # transpose [batch, frames, n_mels]
        return utterance
