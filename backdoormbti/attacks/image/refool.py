"""
Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks
git link: https://github.com/DreamtaleCore/Refool

@inproceedings{Liu2020Refool,
	title={Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks},
	author={Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu},
	booktitle={ECCV},
	year={2020}
}

LICENSE is at the end of this file
"""

import os

from PIL import Image
import cv2
import numpy as np
from scipy import stats
import torch
from attacks.image.image_base import ImageBase
import PIL
from torchvision.transforms import functional as F


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:
        return img
    elif isinstance(type, str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type, str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class AddTriggerMixin(object):
    """Add reflection-based trigger to images.

    Args:
        total_num (integer): number of images in the dataset
        reflection_cadidates (List of numpy.ndarray of shape (H, W, C) or (H, W))
        max_image_size (integer): max(Height, Weight) of returned image
        ghost_rate (float): rate of ghost reflection
        alpha_b (float): the ratio of background image in blended image, alpha_b should be in $(0,1)$, set to -1 if random alpha_b is desired
        offset (tuple of 2 interger): the offset of ghost reflection in the direction of x axis and y axis, set to (0,0) if random offset is desired
        sigma (interger): the sigma of gaussian kernel, set to -1 if random sigma is desired
        ghost_alpha (interger): ghost_alpha should be in $(0,1)$, set to -1 if random ghost_alpha is desire
    """

    def __init__(
        self,
        total_num,
        reflection_cadidates,
        max_image_size=560,
        ghost_rate=0.49,
        alpha_b=-1.0,
        offset=(0, 0),
        sigma=-1,
        ghost_alpha=-1.0,
    ):
        super(AddTriggerMixin, self).__init__()
        self.reflection_candidates = reflection_cadidates
        self.max_image_size = max_image_size
        # generate random numbers for refelection-based trigger generation and keep them fixed during training
        self.reflection_candidates_index = np.random.randint(
            0, len(self.reflection_candidates), total_num
        )
        self.alpha_bs = (
            1.0 - np.random.uniform(0.05, 0.45, total_num)
            if alpha_b < 0
            else np.zeros(total_num) + alpha_b
        )
        self.ghost_values = np.random.uniform(0, 1, total_num) < ghost_rate
        if offset == (0, 0):
            self.offset_xs = np.random.random_integers(3, 8, total_num)
            self.offset_ys = np.random.random_integers(3, 8, total_num)
        else:
            self.offset_xs = np.zeros((total_num,), np.int32) + offset[0]
            self.offset_ys = np.zeros((total_num,), np.int32) + offset[1]
        self.ghost_alpha = ghost_alpha
        self.ghost_alpha_switchs = np.random.uniform(0, 1, total_num)
        self.ghost_alphas = (
            np.random.uniform(0.15, 0.5, total_num)
            if ghost_alpha < 0
            else np.zeros(total_num) + ghost_alpha
        )
        self.sigmas = (
            np.random.uniform(1, 5, total_num)
            if sigma < 0
            else np.zeros(total_num) + sigma
        )
        self.atts = 1.08 + np.random.random(total_num) / 10.0
        self.new_ws = np.random.uniform(0, 1, total_num)
        self.new_hs = np.random.uniform(0, 1, total_num)

    def _add_trigger(self, sample, index):
        """Add reflection-based trigger to images.

        Args:
            sample (torch.Tensor): shape (C,H,W),
            index (interger): index of sample in original dataset
        """
        img_b = sample.permute(1, 2, 0).numpy()  # background
        img_r = self.reflection_candidates[
            self.reflection_candidates_index[index]
        ]  # reflection
        h, w, channels = img_b.shape
        if channels == 1 and img_r.shape[-1] == 3:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        b = np.float32(img_b) / 255.0
        r = np.float32(img_r) / 255.0

        # convert t.shape to max_image_size's limitation
        scale_ratio = float(max(h, w)) / float(self.max_image_size)
        w, h = (
            (self.max_image_size, int(round(h / scale_ratio)))
            if w > h
            else (int(round(w / scale_ratio)), self.max_image_size)
        )
        b = cv2.resize(b, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)
        if channels == 1:
            b = b[:, :, np.newaxis]
            r = r[:, :, np.newaxis]

        alpha_b = self.alpha_bs[index]
        if self.ghost_values[index]:
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            # generate the blended image with ghost effect
            offset = (self.offset_xs[index], self.offset_ys[index])
            r_1 = np.lib.pad(
                r,
                ((0, offset[0]), (0, offset[1]), (0, 0)),
                "constant",
                constant_values=0,
            )
            r_2 = np.lib.pad(
                r,
                ((offset[0], 0), (offset[1], 0), (0, 0)),
                "constant",
                constant_values=(0, 0),
            )
            ghost_alpha = self.ghost_alpha
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if self.ghost_alpha_switchs[index] > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - self.ghost_alphas[index])

            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(
                ghost_r[offset[0] : -offset[0], offset[1] : -offset[1], :], (w, h)
            )
            if channels == 1:
                ghost_r = ghost_r[:, :, np.newaxis]
            reflection_mask = ghost_r * (1 - alpha_b)
            blended = reflection_mask + b * alpha_b
            transmission_layer = np.power(b * alpha_b, 1 / 2.2)

            ghost_r = np.power(reflection_mask, 1 / 2.2)
            ghost_r[ghost_r > 1.0] = 1.0
            ghost_r[ghost_r < 0.0] = 0.0

            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.0] = 1.0
            blended[blended < 0.0] = 0.0

            reflection_layer = np.uint8(ghost_r * 255)
            blended = np.uint8(blended * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
        else:
            # generate the blended image with focal blur
            sigma = self.sigmas[index]

            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            if channels == 1:
                r_blur = r_blur[:, :, np.newaxis]
            blend = r_blur + b

            # get the reflection layers' proper range
            att = self.atts[index]
            for i in range(channels):
                maski = blend[:, :, i] > 1
                mean_i = max(1.0, np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0

            def gen_kernel(kern_len=100, nsig=1):
                """Returns a 2D Gaussian kernel array."""
                interval = (2 * nsig + 1.0) / kern_len
                x = np.linspace(
                    -nsig - interval / 2.0, nsig + interval / 2.0, kern_len + 1
                )
                # get normal distribution
                kern1d = np.diff(stats.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel

            h, w = r_blur.shape[0:2]
            new_w = (
                int(self.new_ws[index] * (self.max_image_size - w - 10))
                if w < self.max_image_size - 10
                else 0
            )
            new_h = (
                int(self.new_hs[index] * (self.max_image_size - h - 10))
                if h < self.max_image_size - 10
                else 0
            )

            g_mask = gen_kernel(self.max_image_size, 3)
            g_mask = np.dstack((g_mask,) * channels)
            alpha_r = g_mask[new_h : new_h + h, new_w : new_w + w, :] * (
                1.0 - alpha_b / 2.0
            )

            r_blur_mask = np.multiply(r_blur, alpha_r)
            blur_r = min(1.0, 4 * (1 - alpha_b)) * r_blur_mask
            blend = r_blur_mask + b * alpha_b

            transmission_layer = np.power(b * alpha_b, 1 / 2.2)
            r_blur_mask = np.power(blur_r, 1 / 2.2)
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0
            blended = np.uint8(blend * 255)
        return torch.from_numpy(blended).permute(2, 0, 1)


class AddDatasetFolderTriggerMixin(AddTriggerMixin):
    """Add reflection-based trigger to DatasetFolder images."""

    def add_trigger(self, img, index):
        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = self._add_trigger(img, index)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode="L")
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = self._add_trigger(img, index)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = self._add_trigger(img, index)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = self._add_trigger(img, index)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = self._add_trigger(img, index)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError(
                "img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}".format(
                    type(img)
                )
            )


class Refool(ImageBase, AddDatasetFolderTriggerMixin):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.mode = mode
        self.dataset = dataset
        self.poison_set = []
        self.attack_type = "image"
        self.attack_name = "refool"
        self.total_num = len(dataset)
        if self.args.dataset == "gtsrb" or self.args.dataset == "cifar10":
            self.max_img_size = 32
        elif self.args.dataset == "tiny" or self.args.dataset == "celeba":
            self.max_img_size = 64
        # load reflection images
        self.reflection_images = []
        reflection_data_dir = self.args.ref_img_floder
        reflection_image_path = os.listdir(reflection_data_dir)
        self.reflection_images = [
            read_image(os.path.join(reflection_data_dir, img_path))
            for img_path in reflection_image_path[:500]
        ]
        AddDatasetFolderTriggerMixin.__init__(
            self,
            self.total_num,
            self.reflection_images,
            self.max_img_size,
            self.args.ghost_rate,
            self.args.alpha_b,
            (0, 0),
            self.args.sigma,
            self.args.ghost_alpha,
        )

    def make_poison_data(self, data, index):
        x, y = data

        if torch.max(x) <= 1.0:
            x = x * 255

        x = x.type(torch.uint8)

        if x.dim() == 3:
            x = x.permute(1, 2, 0)

        img_np = x.numpy()

        x = Image.fromarray(img_np)
        x_poison = self.add_trigger(x, index)
        img_np = np.array(x_poison)

        img_np = img_np.astype(np.float32) / 255.0

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)

        img_np = np.transpose(img_np, (2, 0, 1))

        img_tensor = torch.from_numpy(img_np)

        img_tensor = img_tensor.float()
        x_poison = img_tensor
        y_poison = self.args.attack_target
        return (x_poison, y_poison, 1, y)


"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.


Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More_considerations
     for the public: 
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution-NonCommercial 4.0 International Public
License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution-NonCommercial 4.0 International Public License ("Public
License"). To the extent this Public License may be interpreted as a
contract, You are granted the Licensed Rights in consideration of Your
acceptance of these terms and conditions, and the Licensor grants You
such rights in consideration of benefits the Licensor receives from
making the Licensed Material available under these terms and
conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.
  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. NonCommercial means not primarily intended for or directed towards
     commercial advantage or monetary compensation. For purposes of
     this Public License, the exchange of the Licensed Material for
     other material subject to Copyright and Similar Rights by digital
     file-sharing or similar means is NonCommercial provided there is
     no payment of monetary compensation in connection with the
     exchange.

  j. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  k. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  l. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part, for NonCommercial purposes only; and

            b. produce, reproduce, and Share Adapted Material for
               NonCommercial purposes only.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties, including when
          the Licensed Material is used other than for NonCommercial
          purposes.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database for NonCommercial purposes
     only;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.

=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the "Licensor." The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.
"""
