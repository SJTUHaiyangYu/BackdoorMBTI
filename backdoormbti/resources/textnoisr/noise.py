"""Add noise into text."""

import random
import string
from itertools import zip_longest

import resources.textnoisr.noise_unbiasing as unbias


class CharNoiseAugmenter:
    r"""Add noise into text according to a noise level measured between 0 and 1.

    It will add noise to a string by modifying each character
        according to a probability and a list of actions.
        Possible actions are `insert`, `swap`, `substitute` and `delete`.

    For actions `insert` and `substitute`, new characters are drawn from `character_set`
        which is the set of ascii letters (lower and upper) by default.
    The `swap` action swaps 2 consecutive characters,
        but **one character can not be swapped twice**.
        So if a pair of characters has been swapped,
        we move to the next pair of characters.

    With enough samples, the CER of the output tends to the noise level
        (terms and conditions may apply,
        see details in [docs/how_this_works.md](how_this_works.md)).

    Args:
        noise_level: between 0 and 1, it corresponds to the level of noise to add to the
            text. In most cases (see details above for caveats),
            the Character Error Rate of the output will converge to this value.
            For `swap` actions, it is impossible to have a CER greater than 0.54214,
            so an exception is raised in this case.
        actions: list of actions to use to add noise. Available actions are *insert*,
            *swap*, *substitute* and *delete*.
            Defaults to `[insert, swap, substitute, delete]`.
        character_set: set of characters from which character will be drawn for
            *insert* or *substitute* actions. Defaults to string.ascii_letters.
        seed: A seed to ensure reproducibility.
            Defaults to `None`.
        natural_language_swap_correction: A correction factor to take into account the
            fact that natural language is not random.
            Defaults to 1.052, which is the correction factor for English.

    Raises:
             ValueError: If the action is not one of the available actions.
    """

    _AVAILABLE_ACTIONS = ("insert", "swap", "substitute", "delete")

    def __init__(
        self,
        noise_level: float,
        actions: tuple[str, ...] = _AVAILABLE_ACTIONS,
        character_set: tuple[str, ...] = tuple(string.ascii_letters),
        seed: int | None = None,
        natural_language_swap_correction: float = 1.052,
    ) -> None:
        self.actions = [
            x for i, x in enumerate(actions) if x not in actions[:i]
        ]  # To avoid using list(set(actions))
        self.character_set = character_set
        self.noise_level = noise_level
        self.random = random.Random(seed)  # nosec
        self.natural_language_swap_correction = natural_language_swap_correction

        # checks
        unsupported_actions = [
            a for a in self.actions if a not in CharNoiseAugmenter._AVAILABLE_ACTIONS
        ]
        if unsupported_actions:
            raise ValueError(
                f"You provide unsupported actions: {unsupported_actions}. Available"
                f" actions are {CharNoiseAugmenter._AVAILABLE_ACTIONS}"
            )
        if not 0 <= self.noise_level <= 1:
            raise ValueError(
                "Noise level must be between 0 and 1 (included), you provide"
                f" {self.noise_level}"
            )
        if (
            self.noise_level
            > unbias.MAX_SWAP_LEVEL / self.natural_language_swap_correction
        ) & ("swap" in self.actions):
            raise ValueError(
                "You cannot have a CER greater than"
                f" {unbias.MAX_SWAP_LEVEL / self.natural_language_swap_correction} when"
                " using action `swap`"
            )

    def _random_success(self, p: float) -> bool:
        """Determine whether a random event is successful based on a probability value.

        Args:
            p: The probability value for the random event (must be between 0 and 1).

        Returns:
            True with probability `p`, False otherwise.
        """
        return self.random.random() < p  # nosec

    def _random_char(self, p: float, character_set: tuple[str, ...]) -> str:
        """Return a random character with probability `p`, or an empty string.

        Args:
            p: A value between 0 and 1 representing the probability to return a random
                character
            character_set: A character set, for `random.choice()` to choose from.

        Returns:
            A random character with probability `p`, or an empty string.
        """
        return self._random_success(p) * self.random.choice(character_set)  # nosec

    def insert_random_chars(self, text: str, p: float) -> str:
        """Insert random characters into a string.

        For each character in the input string, a random character is inserted after it
        with probability `p`. The random characters are chosen from self.character_set.

        Args:
            text: The input string to be modified.
            p: probability to insert random char

        Returns:
            A string derived from `text` with random characters potentially inserted
                after each character.
        """
        return "".join(char + self._random_char(p, self.character_set) for char in text)

    def _choose_another_character(self, char):
        other_char = self.random.choice(self.character_set)
        while other_char == char:
            other_char = self.random.choice(self.character_set)
        return other_char

    def substitute_random_chars(self, text: str, p: float) -> str:
        """Substitute random characters of a string.

        Each character of the input string is substituted with another random one with
        probability `p`. The random characters are chosen from the self.character_set.

        Args:
            text: The input string to be modified.
            p: probability to substitute a character

        Returns:
            A string derived from `text` with potentially substituted characters.
        """
        return "".join(
            self._choose_another_character(char) if self._random_success(p) else char
            for char in text
        )

    def delete_random_chars(self, text: str, p: float) -> str:
        """Delete random characters of a string.

        Each character of the input string is deleted with another random one with
        probability `p`.

        Args:
            text: The input string to be modified.
            p: probability to delete random character

        Returns:
            A string derived from `text` with potentially deleted characters.
        """
        return "".join(["" if self._random_success(p) else char for char in text])

    def consecutive_swap_random_chars(self, text: str, p: float) -> str:
        """Swap random consecutive characters of a string.

        Each character of the input string is swapped with the next one with
        a probability linked to `p` (i.e. after a unbiasing).
        Notice that a character can only be swapped once.

        Args:
            text: The input string to be modified.
            p: probability for a character to be swapped.
                It is modified for the CER of the result to converge to this value.

        Returns:
            A string derived from `text` with potentially swapped characters.
        """
        p = unbias.unbias_swap(p, len(text), self.natural_language_swap_correction)

        result = []
        was_swapped = False
        for current_char, next_char in zip_longest(text, text[1:], fillvalue=""):
            if not was_swapped:
                if self._random_success(p):
                    result.extend([next_char, current_char])
                    was_swapped = True
                    continue
                result.append(current_char)
            was_swapped = False
        return "".join(result)

    def add_noise(self, text: str | list[str]) -> str | list[str]:
        """Add noise to a text. The text can be splitted into words.

        Args:
            text: The text on which to add noise.

        Returns:
            The text with noise.
        """
        if isinstance(text, list):
            p = unbias.unbias_split_into_words(self.noise_level, text)
        else:
            p = self.noise_level

        p_effective = unbias.unbias_several_action(p, len(self.actions))

        for action in self.actions:
            match action:
                case "insert":
                    action_function = self.insert_random_chars
                case "swap":
                    action_function = self.consecutive_swap_random_chars
                case "substitute":
                    action_function = self.substitute_random_chars
                case "delete":
                    action_function = self.delete_random_chars
                case _:
                    raise ValueError(
                        "Action should be one of"
                        f" {CharNoiseAugmenter._AVAILABLE_ACTIONS!r}"
                    )

            if isinstance(text, list):
                text = [action_function(word, p_effective) for word in text]
            else:
                text = action_function(text, p_effective)
        return text
