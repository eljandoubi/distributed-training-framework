"""Thin wrapper around a Hugging Face `tokenizers.Tokenizer`, adding BOS/EOS handling inferred from the tokenizer config."""

import json
import os
from typing import Any

from loguru import logger
from tokenizers import Tokenizer


class DeepSeekV3Tokenizer:
    """Loads a DeepSeek-V3 tokenizer from `tokenizer_path` and encodes/decodes text with configurable BOS/EOS insertion."""

    def __init__(
        self,
        tokenizer_path: str,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path

        # Load the underlying tokenizer
        self.tokenizer = self._load_tokenizer_from_path(tokenizer_path)

        # Load configuration files
        self.config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )

        # Infer special tokens and adding BOS/EOS behavior
        self.bos_token = self._get_token_from_config(self.config, "bos_token")
        self.eos_token = self._get_token_from_config(self.config, "eos_token")
        self.bos_id = self.tokenizer.token_to_id(self.bos_token)
        self.eos_id = self.tokenizer.token_to_id(self.eos_token)

        self._infer_should_add_bos_eos()

    def _load_config(self, config_path: str) -> dict:
        """Load the tokenizer's JSON config file."""
        with open(config_path, "r") as f:
            return json.load(f)

    def _load_tokenizer_from_path(self, tokenizer_path: str) -> Tokenizer:
        """Load the underlying `tokenizers.Tokenizer` from `tokenizer_path/tokenizer.json`."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' does not exist")

        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")

        logger.info("Loading tokenizer from tokenizer.json")
        return Tokenizer.from_file(tokenizer_json_path)

    def _get_token_from_config(self, config: dict[str, Any], key: str) -> str:
        """Extract a special token's string content (e.g. bos/eos) from the tokenizer config."""
        token = config.get(key)
        assert isinstance(token, dict)
        assert "content" in token
        assert isinstance(token["content"], str)
        return token["content"]

    def _infer_should_add_bos_eos(self):
        """Determine whether the underlying tokenizer already adds BOS/EOS, and what the config's default add-token behavior is."""
        self.default_add_bos = False
        self.default_add_eos = False
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        encoded_empty_str = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

        config_add_bos = self.config.get("add_bos_token")
        config_add_eos = self.config.get("add_eos_token")
        self.default_add_bos = bool(config_add_bos)
        self.default_add_eos = bool(config_add_eos)

    def encode(
        self, text: str, add_bos: bool | None = None, add_eos: bool | None = None
    ) -> list[int]:
        """Encode `text` to token ids, adding BOS/EOS as requested (or per the tokenizer's defaults) if not already added by the tokenizer itself."""
        add_bos = self.default_add_bos if add_bos is None else add_bos
        add_eos = self.default_add_eos if add_eos is None else add_eos

        token_ids = self.tokenizer.encode(text).ids

        if not self.hf_adds_bos and add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)

        if not self.hf_adds_eos and add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode a list of token ids back to text."""
        return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        """Return the full token-to-id vocabulary mapping."""
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> int | None:
        """Look up the id for a token string, or None if it's not in the vocabulary."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Look up the token string for an id, or None if it's out of range."""
        return self.tokenizer.id_to_token(token_id)