from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import ModelCard

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

    try:
        from datasets import Image as ImageFeature
    except ImportError:
        ImageFeature = None
    try:
        from datasets import Audio as AudioFeature
    except ImportError:
        AudioFeature = None

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.model import CrossEncoder


class CrossEncoderModelCardCallback(BaseModelCardCallback):
    def __init__(self, default_args_dict: dict[str, Any]) -> None:
        super().__init__(default_args_dict)


@dataclass
class CrossEncoderModelCardData(BaseModelCardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "CrossEncoder based on answerdotai/ModernBERT-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/ce-mpnet-base-ms-marco".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic search and paraphrase mining".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "cross-encoder"].
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
            Defaults to False.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = CrossEncoder(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=CrossEncoderModelCardData(
        ...         model_id="tomaarsen/ce-mpnet-base-allnli",
        ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    task_name: str = field(default=None)
    tags: list[str] | None = field(
        default_factory=lambda: [
            "sentence-transformers",
            "cross-encoder",
            "reranker",
        ]
    )

    # Automatically filled by `CrossEncoderModelCardCallback` and the Trainer directly
    predict_example: list[list] | None = field(default=None, init=False)
    ir_model: bool | None = field(default=True, init=False, repr=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default=None, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)

    # Passed via `register_model` only
    model: CrossEncoder | None = field(default=None, init=False, repr=False)

    def set_widget_examples(self, dataset: Dataset | DatasetDict) -> None:
        """
        We don't set widget examples, but only load the prediction example.
        This is because the Hugging Face Hub doesn't currently have a Sentence Ranking
        or Text Classification widget that accepts pairs, which is what CrossEncoder
        models require.
        """
        if isinstance(dataset, DatasetDict):
            dataset = dataset[list(dataset.keys())[0]]

        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            return

        if len(dataset) == 0:
            return

        first_sample = dataset[0]

        # Find the first two columns that are text, image, or audio (skip label/dataset_name columns)
        pair_columns = []
        for column, value in first_sample.items():
            if column in ("dataset_name", "label"):
                continue
            is_text = isinstance(value, str) or (isinstance(value, list) and value and isinstance(value[0], str))
            is_non_text = False
            if hasattr(dataset, "features"):
                feature = dataset.features.get(column)
                if (ImageFeature and isinstance(feature, ImageFeature)) or (
                    AudioFeature and isinstance(feature, AudioFeature)
                ):
                    is_non_text = True
            if is_text or is_non_text:
                pair_columns.append(column)
            if len(pair_columns) == 2:
                break

        if len(pair_columns) < 2:
            return

        query_column, answer_column = pair_columns
        answer_type = type(first_sample[answer_column])

        queries = dataset[:5][query_column]
        answers = dataset[:5][answer_column]

        # If the response is a list, then the first query-answer is a nice example
        if answer_type is list:
            answers = answers[0][:5]
            queries = [queries[0]] * len(answers)

        self.predict_example = [[query, answer] for query, answer in zip(queries, answers)]

    def register_model(self, model) -> None:
        super().register_model(model)

        if self.task_name is None:
            self.task_name = (
                "text reranking and semantic search" if model.num_labels == 1 else "text pair classification"
            )
        if self.pipeline_tag is None:
            self.pipeline_tag = "text-ranking" if model.num_labels == 1 else "text-classification"

    # def tokenize(self, text: str | list[str], **kwargs) -> dict[str, Any]:
    #     return self.model.tokenizer(text)

    def run_usage_snippet(self) -> dict[str, Any]:
        if self.predict_example is None:
            self.predict_example = [
                [
                    "How many calories in an egg",
                    "There are on average between 55 and 80 calories in an egg depending on its size.",
                ],
                [
                    "How many calories in an egg",
                    "Egg whites are very low in calories, have no fat, no cholesterol, and are loaded with protein.",
                ],
                [
                    "How many calories in an egg",
                    "Most of the calories in an egg come from the yellow yolk in the center.",
                ],
            ]

        if not self.generate_widget_examples:
            return

        import numpy as np

        scores = self.model.predict(self.predict_example, convert_to_numpy=True, show_progress_bar=False)
        with np.printoptions(precision=4):
            self.similarities = "\n".join(f"# {line}" for line in str(scores).splitlines())

    def generate_usage_snippet(self) -> str:
        display = self.predict_example_display or self.predict_example
        examples = display or [
            [
                "How many calories in an egg",
                "There are on average between 55 and 80 calories in an egg depending on its size.",
            ],
            [
                "How many calories in an egg",
                "Egg whites are very low in calories, have no fat, no cholesterol, and are loaded with protein.",
            ],
            [
                "How many calories in an egg",
                "Most of the calories in an egg come from the yellow yolk in the center.",
            ],
        ]
        model_id = self.model_id or "cross_encoder_model_id"
        num_labels = self.model.num_labels if self.model else 1

        # Check if any pair element is non-text (from predict_example before asset saving)
        source = self.predict_example or examples
        is_multimodal = any(
            isinstance(pair, list) and any(not isinstance(elem, str) for elem in pair) for pair in source
        )

        lines = [
            "from sentence_transformers import CrossEncoder",
            "",
            "# Download from the \U0001f917 Hub",
            f'model = CrossEncoder("{model_id}")',
            "# Get scores for pairs of inputs",
            "pairs = [",
        ]
        for pair in examples:
            lines.append(f"    {self._format_snippet_value(pair)},")
        lines.extend(
            [
                "]",
                "scores = model.predict(pairs)",
            ]
        )
        if self.similarities:
            lines.append("print(scores)")
            lines.append(self.similarities)
        else:
            shape_str = f"({len(examples)}, {num_labels})" if num_labels > 1 else f"({len(examples)},)"
            lines.extend(
                [
                    "print(scores.shape)",
                    f"# {shape_str}",
                ]
            )

        if num_labels == 1 and not is_multimodal:
            query = examples[0][0] if examples else "How many calories in an egg"
            documents = [pair[1] for pair in examples] if examples else []
            lines.extend(
                [
                    "",
                    "# Or rank different texts based on similarity to a single text",
                    "ranks = model.rank(",
                    f"    {query!r},",
                    "    [",
                ]
            )
            for doc in documents:
                lines.append(f"        {doc!r},")
            lines.extend(
                [
                    "    ]",
                    ")",
                    "# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]",
                ]
            )

        return "```python\n" + "\n".join(lines) + "\n```"

    def get_model_specific_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_specific_metadata()
        metadata.update(
            {
                "model_num_labels": self.model.num_labels,
            }
        )
        return metadata


def generate_model_card(model: CrossEncoder) -> str:
    model_card = ModelCard.from_template(
        card_data=model.model_card_data, template_path=model.model_card_data.template_path, hf_emoji="🤗"
    )
    return model_card.content
