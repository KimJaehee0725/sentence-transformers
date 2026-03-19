"""Tests for BaseModelCardData: snippet generation, asset saving, dataset metrics, and multimodal support."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.model_card import BaseModelCardData, generate_model_card
from sentence_transformers.util import is_datasets_available, is_training_available

try:
    from PIL import Image as PILModule
except ImportError:
    PILModule = None

if is_datasets_available():
    from datasets import Dataset, DatasetDict

    try:
        from datasets import Audio as AudioFeature
        from datasets import Image as ImageFeature
    except ImportError:
        AudioFeature = None
        ImageFeature = None

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def _make_model_card_data(cls=BaseModelCardData, **kwargs) -> BaseModelCardData:
    """Create a BaseModelCardData instance with common defaults for testing."""
    data = cls(**kwargs)
    return data


def _make_pil_image(width: int = 64, height: int = 64) -> PILModule.Image:
    """Create a small dummy PIL image."""
    return PILModule.new("RGB", (width, height), color=(255, 0, 0))


def _reset_for_text_snippet(model: SentenceTransformer) -> None:
    """Reset model_card_data fields to ensure a clean text-only snippet test."""
    model.model_card_data.predict_example = None
    model.model_card_data.predict_example_display = None
    model.model_card_data.similarities = None
    model.model_card_data.ir_model = None


class _FakeLoss:
    """Minimal stand-in for a loss object in compute_dataset_metrics calls."""

    pass


class TestIsTypedMediaDict:
    def test_audio_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict({"array": np.zeros(10), "sampling_rate": 16000}) is True

    def test_video_dict(self) -> None:
        assert (
            BaseModelCardData._is_typed_media_dict({"array": np.zeros((8, 3, 64, 64)), "video_metadata": {"fps": 30}})
            is True
        )

    def test_multimodal_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict({"text": "hello", "image": "url"}) is False

    def test_empty_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict({}) is False

    def test_non_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict("not a dict") is False


class TestFormatSnippetValue:
    """Test _asset_path_to_url and _format_snippet_value."""

    def test_asset_url_with_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = "user/my-model"
        url = data._asset_path_to_url("assets/image_0.jpg")
        assert url == "https://huggingface.co/user/my-model/resolve/main/assets/image_0.jpg"

    def test_asset_url_without_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = None
        url = data._asset_path_to_url("assets/image_0.jpg")
        assert url == "assets/image_0.jpg"

    def test_plain_string(self) -> None:
        data = _make_model_card_data()
        assert data._format_snippet_value("hello") == "'hello'"

    def test_asset_path_with_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = "user/model"
        result = data._format_snippet_value("assets/image_0.jpg")
        assert "huggingface.co/user/model" in result
        assert "assets/image_0.jpg" in result

    def test_asset_path_without_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = None
        result = data._format_snippet_value("assets/image_0.jpg")
        assert result == "'assets/image_0.jpg'"


class TestFormatExampleValue:
    """Test formatting for the dataset examples table in the model card."""

    def test_short_string(self) -> None:
        assert BaseModelCardData._format_example_value("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        long_str = "x" * 1500
        result = BaseModelCardData._format_example_value(long_str)
        assert result.endswith("...")
        assert len(result) == 1003  # 1000 chars + "..."

    def test_list_truncated(self) -> None:
        result = BaseModelCardData._format_example_value([1, 2, 3, 4, 5, 6, 7])
        assert "...]" in result

    def test_short_list_not_truncated(self) -> None:
        result = BaseModelCardData._format_example_value([1, 2, 3])
        assert "..." not in result

    def test_newlines_replaced(self) -> None:
        result = BaseModelCardData._format_example_value("line1\nline2")
        assert "<br>" in result
        assert "\n" not in result

    def test_pipe_escaped(self) -> None:
        result = BaseModelCardData._format_example_value("a|b")
        assert "\\|" in result

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_pil_image_placeholder(self) -> None:
        img = _make_pil_image(100, 200)
        result = BaseModelCardData._format_example_value(img)
        assert "image" in result
        assert "100x200" in result

    def test_audio_dict_placeholder(self) -> None:
        audio = {"array": np.zeros(16000), "sampling_rate": 16000}
        result = BaseModelCardData._format_example_value(audio)
        assert "audio" in result
        assert "1.00s" in result
        assert "16000 Hz" in result

    def test_video_dict_placeholder(self) -> None:
        video = {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}}
        result = BaseModelCardData._format_example_value(video)
        assert "video" in result


class TestSavePredictExampleAssets:
    """Test saving non-text predict examples to files."""

    @pytest.mark.parametrize(
        "predict_example",
        [
            ["hello", "world"],
            [["q", "a1"], ["q", "a2"]],
        ],
        ids=["flat_strings", "list_of_lists"],
    )
    def test_text_only_does_not_create_assets_dir(self, predict_example) -> None:
        """When all examples are text (flat or CrossEncoder-style pairs), no assets dir is created."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = predict_example
            data.save_predict_example_assets()

            assert not os.path.exists(os.path.join(tmpdir, "assets"))
            assert data.predict_example_display is None

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_pil_image_saved_as_jpg(self) -> None:
        """A PIL image in predict_example is saved to assets/image_0.jpg."""
        data = _make_model_card_data()
        img = _make_pil_image(32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = [img]
            data.save_predict_example_assets()

            assert data.predict_example_display == ["assets/image_0.jpg"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_multiple_images_indexed_correctly(self) -> None:
        """Multiple distinct images get sequential indices."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            # Use different sizes so they're not deduplicated
            data.predict_example = [_make_pil_image(32, 32), _make_pil_image(48, 48), _make_pil_image(64, 64)]
            data.save_predict_example_assets()

            assert data.predict_example_display == [
                "assets/image_0.jpg",
                "assets/image_1.jpg",
                "assets/image_2.jpg",
            ]
            for i in range(3):
                assert os.path.isfile(os.path.join(tmpdir, "assets", f"image_{i}.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_duplicate_images_deduplicated(self) -> None:
        """Identical images are saved only once and share the same path."""
        data = _make_model_card_data()
        img = _make_pil_image(32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = [img, img, img]
            data.save_predict_example_assets()

            assert data.predict_example_display == [
                "assets/image_0.jpg",
                "assets/image_0.jpg",
                "assets/image_0.jpg",
            ]
            # Only one file saved
            assert os.listdir(os.path.join(tmpdir, "assets")) == ["image_0.jpg"]

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_multimodal_dict_with_text_and_image(self) -> None:
        """Multimodal dict: text kept as-is, image saved to file."""
        data = _make_model_card_data()
        img = _make_pil_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = [{"text": "A cat", "image": img}]
            data.save_predict_example_assets()

            assert len(data.predict_example_display) == 1
            display = data.predict_example_display[0]
            assert display["text"] == "A cat"
            assert display["image"] == "assets/image_0.jpg"
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_mixed_text_and_images(self) -> None:
        """A mix of strings and images in predict_example."""
        data = _make_model_card_data()
        img = _make_pil_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = ["hello", img, "world"]
            data.save_predict_example_assets()

            assert data.predict_example_display[0] == "hello"
            assert data.predict_example_display[1] == "assets/image_0.jpg"
            assert data.predict_example_display[2] == "world"

    def test_audio_dict_saved_as_wav(self) -> None:
        """AudioDict saved to assets/ as .wav file."""
        try:
            import soundfile  # noqa: F401
        except ImportError:
            pytest.skip("soundfile not installed")

        data = _make_model_card_data()
        audio = {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000}
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = [audio]
            data.save_predict_example_assets()

            assert data.predict_example_display == ["assets/audio_0.wav"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "audio_0.wav"))

    def test_video_dict_saved_as_mp4(self) -> None:
        """VideoDict saved to assets/ as .mp4 file."""
        try:
            from torchcodec.encoders import VideoEncoder  # noqa: F401
        except ImportError:
            pytest.skip("torchcodec not installed")

        data = _make_model_card_data()
        # (T, C, H, W) uint8 video tensor
        video = {"array": np.random.randint(0, 255, (8, 3, 64, 64), dtype=np.uint8), "video_metadata": {"fps": 30}}
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = [video]
            data.save_predict_example_assets()

            assert data.predict_example_display == ["assets/video_0.mp4"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))

    def test_no_save_dir_is_noop(self) -> None:
        """When save_dir is not set, nothing happens."""
        data = _make_model_card_data()
        data.predict_example = [_make_pil_image()] if PILModule else ["text"]
        data.save_dir = None
        data.save_predict_example_assets()

        assert data.predict_example_display is None

    def test_no_predict_example_is_noop(self) -> None:
        """When predict_example is None, nothing happens."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.predict_example = None
            data.save_predict_example_assets()

            assert data.predict_example_display is None


class TestGenerateUsageSnippet:
    """Test snippet generation for both text-only and non-text inputs."""

    def test_text_default_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Default examples are used when predict_example is None."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        snippet = model.model_card_data.generate_usage_snippet()

        assert "```python" in snippet
        assert "```" in snippet
        assert "SentenceTransformer" in snippet
        assert "'The weather is lovely today.'" in snippet
        assert "model.encode(sentences)" in snippet
        assert "model.similarity(embeddings, embeddings)" in snippet

    def test_text_custom_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """predict_example strings appear in the snippet."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.predict_example = ["Hello", "World", "Test"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "'Hello'" in snippet
        assert "'World'" in snippet
        assert "'Test'" in snippet
        assert "# [3," in snippet

    def test_text_with_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """When similarities are computed, they appear in the snippet."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.predict_example = ["A", "B"]
        model.model_card_data.similarities = "# tensor([[1.0, 0.5],\n#         [0.5, 1.0]])"
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities)" in snippet
        assert "# tensor([[1.0, 0.5]," in snippet

    def test_text_without_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """When no similarities, shape comment is shown."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.predict_example = ["A", "B"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities.shape)" in snippet
        assert "# [2, 2]" in snippet

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("tomaarsen/my-cool-model", 'SentenceTransformer("tomaarsen/my-cool-model")'),
            (None, "sentence_transformers_model_id"),
        ],
        ids=["custom_id", "default_placeholder"],
    )
    def test_text_model_id(self, stsb_bert_tiny_model: SentenceTransformer, model_id, expected) -> None:
        """model_id appears in the loading line, or a placeholder is used when None."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.model_id = model_id
        model.model_card_data.predict_example = ["test"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert expected in snippet

    def test_text_output_dimensionality(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Embedding dimension appears in the shape comment."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.predict_example = ["A", "B"]
        dim = model.get_embedding_dimension()
        snippet = model.model_card_data.generate_usage_snippet()

        assert f"# [2, {dim}]" in snippet

    def test_display_precedence(self) -> None:
        """predict_example_display takes precedence over predict_example for rendering."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.predict_example = ["original A", "original B"]
        data.predict_example_display = ["display A", "display B"]
        snippet = data.generate_usage_snippet()

        assert "'display A'" in snippet
        assert "'display B'" in snippet
        assert "original" not in snippet

    def test_non_text_multimodal_dict(self) -> None:
        """Multimodal dicts produce inputs = [{...}, ...] format."""
        data = _make_model_card_data()
        data.model_id = "user/model"
        data.similarities = None
        data.model = None
        display = [
            {"text": "A cat", "image": "assets/image_0.jpg"},
            {"text": "A dog", "image": "assets/image_1.jpg"},
        ]
        snippet = data._generate_non_text_snippet(display)

        assert "inputs = [" in snippet
        assert "'text'" in snippet
        assert "'image'" in snippet
        assert "huggingface.co/user/model" in snippet
        assert "model.encode(inputs)" in snippet
        assert "model.similarity(" in snippet

    def test_non_text_single_modality(self) -> None:
        """Single non-text modality produces inputs = [...] format."""
        data = _make_model_card_data()
        data.model_id = "user/model"
        data.similarities = None
        data.model = None
        display = ["assets/image_0.jpg", "assets/image_1.jpg"]
        snippet = data._generate_non_text_snippet(display)

        assert "inputs = [" in snippet
        assert "huggingface.co/user/model" in snippet
        assert "model.encode(inputs)" in snippet

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_non_text_dispatch_multimodal(self) -> None:
        """generate_usage_snippet dispatches to multimodal for dict predict_example."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.predict_example = [{"text": "A", "image": _make_pil_image()}]
        data.predict_example_display = [{"text": "A", "image": "assets/image_0.jpg"}]
        snippet = data.generate_usage_snippet()

        assert "inputs = [" in snippet
        assert "'text'" in snippet

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_non_text_dispatch_single_modality(self) -> None:
        """PIL images in predict_example cause single-modality dispatch, rendering display paths."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.predict_example = [_make_pil_image(), _make_pil_image()]
        data.predict_example_display = ["assets/image_0.jpg", "assets/image_1.jpg"]
        snippet = data.generate_usage_snippet()

        assert "inputs = [" in snippet
        assert "assets/image_0.jpg" in snippet


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestSetMultimodalPredictExample:
    """Test multimodal predict_example extraction from datasets."""

    def _make_text_image_dataset(self, n: int = 5) -> DatasetDict:
        images = [_make_pil_image() for _ in range(n)]
        ds = Dataset.from_dict({"text": [f"text {i}" for i in range(n)], "image": images})
        ds = ds.cast_column("image", ImageFeature())
        return DatasetDict(train=ds)

    def _make_image_dataset(self, n: int = 5) -> DatasetDict:
        ds = Dataset.from_dict({"image": [_make_pil_image() for _ in range(n)]})
        ds = ds.cast_column("image", ImageFeature())
        return DatasetDict(train=ds)

    def test_combined_modality_builds_dicts(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A model with tuple modality ("image", "text") builds multimodal dicts from text+image columns."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        dd = self._make_text_image_dataset()

        # BLIP-like: supports text, image, and the combined ("image", "text")
        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=["text", "image", ("image", "text")],
        ):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert model.model_card_data.predict_example is not None
        assert len(model.model_card_data.predict_example) == 3
        first = model.model_card_data.predict_example[0]
        assert isinstance(first, dict)
        assert "text" in first
        assert "image" in first

    def test_separate_modalities_no_combined_dict(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A CLIP-like model (text + image separately, no tuple) does NOT build combined dicts.

        Instead, it picks the first non-text modality and shows single-modality examples.
        """
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        dd = self._make_text_image_dataset()

        # CLIP-like: supports text and image independently, but no ("image", "text") tuple
        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=["text", "image"],
        ):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert model.model_card_data.predict_example is not None
        assert len(model.model_card_data.predict_example) == 3
        # Should be raw images, NOT dicts — CLIP can't process combined inputs
        assert not isinstance(model.model_card_data.predict_example[0], dict)

    def test_image_only_model(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image-only model: predict_example is a list of images, not dicts."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        dd = self._make_image_dataset()

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert model.model_card_data.predict_example is not None
        assert len(model.model_card_data.predict_example) == 3
        assert not isinstance(model.model_card_data.predict_example[0], dict)

    def test_text_only_model_skips(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Text-only model does not produce multimodal predict_example."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.predict_example = ["original"]
        dd = self._make_text_image_dataset()

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text"]):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert model.model_card_data.predict_example == ["original"]

    def test_combined_only_model(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A Kosmos-like model with only ("image", "text") still builds combined dicts."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        dd = self._make_text_image_dataset()

        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=[("image", "text")],
        ):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert len(model.model_card_data.predict_example) == 3
        first = model.model_card_data.predict_example[0]
        assert isinstance(first, dict)
        assert "text" in first and "image" in first

    def test_small_dataset_fewer_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Dataset smaller than 3 examples: uses what's available."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        dd = self._make_image_dataset(n=1)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert len(model.model_card_data.predict_example) == 1

    def test_no_matching_columns_skips(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """If dataset has no columns matching the model's non-text modalities, nothing happens."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.predict_example = ["original"]

        # Dataset with only text, but model wants audio
        ds = Dataset.from_dict({"text": ["hello", "world"]})
        dd = DatasetDict(train=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["audio"]):
            model.model_card_data._set_multimodal_predict_example(dd)

        assert model.model_card_data.predict_example == ["original"]


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestComputeDatasetMetricsNonText:
    """Test dataset statistics for image/audio columns."""

    def test_image_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image columns show width x height stats."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        images = [_make_pil_image(w, w) for w in [32, 64, 128]]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "stats" in result
        assert "image" in result["stats"]
        assert result["stats"]["image"]["dtype"] == "image"
        assert "32x32 px" in result["stats"]["image"]["data"]["min"]
        assert "128x128 px" in result["stats"]["image"]["data"]["max"]

    def test_image_example_placeholder_without_save_dir(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Without save_dir, image examples show placeholder text."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.save_dir = None
        images = [_make_pil_image(64, 64) for _ in range(5)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "examples_table" in result
        assert "image" in result["examples_table"].lower() or "64x64" in result["examples_table"]
        assert "<img" not in result["examples_table"]

    def test_image_example_saved_with_save_dir(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """With save_dir, image examples are saved as files and rendered with <img> tags."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        # Use different sizes so they're not deduplicated
        images = [_make_pil_image(32 + i * 16, 32 + i * 16) for i in range(5)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            info = {"name": "test"}
            result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "examples_table" in result
            assert "<img" in result["examples_table"]
            assert "example_image_0.jpg" in result["examples_table"]
            # 3 example rows, 1 image column, all distinct = 3 images saved
            for i in range(3):
                assert os.path.isfile(os.path.join(tmpdir, "assets", f"example_image_{i}.jpg"))


class TestEndToEnd:
    """End-to-end: model card generation produces valid output."""

    def test_text_model_card_valid(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Text model generates a valid card with no triple newlines."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.local_files_only = True
        model_card = generate_model_card(model)

        assert "```python" in model_card
        assert "SentenceTransformer" in model_card
        assert "\n\n\n" not in model_card

    def test_model_card_with_model_id(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """model_id flows through to the generated card."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "tomaarsen/test-model"
        model_card = generate_model_card(model)

        assert 'SentenceTransformer("tomaarsen/test-model")' in model_card

    def test_text_modality_in_card(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Text-only model shows 'Text' in supported modalities."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.local_files_only = True
        model.model_card_data.predict_example = ["A", "B"]
        model_card = generate_model_card(model)

        assert "Supported Modality" in model_card
        assert "Text" in model_card

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_model_card_with_image_predict_example(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image predict_example causes multimodal snippet generation."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/multimodal-model"
        model.model_card_data.predict_example = [
            {"text": "A cat", "image": _make_pil_image()},
            {"text": "A dog", "image": _make_pil_image()},
        ]
        # We need a save_dir for asset saving
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            # Should have multimodal snippet
            assert "inputs = [" in model_card
            assert "'text'" in model_card
            assert "'image'" in model_card
            # Assets should be saved
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_model_card_with_image_only_predict_example(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image-only predict_example generates single-modality snippet."""
        model = stsb_bert_tiny_model
        _reset_for_text_snippet(model)
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/image-model"
        model.model_card_data.predict_example = [_make_pil_image(), _make_pil_image()]
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            assert "inputs = [" in model_card
            assert "huggingface.co/user/image-model" in model_card
