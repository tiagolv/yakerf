"""Tests for the CLI interface."""

import os
import tempfile
import pytest
from click.testing import CliRunner
from yake.cli import keywords


class TestCLI:
    """Test suite for YAKE CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Fixture that returns a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_text(self):
        """Fixture that provides sample text for keyword extraction."""
        return "YAKE is a light-weight unsupervised automatic keyword extraction method which rests on statistical text features extracted from single documents."

    @pytest.fixture
    def sample_file(self, sample_text):
        """Fixture that creates a temporary file with sample text."""
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_text)
            filename = f.name

        yield filename

        # Cleanup after test
        if os.path.exists(filename):
            os.unlink(filename)

    def test_help_command(self, runner):
        """Test the help command."""
        result = runner.invoke(keywords, ["--help"])
        assert result.exit_code == 0
        assert "Extract keywords using YAKE!" in result.output
        assert "--text_input" in result.output
        assert "--input_file" in result.output
        assert "--language" in result.output
        assert "--ngram_size" in result.output

    def test_text_input(self, runner, sample_text):
        """Test keyword extraction from text input."""
        result = runner.invoke(keywords, ["--text_input", sample_text])
        assert result.exit_code == 0
        assert "keyword" in result.output
        # Should extract some keywords
        lines = result.output.strip().split('\n')
        assert len(lines) > 1  # At least header + some keywords

    def test_text_input_short_flag(self, runner, sample_text):
        """Test keyword extraction using short flag -ti."""
        result = runner.invoke(keywords, ["-ti", sample_text])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_file_input(self, runner, sample_file):
        """Test keyword extraction from file input."""
        result = runner.invoke(keywords, ["--input_file", sample_file])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_file_input_short_flag(self, runner, sample_file):
        """Test keyword extraction from file using short flag -i."""
        result = runner.invoke(keywords, ["-i", sample_file])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_verbose_output(self, runner, sample_text):
        """Test verbose output with scores."""
        result = runner.invoke(keywords, ["--text_input", sample_text, "--verbose"])
        assert result.exit_code == 0
        assert "keyword" in result.output
        assert "score" in result.output

    def test_verbose_short_flag(self, runner, sample_text):
        """Test verbose output using short flag -v."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-v"])
        assert result.exit_code == 0
        assert "keyword" in result.output
        assert "score" in result.output

    def test_ngram_size_option(self, runner, sample_text):
        """Test the ngram-size option."""
        # Test with ngram size 1 (single words only)
        result = runner.invoke(keywords, ["--text_input", sample_text, "--ngram_size", "1"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_ngram_size_short_flag(self, runner, sample_text):
        """Test ngram-size using short flag -n."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-n", "2"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_top_option(self, runner, sample_text):
        """Test the top option to limit number of keywords."""
        result = runner.invoke(keywords, ["--text_input", sample_text, "--top", "5"])
        assert result.exit_code == 0
        assert "keyword" in result.output
        # Count lines excluding header and empty lines
        lines = [line for line in result.output.strip().split('\n') if line.strip()]
        # Should have header + up to 5 keywords
        assert len(lines) <= 6  # header + 5 keywords max

    def test_top_short_flag(self, runner, sample_text):
        """Test top option using short flag -t."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-t", "3"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_language_option(self, runner, sample_text):
        """Test the language option."""
        # Test with English (default)
        result_en = runner.invoke(keywords, ["--text_input", sample_text, "--language", "en"])
        assert result_en.exit_code == 0
        assert "keyword" in result_en.output

        # Test with Portuguese
        result_pt = runner.invoke(keywords, ["--text_input", sample_text, "--language", "pt"])
        assert result_pt.exit_code == 0
        assert "keyword" in result_pt.output

    def test_language_short_flag(self, runner, sample_text):
        """Test language option using short flag -l."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-l", "en"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_dedup_func_option(self, runner, sample_text):
        """Test the deduplication function option."""
        # Test each deduplication function
        for func in ["leve", "jaro", "seqm"]:
            result = runner.invoke(keywords, ["--text_input", sample_text, "--dedup_func", func])
            assert result.exit_code == 0
            assert "keyword" in result.output

    def test_dedup_func_short_flag(self, runner, sample_text):
        """Test dedup_func using short flag -df."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-df", "jaro"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_dedup_lim_option(self, runner, sample_text):
        """Test the deduplication limit option."""
        result = runner.invoke(keywords, ["--text_input", sample_text, "--dedup_lim", "0.8"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_dedup_lim_short_flag(self, runner, sample_text):
        """Test dedup_lim using short flag -dl."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-dl", "0.7"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_window_size_option(self, runner, sample_text):
        """Test the window size option."""
        result = runner.invoke(keywords, ["--text_input", sample_text, "--window_size", "2"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_window_size_short_flag(self, runner, sample_text):
        """Test window_size using short flag -ws."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-ws", "3"])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_combined_options(self, runner, sample_text):
        """Test multiple options combined."""
        result = runner.invoke(keywords, [
            "--text_input", sample_text,
            "--language", "en",
            "--ngram_size", "2",
            "--top", "5",
            "--verbose",
            "--dedup_func", "jaro",
            "--dedup_lim", "0.8",
            "--window_size", "2"
        ])
        assert result.exit_code == 0
        assert "keyword" in result.output
        assert "score" in result.output

    def test_combined_short_flags(self, runner, sample_text):
        """Test multiple short flags combined."""
        result = runner.invoke(keywords, [
            "-ti", sample_text,
            "-l", "en",
            "-n", "2",
            "-t", "3",
            "-v",
            "-df", "seqm",
            "-dl", "0.9",
            "-ws", "1"
        ])
        assert result.exit_code == 0
        assert "keyword" in result.output
        assert "score" in result.output

    def test_error_no_input(self, runner):
        """Test error when no input is provided."""
        result = runner.invoke(keywords, [])
        assert result.exit_code == 1
        assert "Specify either an input file or direct text input" in result.output

    def test_error_both_inputs(self, runner, sample_text, sample_file):
        """Test error when both inputs are provided."""
        result = runner.invoke(keywords, ["--text_input", sample_text, "--input_file", sample_file])
        assert result.exit_code == 1
        assert "Specify either an input file or direct text input, not both!" in result.output

    def test_error_both_inputs_short_flags(self, runner, sample_text, sample_file):
        """Test error when both inputs are provided using short flags."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-i", sample_file])
        assert result.exit_code == 1
        assert "Specify either an input file or direct text input, not both!" in result.output

    def test_nonexistent_file(self, runner):
        """Test error when input file does not exist."""
        result = runner.invoke(keywords, ["--input_file", "nonexistent_file.txt"])
        assert result.exit_code == 1
        assert "File 'nonexistent_file.txt' not found." in result.output

    def test_nonexistent_file_short_flag(self, runner):
        """Test error when input file does not exist using short flag."""
        result = runner.invoke(keywords, ["-i", "nonexistent_file.txt"])
        assert result.exit_code == 1
        assert "File 'nonexistent_file.txt' not found." in result.output

    def test_invalid_dedup_func(self, runner, sample_text):
        """Test error with invalid deduplication function."""
        result = runner.invoke(keywords, ["--text_input", sample_text, "--dedup_func", "invalid"])
        assert result.exit_code != 0
        # Click should handle this validation and show error

    def test_invalid_numeric_values(self, runner, sample_text):
        """Test error with invalid numeric values."""
        # Test invalid ngram_size
        result = runner.invoke(keywords, ["--text_input", sample_text, "--ngram_size", "not_a_number"])
        assert result.exit_code != 0

        # Test invalid top
        result = runner.invoke(keywords, ["--text_input", sample_text, "--top", "not_a_number"])
        assert result.exit_code != 0

        # Test invalid dedup_lim
        result = runner.invoke(keywords, ["--text_input", sample_text, "--dedup_lim", "not_a_number"])
        assert result.exit_code != 0

        # Test invalid window_size
        result = runner.invoke(keywords, ["--text_input", sample_text, "--window_size", "not_a_number"])
        assert result.exit_code != 0

    def test_empty_text_input(self, runner):
        """Test behavior with empty text input."""
        result = runner.invoke(keywords, ["--text_input", ""])
        # Should not crash, but may not produce meaningful results
        assert result.exit_code == 0

    def test_empty_file(self, runner):
        """Test behavior with empty file."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as f:
            f.write("")  # Empty file
            filename = f.name

        try:
            result = runner.invoke(keywords, ["--input_file", filename])
            # Should not crash, but may not produce meaningful results
            assert result.exit_code == 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_multiple_verbose_flags(self, runner, sample_text):
        """Test multiple verbose flags (should work the same as single -v)."""
        result = runner.invoke(keywords, ["-ti", sample_text, "-v", "-v", "-v"])
        assert result.exit_code == 0
        assert "keyword" in result.output
        assert "score" in result.output

    def test_large_text_input(self, runner):
        """Test with larger text input."""
        large_text = "Machine learning is a method of data analysis that automates analytical model building. " * 10
        result = runner.invoke(keywords, ["--text_input", large_text])
        assert result.exit_code == 0
        assert "keyword" in result.output

    def test_special_characters_in_text(self, runner):
        """Test with text containing special characters."""
        special_text = "Text with special chars: áéíóú, ñ, ç, and symbols like @#$%"
        result = runner.invoke(keywords, ["--text_input", special_text])
        assert result.exit_code == 0
        # Should handle gracefully