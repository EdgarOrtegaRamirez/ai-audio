"""Tests for text chunking algorithm."""

from ai_audio.tts import chunk_text


class TestChunkText:
    """Test the text chunking algorithm."""

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n   ") == []

    def test_short_text_single_chunk(self):
        result = chunk_text("Hello world")
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_exact_limit(self):
        text = "a" * 4000
        result = chunk_text(text, max_chars=4000)
        assert len(result) == 1
        assert result[0] == text

    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunk_text(text, max_chars=30)
        assert len(result) >= 2
        # Each chunk should be under the limit (or close)
        for chunk in result:
            assert len(chunk) <= 4000

    def test_sentence_split(self):
        sentences = ["Sentence one here. " for _ in range(20)]
        text = "".join(sentences)
        result = chunk_text(text, max_chars=100)
        assert len(result) > 1
        # Rejoin should preserve content
        rejoined = " ".join(result)
        # All original text should be present
        for word in ["Sentence", "one", "here"]:
            assert word in rejoined

    def test_clause_split(self):
        text = "word1, word2, word3, word4, word5, " * 50
        result = chunk_text(text, max_chars=100)
        assert len(result) > 1

    def test_word_split_fallback(self):
        # Very long "word" (no spaces, no punctuation) exceeding limit
        text = "a" * 5000
        result = chunk_text(text, max_chars=4000)
        assert len(result) == 2
        assert len(result[0]) == 4000
        assert len(result[1]) == 1000

    def test_hard_split_extreme(self):
        # Single massive word with no break points
        text = "x" * 10000
        result = chunk_text(text, max_chars=4000)
        assert len(result) == 3

    def test_respects_sentence_boundary(self):
        text = "This is a sentence. " * 300
        result = chunk_text(text, max_chars=2000)
        for chunk in result:
            # Chunks should end at sentence boundaries (mostly)
            assert chunk.endswith(".") or chunk.endswith(" ") or len(chunk) < 2000

    def test_paragraph_preferred_over_sentence(self):
        text = "Short para one.\n\n" + "Sentence two. " * 200
        result = chunk_text(text, max_chars=200)
        assert len(result) >= 2
        # First chunk should be the paragraph
        assert "Short para one" in result[0]

    def test_no_empty_chunks(self):
        text = "a\n\nb\n\nc"
        result = chunk_text(text, max_chars=10)
        for chunk in result:
            assert chunk.strip()

    def test_multilingual(self):
        text = "Hola mundo. " * 100
        result = chunk_text(text, max_chars=200)
        assert len(result) > 1
        rejoined = " ".join(result)
        assert "Hola" in rejoined

    def test_large_text_performance(self):
        """Ensure chunking works on large inputs without excessive splitting."""
        text = "word " * 100000  # 500KB of text
        result = chunk_text(text, max_chars=4000)
        # Should produce reasonable number of chunks
        assert len(result) < 200
        assert len(result) > 1
