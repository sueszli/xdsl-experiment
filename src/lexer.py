import re
from enum import Enum, auto
from typing import TypeAlias

from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Lexer, Position, Span, Token


class AzizTokenKind(Enum):
    # punctuation
    PAREN_OPEN = auto()
    PAREN_CLOSE = auto()
    EOF = auto()

    # keywords
    DEFUN = auto()
    PRINT = auto()
    IF = auto()

    # literals and identifiers
    IDENTIFIER = auto()  # function or variable name
    NUMBER = auto()
    STRING = auto()


SINGLE_CHAR_TOKENS = {
    "(": AzizTokenKind.PAREN_OPEN,
    ")": AzizTokenKind.PAREN_CLOSE,
}

KEYWORD_TOKENS = {
    "defun": AzizTokenKind.DEFUN,
    "print": AzizTokenKind.PRINT,
    "if": AzizTokenKind.IF,
}

AzizToken: TypeAlias = Token[AzizTokenKind]


class AzizLexer(Lexer[AzizTokenKind]):
    def _is_in_bounds(self, size: Position = 1) -> bool:
        return self.pos + size - 1 < self.input.len

    def _get_chars(self, size: int = 1) -> str | None:
        if not self._is_in_bounds(size):
            return None
        res = self.input.slice(self.pos, self.pos + size)
        self.pos += size
        return res

    def _peek_chars(self, size: int = 1) -> str | None:
        if not self._is_in_bounds(size):
            return None
        return self.input.slice(self.pos, self.pos + size)

    def _consume_chars(self, size: int = 1) -> None:
        self.pos += size

    def _consume_regex(self, regex: re.Pattern[str]) -> re.Match[str] | None:
        match = regex.match(self.input.content, self.pos)
        if match is None:
            return None
        self.pos = match.end()
        return match

    _whitespace_regex = re.compile(r"((;[^\n]*(\n)?)|(\s+))*", re.ASCII)  # comment or whitespace

    def _consume_whitespace(self) -> None:
        self._consume_regex(self._whitespace_regex)

    def lex(self) -> AzizToken:
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._peek_chars()

        if current_char is None:
            return self._form_token(AzizTokenKind.EOF, start_pos)

        if current_char == '"':
            return self._lex_string(start_pos)

        # Basic number check (digit or -digit)
        if current_char.isdigit() or (current_char == "-" and self._peek_chars(2) and self._peek_chars(2)[1].isdigit()):
            return self._lex_number(start_pos)

        # Single char tokens
        if current_char in SINGLE_CHAR_TOKENS:
            self._consume_chars()
            return self._form_token(SINGLE_CHAR_TOKENS[current_char], start_pos)

        # Identifiers
        if self._is_identifier_char(current_char):
            return self._lex_identifier(start_pos)

        raise ParseError(
            Span(start_pos, start_pos + 1, self.input),
            f"Unexpected character: {current_char}",
        )

    def _is_identifier_char(self, char: str) -> bool:
        # Allow typical lisp identifier chars
        return char.isalnum() or char in "_+-*/%<>=!?"

    _identifier_regex = re.compile(r"[a-zA-Z0-9_+\-*/%<>=!?]+")

    def _lex_identifier(self, start_pos: Position) -> AzizToken:
        self._consume_regex(self._identifier_regex)

        span = Span(start_pos, self.pos, self.input)
        kind = KEYWORD_TOKENS.get(span.text, AzizTokenKind.IDENTIFIER)
        return Token(kind, span)

    _number_regex = re.compile(r"-?\d+(\.\d+)?")

    def _lex_number(self, start_pos: Position) -> AzizToken:
        self._consume_regex(self._number_regex)
        return self._form_token(AzizTokenKind.NUMBER, start_pos)

    def _lex_string(self, start_pos: Position) -> AzizToken:
        self._consume_chars()  # consume opening quote

        while True:
            char = self._get_chars()
            if char is None:
                raise ParseError(
                    Span(start_pos, self.pos, self.input),
                    "Unterminated string literal",
                )

            if char == '"':
                break

            if char == "\\":
                self._consume_chars()  # skip escaped char

        return self._form_token(AzizTokenKind.STRING, start_pos)
