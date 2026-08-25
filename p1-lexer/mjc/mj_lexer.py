import argparse
import pathlib
import sys

from sly import Lexer


class MJLexer(Lexer):
    """A lexer for the Minijava language. After building it, set the
    input text, and call token() to get new
    tokens.
    """

    def __init__(self, error_func):
        self.error_func = error_func
        self.filename = ""

        # Keeps track of the last token returned from self.token()
        self.last_token = None

    def _error(self, msg, token):
        location = self._make_tok_location(token)
        self.error_func(msg, location[0], location[1])
        self.index += 1

    def find_tok_column(self, token):
        """Find the column of the token in its line."""
        last_cr = self.text.rfind("\n", 0, token.index)
        return token.index - last_cr

    def _make_tok_location(self, token):
        return (self.lineno, self.find_tok_column(token))

    # Error handling rule
    def error(self, t):
        msg = f"Illegal character {t.value[0]!r}"
        self._error(msg, t)

    def scan(self, data):
        output = ""
        for token in self.tokenize(data):
            token_str = (
                f"LexToken({token.type},{token.value!r},{token.lineno},{token.index})"
            )
            print(token_str)
            output += token_str + "\n"
        return output

    # Set of token names.
    tokens = {
        # Keywords
        "CLASS",
        "EXTENDS",
        "PUBLIC",
        "STATIC",
        "VOID",
        "MAIN",
        "STRING",
        "BOOLEAN",
        "CHAR",
        "INT",
        "IF",
        "ELSE",
        "WHILE",
        "FOR",
        "ASSERT",
        "BREAK",
        "RETURN",
        "NEW",
        "THIS",
        "TRUE",
        "FALSE",
        "LENGTH",
        "PRINT",
        # Literals
        "ID",
        "INT_LITERAL",
        "CHAR_LITERAL",
        "STRING_LITERAL",
        # Operators
        "EQ",
        "NE",
        "LE",
        "GE",
        "AND",
        "OR",
        "ASSIGN",
        "LT",
        "GT",
        "PLUS",
        "MINUS",
        "TIMES",
        "DIVIDE",
        "MOD",
        "NOT",
        # Punctuation
        "DOT",
        "SEMI",
        "COMMA",
        "LPAREN",
        "RPAREN",
        "LBRACKET",
        "RBRACKET",
        "LBRACE",
        "RBRACE",
    }

    # ----------------------------------------------------------------
    # Identifiers and reserved words
    # ----------------------------------------------------------------
    # A dictionary mapping reserved words to token types.
    keywords = {
        "class": "CLASS",
        "extends": "EXTENDS",
        "public": "PUBLIC",
        "static": "STATIC",
        "void": "VOID",
        "main": "MAIN",
        "String": "STRING",
        "boolean": "BOOLEAN",
        "print": "PRINT",
        "char": "CHAR",
        "int": "INT",
        "if": "IF",
        "else": "ELSE",
        "while": "WHILE",
        "for": "FOR",
        "assert": "ASSERT",
        "break": "BREAK",
        "return": "RETURN",
        "new": "NEW",
        "this": "THIS",
        "true": "TRUE",
        "false": "FALSE",
        "length": "LENGTH",
    }

    # ----------------------------------------------------------------
    # Rules
    # ----------------------------------------------------------------
    # NOTES:
    # [abc]: any character in the set {a, b, c}
    # [a-z]: any character in the range a to z
    # [^abc]: any character not in the set {a, b, c}
    # (): grouping
    # |: alternation (OR)
    # \d: any digit character (0-9)
    # \s: any whitespace character (e.g., space, tab, newline)
    # \S: any non-whitespace character
    # \\: a \ caracter
    # .: any character except a newline
    # \\.: a literal backslash followed by any non-newline character (used for escape sequences, e.g., \n, \t, \", \\, etc.)
    # ?: zero or one occurrence of the preceding pattern
    # *: zero or more occurrences of the preceding regex
    # *?: zero or more occurrences of the preceding regex, but as few as possible (non-greedy)
    # +: one or more occurrences of the preceding regex
    # +?: one or more occurrences of the preceding regex, but as few as possible (non-greedy)

    # String containing ignored characters (spaces and tabs)
    ignore = " \t"

    # Newlines
    @_(r"\n+")  # One or more consecutive newline characters.
    def ignore_newline(self, t):
        self.lineno += len(t.value)

    # Comments one line
    @_(r"//.*") # A double slash followed by any characters until the end of the line.
    def ignore_comment(self, t):
        self.lineno += t.value.count("\n")

    # Comments multiple lines
    @_(r"/\*[\s\S]*?\*/")   # /* followed by any characters (including newlines) until the first */
    def ignore_multiline_comment(self, t):
        self.lineno += t.value.count("\n")

    # Identifiers
    @_(r"[a-zA-Z_][a-zA-Z0-9_]*")   # An identifier starts with a letter or _, followed by any number of letters, digits, or _.
    def ID(self, t):
        # Check if the identifier is a reserved word.
        t.type = self.keywords.get(t.value, "ID")
        return t

    # Literals
    @_(r"\d+")  # Positive integers
    def INT_LITERAL(self, t):
        return t

    """
    A character literal is a single character enclosed in single quotes.
    The character can be any character except a backslash ('\'), single quote ('''), or newline ('\n').
    An escape sequence (a backslash followed by any character) is also allowed.
    """
    @_(r"\'([^\\\'\n]|(\\.))\'")
    def CHAR_LITERAL(self, t):
        return t

    """
    A string literal is a sequence of characters enclosed in double quotes.
    The sequence can be empty or contain any character except a backslash ('\'), double quote ('"'), or newline ('\n').
    An escape sequence (a backslash followed by any character) is also allowed.
    """
    @_(r"\"([^\\\"\n]|(\\.))*?\"")
    def STRING_LITERAL(self, t):
        return t

    # ----------------------------------------------------------------
    # Operators and punctuation (order matters: longer tokens first)
    # ----------------------------------------------------------------
    # Punctuation
    LPAREN = r"\("
    RPAREN = r"\)"
    LBRACE = r"\{"
    RBRACE = r"\}"
    LBRACKET = r"\["
    RBRACKET = r"\]"
    SEMI = r"\;"
    COMMA = r"\,"
    DOT = r"\."
    # Operators
    EQ     = r"=="
    NE     = r"!="
    LE     = r"<="
    GE     = r">="
    AND    = r"&&"
    OR     = r"\|\|"
    ASSIGN = r"="
    LT     = r"<"
    GT     = r">"
    PLUS   = r"\+"
    MINUS  = r"-"
    TIMES  = r"\*"
    DIVIDE = r"/"
    MOD    = r"%"
    NOT    = r"!"


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be scanned", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    def print_error(msg, x, y):
        # use stdout to match with the output in the .out test files
        print(f"Lexical error: {msg} at {x}:{y}", file=sys.stdout)

    # Create the lexer and set error function
    lexer = MJLexer(print_error)

    # open file and print tokens
    with open(input_path) as f:
        lexer.scan(f.read())


if __name__ == "__main__":
    main()
