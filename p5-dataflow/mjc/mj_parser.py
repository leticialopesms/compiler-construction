import argparse
import pathlib
import sys
from io import StringIO

from sly import Parser

from mjc.mj_ast import *
from mjc.mj_lexer import MJLexer


class Coord:
    """Coordinates of a syntactic element. Consists of:
    - Line number
    - (optional) column number, for the Lexer
    """

    __slots__ = ("line", "column")

    def __init__(self, line, column=None):
        self.line = line
        self.column = column

    def __str__(self):
        if self.line and self.column is not None:
            coord_str = "@ %s:%s" % (self.line, self.column)
        elif self.line:
            coord_str = "@ %s" % (self.line)
        else:
            coord_str = ""
        return coord_str


class ParserLogger:
    """Logger Class used to log messages about the parser in a text stream.
    NOTE: This class overrides the default SlyLogger class
    """

    def __init__(self):
        self.stream = StringIO()

    @property
    def text(self):
        return self.stream.getvalue()

    def debug(self, msg, *args, **kwargs):
        self.stream.write((msg % args) + "\n")

    info = debug

    def warning(self, msg, *args, **kwargs):
        self.stream.write("WARNING: " + (msg % args) + "\n")

    def error(self, msg, *args, **kwargs):
        self.stream.write("ERROR: " + (msg % args) + "\n")

    critical = debug


class MJParser(Parser):
    tokens = MJLexer.tokens
    start = "program"
    debugfile = "parser.debug"
    log = ParserLogger()

    def __init__(self, debug=True):
        """Create a new MJParser."""
        self.debug = debug
        self.mjlex = MJLexer(self._lexer_error)

        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def parse(self, text):
        self._last_yielded_token = None
        return super().parse(self.mjlex.tokenize(text))

    def _lexer_error(self, msg, line, column):
        # use stdout to match with the output in the .out test files
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    def _parser_error(self, msg, coord=None):
        # use stdout to match with the output in the .out test files
        if coord is None:
            print("ParserError: %s" % (msg), file=sys.stdout)
        else:
            print("ParserError: %s %s" % (msg, coord), file=sys.stdout)
        sys.exit(1)

    def _token_coord(self, p):
        last_cr = self.mjlex.text.rfind("\n", 0, p.index)
        if last_cr < 0:
            last_cr = -1
        column = p.index - (last_cr)
        return Coord(p.lineno, column)

    # Solve ambiguity
    precedence = (
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'EQ', 'NE'),
        ('left', 'LT', 'LE', 'GT', 'GE'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE', 'MOD'),
        ('right', 'NOT'),
    )

    # ------------------------------------------------------------
    # Parser Rules
    # ------------------------------------------------------------

    # Rule 1
    @_("")
    def empty(self, p):
        return None

    # Rule 2
    @_("ID")
    def identifier(self, p):
        id = ID(name=p.ID, coord=self._token_coord(p))
        return id

    # Rule 3
    @_("class_declaration_list")
    def program(self, p):
        # The root node of AST
        return Program(class_decls=p.class_declaration_list, coord=None)

    # Rule 4
    @_("class_declaration_list class_declaration")
    def class_declaration_list(self, p):
        p.class_declaration_list.append(p.class_declaration)
        return p.class_declaration_list

    # Rule 5
    @_("class_declaration")
    def class_declaration_list(self, p):
        return [p.class_declaration]

    # Rule 6
    @_("CLASS identifier extends_opt LBRACE compound_declarations method_declarations RBRACE")
    def class_declaration(self, p):
        return ClassDecl(
            name=p.identifier,
            extends=p.extends_opt,
            var_decls=p.compound_declarations,
            method_decls=p.method_declarations,
            coord=self._token_coord(p),
        )

    # Rule 7
    @_("empty")
    def extends_opt(self, p):
        return None

    # Rule 8
    @_("EXTENDS identifier")
    def extends_opt(self, p):
        return p.identifier

    # Rule 9
    @_("compound_declarations compound_declaration")
    def compound_declarations(self, p):
        compound_declarations = p.compound_declarations + p.compound_declaration
        return compound_declarations

    # Rule 10
    @_("empty")
    def compound_declarations(self, p):
        return []

    # Rule 11
    @_("type_specifier init_declarator_list SEMI")
    def compound_declaration(self, p):
        result = []
        for var_decl in p.init_declarator_list:
            # If the init_declarator returned a VarDecl (with init), keep it and set its type.
            if isinstance(var_decl, VarDecl):
                var_decl.type = p.type_specifier
                result.append(var_decl)
            else:
                # If it returned an ID (no initializer), wrap it into a VarDecl
                # using the ID's coord for the declaration coord.
                wrapped = VarDecl(
                    type=p.type_specifier,
                    name=var_decl,
                    init=None,
                    coord=(var_decl.coord if hasattr(var_decl, "coord") else self._token_coord(p)),
                )
                result.append(wrapped)
        return result

    # Rule 12
    @_("method_declarations method_declaration")
    def method_declarations(self, p):
        p.method_declarations.append(p.method_declaration)
        return p.method_declarations

    # Rule 13
    @_("empty")
    def method_declarations(self, p):
        return []

    # Rule 14
    @_("main_method_declaration")
    def method_declaration(self, p):
        return p.main_method_declaration

    # Rule 15
    @_("regular_method_declaration")
    def method_declaration(self, p):
        return p.regular_method_declaration

    # Rule 16
    @_("PUBLIC type_specifier identifier LPAREN parameter_list_opt RPAREN compound_statement")
    def regular_method_declaration(self, p):
        return MethodDecl(
            type=p.type_specifier,
            name=p.identifier,
            param_list=p.parameter_list_opt,
            body=p.compound_statement,
            coord=self._token_coord(p),
        )

    # Rule 17
    @_("PUBLIC STATIC VOID MAIN LPAREN STRING LBRACKET RBRACKET identifier RPAREN compound_statement")
    def main_method_declaration(self, p):
        main_decl =  MainMethodDecl(
            args=p.identifier,
            body=p.compound_statement,
            coord=self._token_coord(p),
        )
        return main_decl

    # Rule 18
    @_("empty")
    def parameter_list_opt(self, p):
        return None

    # Rule 19
    @_("parameter_list")
    def parameter_list_opt(self, p):
        return ParamList(params=p.parameter_list, coord=None)

    # Rule 20
    @_("parameter_list COMMA parameter_declaration")
    def parameter_list(self, p):
        p.parameter_list.append(p.parameter_declaration)
        return p.parameter_list

    # Rule 21
    @_("parameter_declaration")
    def parameter_list(self, p):
        return [p.parameter_declaration]

    # Rule 22
    @_("type_specifier identifier")
    def parameter_declaration(self, p):
        return ParamDecl(
            type=p.type_specifier,
            name=p.identifier,
            coord=self._token_coord(p),
        )

    # Rule 23
    @_("init_declarator_list COMMA init_declarator")
    def init_declarator_list(self, p):
        p.init_declarator_list.append(p.init_declarator)
        return p.init_declarator_list

    # Rule 24
    @_("init_declarator")
    def init_declarator_list(self, p):
        return [p.init_declarator]

    # Rule 25
    @_("declarator ASSIGN initializer")
    def init_declarator(self, p):
        vardecl = VarDecl(
                type=None,
                name=p.declarator,
                init=p.initializer,
                coord=self._token_coord(p),
        )
        return vardecl

    # Rule 26
    @_("declarator")
    def init_declarator(self, p):
        return p[0]

    # Rule 27
    @_("LBRACE initializer_list COMMA RBRACE")
    def initializer(self, p):
        coord = p.initializer_list[0].coord if p.initializer_list else self._token_coord(p)
        return InitList(exprs=p.initializer_list, coord=coord)

    # Rule 28
    @_("LBRACE initializer_list_opt RBRACE")
    def initializer(self, p):
        coord = p.initializer_list_opt[0].coord if p.initializer_list_opt else self._token_coord(p)
        return InitList(exprs=p.initializer_list_opt, coord=coord)

    # Rule 29
    @_("assignment_expression")
    def initializer(self, p):
        return p.assignment_expression

    # Rule 30
    @_("initializer_list")
    def initializer_list_opt(self, p):
        return p.initializer_list

    # Rule 31
    @_("empty")
    def initializer_list_opt(self, p):
        return []

    # Rule 32
    @_("initializer_list COMMA initializer")
    def initializer_list(self, p):
        p.initializer_list.append(p.initializer)
        return p.initializer_list

    # Rule 33
    @_("initializer")
    def initializer_list(self, p):
        return [p.initializer]

    # Rule 34
    @_("declarator LBRACKET RBRACKET")
    def declarator(self, p):
        return p.declarator

    # Rule 35
    @_("LPAREN declarator RPAREN")
    def declarator(self, p):
        return p.declarator

    # Rule 36
    @_("identifier")
    def declarator(self, p):
        return p.identifier

    # Rule 37
    @_("identifier")
    def type_specifier(self, p):
        type = Type(name=p.identifier, coord=self._token_coord(p))
        return type

    # Rule 38
    @_("INT LBRACKET RBRACKET")
    def type_specifier(self, p):
        return Type(name="int[]", coord=self._token_coord(p))

    # Rule 39
    @_("CHAR LBRACKET RBRACKET")
    def type_specifier(self, p):
        return Type(name="char[]", coord=self._token_coord(p))

    # Rules 40 - 44
    @_("INT", "CHAR", "BOOLEAN", "VOID", "STRING")
    def type_specifier(self, p):
        return Type(name=p[0], coord=self._token_coord(p))

    # Rule 45
    @_("expression COMMA assignment_expression")
    def expression(self, p):
        expr_list = p.expression
        if not isinstance(expr_list, ExprList):
            expr_list = ExprList(exprs=[p.expression], coord=p.expression.coord)
        expr_list.exprs.append(p.assignment_expression)
        return expr_list

    # Rule 46
    @_("assignment_expression")
    def expression(self, p):
        return p.assignment_expression

    # Rule 47
    @_("unary_expression ASSIGN assignment_expression")
    def assignment_expression(self, p):
        return Assignment(
            op="=",
            lvalue=p.unary_expression,
            rvalue=p.assignment_expression,
            coord=self._token_coord(p),
        )

        # Rule 48
    @_("binary_expression")
    def assignment_expression(self, p):
        return p.binary_expression

        # Rules 49 - 61
    @_(
            "binary_expression OR binary_expression",     # Rule 49
            "binary_expression AND binary_expression",    # Rule 50
            "binary_expression EQ binary_expression",     # Rule 51
            "binary_expression NE binary_expression",     # Rule 52
            "binary_expression LT binary_expression",     # Rule 53
            "binary_expression LE binary_expression",     # Rule 54
            "binary_expression GT binary_expression",     # Rule 55
            "binary_expression GE binary_expression",     # Rule 56
            "binary_expression PLUS binary_expression",   # Rule 57
            "binary_expression MINUS binary_expression",  # Rule 58
            "binary_expression DIVIDE binary_expression", # Rule 59
            "binary_expression MOD binary_expression",    # Rule 60
            "binary_expression TIMES binary_expression"   # Rule 61
    )
    def binary_expression(self, p):
        """
        Some functions, like this one, may handle multiple production rules.
        This is useful if different rules should trigger the same semantic action.
        In this case, regardless of the operator, we wish to generate the same tuple,
        so the same semantic action is used.
        Each element in the production is a attribute in p
        expr    PLUS    expr
        p[0]    p[1]    p[2]
        """
        # return (p[0], p[1], p[2], str(self._token_coord(p)))
        return BinaryOp(op=p[1], left=p[0], right=p[2], coord=self._token_coord(p))

    # Rule 62
    @_("unary_expression")
    def binary_expression(self, p):
        return p.unary_expression

    # Rule 63
    @_("unary_operator unary_expression")
    def unary_expression(self, p):
        return UnaryOp(op=p[0], expr=p[1], coord=self._token_coord(p))

    # Rule 64
    @_("postfix_expression")
    def unary_expression(self, p):
        return p.postfix_expression

    # Rules 65-67
    @_(
        "NOT",      # Rule 65
        "MINUS",    # Rule 66
        "PLUS"      # Rule 67
    )
    def unary_operator(self, p):
        return p[0]

    # Rule 68
    @_("postfix_expression LBRACKET expression RBRACKET")
    def postfix_expression(self, p):
        return ArrayRef(
            name=p.postfix_expression,
            subscript=p.expression,
            coord=self._token_coord(p)
        )

    # Rule 69
    @_("postfix_expression DOT LENGTH")
    def postfix_expression(self, p):
        return Length(
            expr=p.postfix_expression,
            coord=self._token_coord(p)
        )
    # Rule 70
    @_("postfix_expression DOT identifier LPAREN argument_expression_opt RPAREN")
    def postfix_expression(self, p):
        return MethodCall(
            object=p.postfix_expression,
            # p.identifier is already an AST `ID` node (created by the identifier rule).
            # Use it directly as the method_name and its stored coord.
            method_name=p.identifier,
            args=p.argument_expression_opt,
            coord=self._token_coord(p)
        )

    # Rule 71
    @_("postfix_expression DOT identifier")
    def postfix_expression(self, p):
        field_access = FieldAccess(
            object=p.postfix_expression,
            field_name=p.identifier,
            coord=self._token_coord(p)
        )
        return field_access

    # Rule 72
    @_("primary_expression")
    def postfix_expression(self, p):
        return p.primary_expression

    # Rule 73
    @_("LPAREN expression RPAREN")
    def primary_expression(self, p):
        return p.expression

    # Rule 74
    @_("new_expression")
    def primary_expression(self, p):
        return p.new_expression

    # Rule 75
    @_("this_expression")
    def primary_expression(self, p):
        return p.this_expression

    # Rule 76
    @_("constant")
    def primary_expression(self, p):
        return p.constant

    # Rule 77
    @_("identifier")
    def primary_expression(self, p):
        return p.identifier

    # Rule 78
    @_("empty")
    def argument_expression_opt(self, p):
        return None

    # Rule 79
    @_("argument_expression")
    def argument_expression_opt(self, p):
        exprs = p.argument_expression
        if isinstance(exprs, list):
            lst = exprs
            return ExprList(exprs=lst, coord=self._token_coord(p))
        else:
            return p.argument_expression

    # Rule 80
    @_("argument_expression COMMA assignment_expression")
    def argument_expression(self, p):
        # p.argument_expression may be a single Expr or a list; normalize to list
        if isinstance(p.argument_expression, list):
            return p.argument_expression + [p.assignment_expression]
        else:
            return [p.argument_expression, p.assignment_expression]

    # Rule 81
    @_("assignment_expression")
    def argument_expression(self, p):
        return p.assignment_expression

    # Rule 82
    @_("THIS")
    def this_expression(self, p):
        return This(coord=self._token_coord(p))

    # Rule 83
    @_("NEW identifier LPAREN RPAREN")
    def new_expression(self, p):
        type = Type(name=p.identifier, coord=p.identifier.coord)
        newobj = NewObject(type=type, coord=self._token_coord(p))
        return newobj

    # Rule 84
    # TODO: Correct the coord of the arr_size
    # The current coord corresponds to the 'NEW' token, but aparently it should correspond to the 'INT_LITERAL' token
    @_("NEW INT LBRACKET constant RBRACKET")
    def new_expression(self, p):
        arr_type = Type(name="int[]", coord=None)
        arr_size = p.constant
        return NewArray(type=arr_type, size=arr_size, coord=self._token_coord(p))

    # Rule 85
    # TODO: Correct the coord of the arr_size
    # The current coord corresponds to the 'NEW' token, but aparently it should correspond to the 'INT_LITERAL' token
    @_("NEW CHAR LBRACKET constant RBRACKET")
    def new_expression(self, p):
        arr_type = Type(name="char[]", coord=None)
        arr_size = p.constant
        return NewArray(type=arr_type, size=arr_size, coord=self._token_coord(p))

    # Rule 86
    @_("STRING_LITERAL")
    def constant(self, p):
        return Constant(type="String", value=p.STRING_LITERAL, coord=self._token_coord(p))

    # Rule 87
    @_("INT_LITERAL")
    def constant(self, p):
        return Constant(type="int", value=p.INT_LITERAL, coord=self._token_coord(p))

    # Rule 88
    @_("CHAR_LITERAL")
    def constant(self, p):
        return Constant(type="char", value=p.CHAR_LITERAL, coord=self._token_coord(p))

    # Rule 89
    @_("boolean_literal")
    def constant(self, p):
        # return Constant(type="boolean", value=p.boolean_literal, coord=self._token_coord(p))
        return p.boolean_literal

    # Rule 90
    @_("FALSE")
    def boolean_literal(self, p):
        # return False
        return Constant(type="boolean", value=p.FALSE, coord=self._token_coord(p))

    # Rule 91
    @_("TRUE")
    def boolean_literal(self, p):
        # return True
        return Constant(type="boolean", value=p.TRUE, coord=self._token_coord(p))

    # Rule 92
    @_("statement_list statement")
    def statement_list(self, p):
        return p.statement_list + [p.statement]

    # Rule 93
    @_("statement")
    def statement_list(self, p):
        return [p.statement]

    # Rules 94 - 101
    # General statement nonterminal: all concrete statements map to a
    # single `statement` nonterminal that returns an instance of
    # `Statement` (one of its concrete subclasses from mj_ast).
    @_(
        "assert_statement",     # Rule 96
        "print_statement",      # Rule 95
        "jump_statement",       # Rule 94
        "for_statement",        # Rule 97
        "while_statement",      # Rule 98
        "if_statement",         # Rule 99
        "compound_statement",   # Rule 101
        "expression_statement", # Rule 100
    )
    def statement(self, p):
        # every alternative places the concrete Statement node at p[0]
        return p[0]

    # Rule 102
    @_("LBRACE compound_body RBRACE")
    def compound_statement(self, p):
        # return p.compound_body
        compound = Compound(
            statements=p.compound_body,
            coord=self._token_coord(p),
        )
        return compound

    # Rule 103
    @_("compound_declarations statement_opt")
    def compound_body(self, p):
        # compound = Compound(
        #     statements=p.compound_declarations + p.statement_opt,
        #     coord=self._token_coord(p),
        # )
        # return compound
        return p.compound_declarations + p.statement_opt

    # Rule 104
    @_("statement_list")
    def statement_opt(self, p):
        return p.statement_list

    # Rule 105
    @_("empty")
    def statement_opt(self, p):
        return []

    # Rule 106
    @_("expression SEMI")
    def expression_statement(self, p):
        return p.expression

    # Rule 107
    @_("IF LPAREN expression RPAREN statement ELSE statement")
    def if_statement(self, p):
        return If(cond=p[2], iftrue=p[4], iffalse=p[6], coord=self._token_coord(p))

    # Rule 108
    @_("IF LPAREN expression RPAREN statement")
    def if_statement(self, p):
        return If(cond=p[2], iftrue=p[4], iffalse=None, coord=self._token_coord(p))

    # Rule 109
    @_("WHILE LPAREN expression RPAREN statement")
    def while_statement(self, p):
        return While(cond=p[2], body=p[4], coord=self._token_coord(p))

    # Rule 110
    @_("FOR LPAREN compound_declaration expression_opt SEMI expression_opt RPAREN statement")
    def for_statement(self, p):
        # p[2] is a list of VarDecl returned by compound_declaration
        init_decls = p[2]
        init = DeclList(init_decls, self._token_coord(p)) if init_decls else None
        cond = p[3] # expression_opt
        step = p[5] # expression_opt
        body = p[7] # statement
        return For(init=init, cond=cond, next=step, body=body, coord=self._token_coord(p))

    # Rule 111
    @_("FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement")
    def for_statement(self, p):
        init_expr = p[2]
        init = init_expr
        cond = p[4]
        step = p[6]
        body = p[8]
        return For(init=init, cond=cond, next=step, body=body, coord=self._token_coord(p))

    # Rule 112
    @_("expression")
    def expression_opt(self, p):
        return p.expression

    # Rule 113
    @_("empty")
    def expression_opt(self, p):
        return None

    # Rule 114
    @_("ASSERT expression SEMI")
    def assert_statement(self, p):
        return Assert(expr=p.expression, coord=self._token_coord(p))

    # Rule 115
    @_("PRINT LPAREN expression_opt RPAREN SEMI")
    def print_statement(self, p):
        return Print(expr=p.expression_opt, coord=self._token_coord(p))

    # Rule 116
    @_("RETURN SEMI")
    def jump_statement(self, p):
        return Return(expr=None, coord=self._token_coord(p))

    # Rule 117
    @_("RETURN expression SEMI")
    def jump_statement(self, p):
        return Return(expr=p.expression, coord=self._token_coord(p))

    # Rule 118
    @_("BREAK SEMI")
    def jump_statement(self, p):
        return Break(coord=self._token_coord(p))


    # ------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------
    def error(self, p):
        if p:
            self._parser_error(
                "Before %s" % p.value, Coord(p.lineno, self.mjlex.find_tok_column(p))
            )
        else:
            self._parser_error("At the end of input (%s)" % self.mjlex.filename)


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be parsed", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("ERROR: Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    parser = MJParser()

    # open file and print ast
    with open(input_path) as f:
        ast = parser.parse(f.read())
        print(parser.log.text)
        ast.show(buf=sys.stdout, showcoord=True)
    
#     # Teste
#     parser = MJParser(debug=True)
#     # exp = "a =3 * 4 + 5;"
#     exp = """
# class Example1 {
#   int a = 3 * 4 + 5, c;
#   int[] b = new int[10];
# }
# """
#     ast = parser.parse(exp)
#     ast.show(buf=sys.stdout, showcoord=True)


if __name__ == "__main__":
    main()
