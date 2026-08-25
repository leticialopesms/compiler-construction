import argparse
import pathlib
import sys
from copy import deepcopy
from typing import Any, Dict, Union

from mjc.mj_ast import *
from mjc.mj_parser import MJParser
from mjc.mj_serror import SE, assert_semantic
from mjc.mj_type import (
    BooleanType,
    CharArrayType,
    CharType,
    IntArrayType,
    IntType,
    MJType,
    ObjectType,
    StringType,
    VoidType,
)


def _type_name(type_node) -> str:
    """Return the string name of a Type node.

    Primitive types (int, char, …) are stored as str; user-defined class types
    are stored as an ID AST node — this helper normalises both cases.
    """
    name = type_node.name
    return name.name if isinstance(name, ID) else name


class ClassMetaData:
    """Class metadata storage.
    
    Stores information about a class including its fields and methods.
    """
    
    def __init__(self, name: str):
        """Initialize class metadata.
        
        :param name: the class name
        """
        self.name = name
        self.fields = {}  # field_name -> type_name
        self.methods = {}  # method_name -> (return_type, [param_names], [param_types])
        self.extends = None  # parent class name


class SymbolTable:
    """Class representing a symbol table.

    `add` and `lookup` methods are given, however you still need to find a way to
    deal with scopes.

    ## Attributes
    :data: the content of the SymbolTable
    """

    def __init__(self) -> None:
        """Initializes the SymbolTable."""
        self.__data = dict()

    @property
    def data(self) -> Dict[str, Any]:
        """Returns a copy of the SymbolTable."""
        return deepcopy(self.__data)

    def add(self, name: str, value: Any) -> None:
        """Adds to the SymbolTable.

        :param name: the identifier on the SymbolTable
        :param value: the value to assign to the given `name`
        """
        self.__data[name] = value

    def lookup(self, name: str) -> Union[Any, None]:
        """Searches `name` on the SymbolTable and returns the value
        assigned to it.

        :param name: the identifier that will be searched on the SymbolTable
        :return: the value assigned to `name` on the SymbolTable. If `name` is not found, `None` is returned.
        """
        return self.__data.get(name)


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class SymbolTableBuilder(NodeVisitor):
    """Symbol Table Builder class.
    This class build the Symbol table of the program by visiting all the AST nodes
    using the visitor pattern.
    """

    def __init__(self):
        self.global_symtab = SymbolTable()
        self.typemap = {
            "boolean": BooleanType,
            "char": CharType,
            "int": IntType,
            "String": StringType,
            "void": VoidType,
            "int[]": IntArrayType,
            "char[]": CharArrayType,
            "object": ObjectType,
        }

    def visit_Program(self, node: Program):
        """Visit the program node to fill in the global symbol table"""
        # First, register all classes in the program.
        # Populating the global symbol table with these classes
        for class_decl in node.class_decls:
            # Check if the class has already been declared
            class_name = class_decl.name.name
            assert_semantic(
                condition=(self.global_symtab.lookup(class_name) is None),
                error_type=SE.ALREADY_DECLARED_CLASS,
                coord=class_decl.coord,
                name=class_name,
            )
            
            # Create class metadata and register in global symbol table
            class_meta = ClassMetaData(name=class_name)
            self.global_symtab.add(class_name, class_meta)

        # Now, process each class to fill in fields and methods
        for class_decl in node.class_decls:
            self.visit(class_decl)

        # Finally, return the global symtab to use in the next steps
        return self.global_symtab

    def visit_ClassDecl(self, node: ClassDecl):
        # Set the current class to ensure the context for internal visits
        self.current_class = self.global_symtab.lookup(node.name.name)

        # First, if the class extends another, check that the parent exists.
        if node.extends is not None:
            parent_name = node.extends.name
            assert_semantic(
                condition=(self.global_symtab.lookup(parent_name) is not None),
                error_type=SE.UNDECLARED_CLASS,
                coord=node.coord,
                name=parent_name,
            )
            # Record the parent class in the current class metadata
            self.current_class.extends = parent_name

        # Then, visit all fields (var_decls) of the class
        for field in node.var_decls:
            self.visit(field)

        # Finally, visit all class methods (method_decls)
        for method in node.method_decls:
            self.visit(method)

        # Unset the current class context
        self.current_class = None

    def visit_VarDecl(self, node: VarDecl):
        # First, check if the field has already been declared
        field_name = node.name.name
        assert_semantic(
            condition=(field_name not in self.current_class.fields),
            error_type=SE.ALREADY_DECLARED_FIELD,
            coord=node.coord,
            name=field_name,
        )
        
        # Record the field and its type
        self.current_class.fields[field_name] = _type_name(node.type)

    def visit_MethodDecl(self, node: MethodDecl):
        # First, check if the method has already been declared
        method_name = node.name.name
        assert_semantic(
            condition=(method_name not in self.current_class.methods),
            error_type=SE.ALREADY_DECLARED_METHOD,
            coord=node.coord,
            name=method_name,
        )
        
        # Gather parameter types
        param_names = []
        param_types = []
        if node.param_list is not None:
            for param in node.param_list.params:
                param_names.append(param.name.name)
                param_type = self.typemap.get(_type_name(param.type))
                param_types.append(param_type)

        # Record the method and its signature (return_type, parameter_types)
        return_type = self.typemap.get(_type_name(node.type))
        self.current_class.methods[method_name] = (return_type, param_names, param_types)

    def visit_MainMethodDecl(self, node: MainMethodDecl):
        # The main method must have the name "main"
        main_method_name = "main"
        # First, check if the main method has already been declared
        assert_semantic(
            condition=(main_method_name not in self.current_class.methods),
            error_type=SE.ALREADY_DECLARED_METHOD,
            coord=node.coord,
            name=main_method_name,
        )
        
        # Record the main method and its signature (return_type="void", no parameters)
        self.current_class.methods[main_method_name] = (VoidType, [], [])


class SemanticScope:
    """
    Represents a scope in the semantic analysis phase.
    We implement as a stack of SymbolTables, where each SymbolTable represents a scope level.
    """
    def __init__(self):
        self.scope_stack = []

    def enter(self):
        """Enter a new scope level."""
        self.scope_stack.append(SymbolTable())

    def exit(self):
        """Exit the current scope level."""
        if self.scope_stack:
            self.scope_stack.pop()

    def add(self, name: str, value: Any) -> None:
        """Add a name-value pair to the current scope."""
        if self.scope_stack:
            self.scope_stack[-1].add(name, value)

    def lookup(self, name: str) -> Union[Any, None]:
        """Look up a name in the current scope stack, starting from the innermost scope."""
        for scope in reversed(self.scope_stack):
            value = scope.lookup(name)
            if value is not None:
                return value
        return None


class SemanticAnalyzer(NodeVisitor):
    """Semantic Analyzer class.
    This class performs semantic analysis on the AST of a MiniJava program.
    You need to define methods of the form visit_NodeName()
    for each kind of AST node that you want to process.
    """

    def __init__(self, global_symtab: SymbolTable):
        """
        :param global_symtab: Global symbol table with all class declaration metadata.
        """
        self.scope = SemanticScope()
        self.current_class = None
        self.in_loop = False
        self.current_return_type = None
        self.global_symtab = global_symtab
        self._var_class_names = {}
        self.typemap = {
            "boolean": BooleanType,
            "char": CharType,
            "int": IntType,
            "String": StringType,
            "void": VoidType,
            "int[]": IntArrayType,
            "char[]": CharArrayType,
            "object": ObjectType,
        }

    def visit_Program(self, node: Program):
        # Visit all class declarations in the program
        for cls in node.class_decls:
            self.visit(cls)

    def visit_ClassDecl(self, node: ClassDecl):
        self.current_class = node.name.name
        # Visit the fields of the class (var_decls)
        for field in node.var_decls:
            self.visit(field)

        # Then, visit the methods of the class (method_decls)
        for method in node.method_decls:
            self.visit(method)

        self.current_class = None

    def visit_VarDecl(self, node: VarDecl):
        type_name = _type_name(node.type)
        var_type = self.typemap.get(type_name)

        # pode ser tipo primitivo ou classe
        if var_type is None:
            class_info = self.global_symtab.lookup(type_name)

            assert_semantic(
                condition=(class_info is not None),
                error_type=SE.UNDECLARED_CLASS,
                coord=node.coord,
                name=type_name,
            )

            var_type = ObjectType
            self._var_class_names[node.name.name] = type_name

        # tipo no nó
        node.mj_type = var_type

        if self.scope.scope_stack:
            current_scope = self.scope.scope_stack[-1]
            assert_semantic(
                condition=(current_scope.lookup(node.name.name) is None),
                error_type=SE.ALREADY_DECLARED_NAME,
                coord=node.coord,
                name=node.name.name,
            )

        self.scope.add(node.name.name, var_type)

        # se houver inicialização, validar compatibilidade
        if node.init is not None:
            self.visit(node.init)

            init_type = node.init.mj_type
            compatible = (init_type == var_type) or (
                var_type == CharArrayType and init_type == StringType
            )
            assert_semantic(
                condition=compatible,
                error_type=SE.ASSIGN_TYPE_MISMATCH,
                coord=node.coord,
                ltype=var_type,
                rtype=init_type,
            )

    def visit_MethodDecl(self, node: MethodDecl):
        type_name = _type_name(node.type)
        return_type = self.typemap.get(type_name)

        if return_type is None:
            class_info = self.global_symtab.lookup(type_name)

            assert_semantic(
                condition=(class_info is not None),
                error_type=SE.UNDECLARED_CLASS,
                coord=node.coord,
                name=type_name,
            )

            return_type = ObjectType

        node.mj_type = return_type

        self.current_return_type = return_type

        self.scope.enter()

        if node.param_list is not None:
            self.visit(node.param_list)

        if node.body is not None:
            self.visit(node.body)

        self.scope.exit()

        self.current_return_type = None

    def visit_MainMethodDecl(self, node: MainMethodDecl):
        # main always returns void
        self.current_return_type = VoidType

        if node.body is not None:
            self.visit(node.body)

        self.current_return_type = None

    def visit_ParamList(self, node: ParamList):
        for param in node.params or []:
            self.visit(param)

    def visit_ParamDecl(self, node: ParamDecl):
        type_name = _type_name(node.type)
        param_type = self.typemap.get(type_name)

        if param_type is None:
            class_info = self.global_symtab.lookup(type_name)

            assert_semantic(
                condition=(class_info is not None),
                error_type=SE.UNDECLARED_CLASS,
                coord=node.coord,
                name=type_name,
            )

            param_type = ObjectType

        node.mj_type = param_type

        param_name = node.name.name
        assert_semantic(
            condition=(self.scope.scope_stack[-1].lookup(param_name) is None),
            error_type=SE.PARAMETER_ALREADY_DECLARED,
            coord=node.coord,
            name=param_name,
        )
        self.scope.add(param_name, param_type)

    def visit_Compound(self, node: Compound):
        self.scope.enter()

        for stmt in node.statements or []:
            self.visit(stmt)

        self.scope.exit()

    def visit_If(self, node: If):
        self.visit(node.cond)

        cond_type = node.cond.mj_type

        assert_semantic(
            condition=(cond_type == BooleanType),
            error_type=SE.CONDITIONAL_EXPRESSION_TYPE_MISMATCH,
            coord=node.coord,
            ltype=cond_type,
        )

        self.scope.enter()
        self.visit(node.iftrue)
        self.scope.exit()

        if node.iffalse is not None:
            self.scope.enter()
            self.visit(node.iffalse)
            self.scope.exit()

    def visit_While(self, node: While):
        self.visit(node.cond)

        cond_type = node.cond.mj_type

        assert_semantic(
            condition=(cond_type == BooleanType),
            error_type=SE.CONDITIONAL_EXPRESSION_TYPE_MISMATCH,
            coord=node.coord,
            ltype=cond_type,
        )

        old_in_loop = getattr(self, "in_loop", False)
        self.in_loop = True

        self.scope.enter()
        self.visit(node.body)
        self.scope.exit()
        self.in_loop = old_in_loop

    def visit_For(self, node: For):
        old_in_loop = getattr(self, "in_loop", False)
        self.in_loop = True

        self.scope.enter()

        if node.init is not None:
            self.visit(node.init)

        if node.cond is not None:
            self.visit(node.cond)

            cond_type = node.cond.mj_type

            assert_semantic(
                condition=(cond_type == BooleanType),
                error_type=SE.CONDITIONAL_EXPRESSION_TYPE_MISMATCH,
                coord=node.coord,
                ltype=cond_type,
            )

        if node.body is not None:
            self.visit(node.body)

        if node.next is not None:
            self.visit(node.next)

        self.scope.exit()

        self.in_loop = old_in_loop

    def visit_DeclList(self, node: DeclList):
        for decl in node.decls or []:
            self.visit(decl)

    def visit_Print(self, node: Print):
        # print() with no arguments is valid
        if node.expr is None:
            return

        self.visit(node.expr)

        def is_valid_print_type(t):
            return t in (IntType, CharType, StringType)

        if hasattr(node.expr, "exprs"): 
            for expr in node.expr.exprs:
                expr_type = expr.mj_type

                assert_semantic(
                    condition=is_valid_print_type(expr_type),
                    error_type=SE.PRINT_EXPRESSION_TYPE_MISMATCH,
                    coord=node.coord,
                )
        else:
            expr_type = node.expr.mj_type

            assert_semantic(
                condition=is_valid_print_type(expr_type),
                error_type=SE.PRINT_EXPRESSION_TYPE_MISMATCH,
                coord=node.coord,
            )

    def visit_Assert(self, node: Assert):
        self.visit(node.expr)

        expr_type = node.expr.mj_type

        assert_semantic(
            condition=(expr_type == BooleanType),
            error_type=SE.ASSERT_EXPRESSION_TYPE_MISMATCH,
            coord=node.coord,
        )

    def visit_Break(self, node: Break):
        assert_semantic(
            condition=getattr(self, "in_loop", False),
            error_type=SE.WRONG_BREAK_STATEMENT,
            coord=node.coord,
        )

    def visit_Return(self, node: Return):
        if node.expr is not None:
            self.visit(node.expr)
            return_type = node.expr.mj_type
        else:
            return_type = VoidType

        # Check that the returned type matches the method's declared return type
        assert_semantic(
            condition=(return_type == self.current_return_type),
            error_type=SE.RETURN_TYPE_MISMATCH,
            coord=node.coord,
            ltype=return_type,
            rtype=self.current_return_type,
        )

    def visit_Assignment(self, node: Assignment):
        # Visit the right side
        self.visit(node.rvalue)

        # Visit the left side
        self.visit(node.lvalue)

        # Check if the name is defined
        if isinstance(node.lvalue, ID):
            assert_semantic(
                condition=(node.lvalue.scope is not None),
                error_type=SE.UNDECLARED_NAME,
                coord=node.coord,
                name=node.lvalue.name,
            )

        # Check if the assignment is allowed - types must match
        lvalue_type = node.lvalue.mj_type
        rvalue_type = node.rvalue.mj_type
        compatible = (lvalue_type == rvalue_type) or (
            lvalue_type == CharArrayType and rvalue_type == StringType
        )
        assert_semantic(
            condition=compatible,
            error_type=SE.ASSIGN_TYPE_MISMATCH,
            coord=node.coord,
            ltype=lvalue_type,
            rtype=rvalue_type,
        )

    def visit_BinaryOp(self, node: BinaryOp):
        # Visit the left expression
        self.visit(node.lvalue)
        # Visit the right expression
        self.visit(node.rvalue)
        # Check if left and right operands have the same type
        ltype = node.lvalue.mj_type
        rtype = node.rvalue.mj_type
        assert_semantic(
            condition=(ltype == rtype),
            error_type=SE.BINARY_EXPRESSION_TYPE_MISMATCH,
            coord=node.coord,
            name=node.op,
            ltype=ltype,
            rtype=rtype,
        )
        # Check if the operation is supported by the type
        is_rel_op = node.op in ltype.rel_ops
        is_bin_op = node.op in ltype.binary_ops
        assert_semantic(
            condition=(is_rel_op or is_bin_op),
            error_type=SE.UNSUPPORTED_BINARY_OPERATION,
            coord=node.coord,
            name=node.op,
            ltype=ltype,
        )
        # Relational ops produce bool; arithmetic/logical ops preserve the operand type
        node.mj_type = BooleanType if is_rel_op else ltype

    def visit_UnaryOp(self, node: UnaryOp):
        # Visit the operand expression
        self.visit(node.expr)

        expr_type = node.expr.mj_type

        # Check if the operator is supported by the operand's type
        assert_semantic(
            condition=(node.op in expr_type.unary_ops),
            error_type=SE.UNSUPPORTED_UNARY_OPERATION,
            coord=node.coord,
            name=node.op,
        )

        # The result type is the same as the operand type
        node.mj_type = expr_type

    def visit_ArrayRef(self, node: ArrayRef):
        # Visit the array expression and the subscript index
        self.visit(node.name)
        self.visit(node.subscript)

        array_type = node.name.mj_type
        subscript_type = node.subscript.mj_type

        # Check that the target is an array type
        assert_semantic(
            condition=(array_type in (IntArrayType, CharArrayType)),
            error_type=SE.ARRAY_REF_TYPE_MISMATCH,
            coord=node.coord,
            ltype=array_type,
        )

        # Check that the subscript is an integer
        assert_semantic(
            condition=(subscript_type == IntType),
            error_type=SE.ARRAY_DIMENSION_MISMATCH,
            coord=node.coord,
            ltype=subscript_type,
        )

        # The result type is the element type of the array (int or char)
        array_to_element_type = {
            IntArrayType: IntType,
            CharArrayType: CharType,
        }
        # node.mj_type = array_type.element_type
        node.mj_type = array_to_element_type.get(array_type, ObjectType)

    def visit_FieldAccess(self, node: FieldAccess):
        # Visit the object expression
        self.visit(node.object)

        obj_type = node.object.mj_type

        # The object must be of class (object) type
        assert_semantic(
            condition=(obj_type == ObjectType),
            error_type=SE.OBJECT_TYPE_MUST_BE_A_CLASS,
            coord=node.coord,
            name=getattr(node.object, "name", node.field_name.name),
        )

        # Get the specific class name from the object node
        class_name = getattr(node.object, "class_name", None)

        if class_name is not None:
            field_name = node.field_name.name
            
            # Search for field in class hierarchy (including parent classes)
            field_type_name = None
            current_class = self.global_symtab.lookup(class_name)
            while current_class is not None:
                if field_name in current_class.fields:
                    field_type_name = current_class.fields[field_name]
                    break
                # Move to parent class if it exists
                if current_class.extends is not None:
                    current_class = self.global_symtab.lookup(current_class.extends)
                else:
                    current_class = None

            assert_semantic(
                condition=(field_type_name is not None),
                error_type=SE.UNDECLARED_FIELD,
                coord=node.coord,
                name=field_name,
            )

            node.mj_type = self.typemap.get(field_type_name, ObjectType)
        else:
            node.mj_type = ObjectType

    def visit_MethodCall(self, node: MethodCall):
        # Visit the object expression
        self.visit(node.object)

        obj_type = node.object.mj_type

        # The object must be of class (object) type
        assert_semantic(
            condition=(obj_type == ObjectType),
            error_type=SE.OBJECT_TYPE_MUST_BE_A_CLASS,
            coord=node.coord,
            name=getattr(node.object, "name", node.method_name.name),
        )

        # Visit the argument list
        if node.args is not None:
            self.visit(node.args)
            # If args is an ExprList, get the list; otherwise wrap the single expression
            if hasattr(node.args, "exprs"):
                arg_exprs = node.args.exprs
            else:
                arg_exprs = [node.args]
        else:
            arg_exprs = []

        # Get the specific class name from the object node
        class_name = getattr(node.object, "class_name", None)

        if class_name is not None:
            class_info = self.global_symtab.lookup(class_name)
            method_name = node.method_name.name

            assert_semantic(
                condition=(
                    class_info is not None
                    and method_name in class_info.methods
                ),
                error_type=SE.UNDECLARED_METHOD,
                coord=node.coord,
                name=method_name,
            )

            method_info = class_info.methods[method_name]
            param_names = method_info[1]
            param_types = method_info[2]    # parameter types

            # Check argument count matches parameter count
            assert_semantic(
                condition=(len(arg_exprs) == len(param_types)),
                error_type=SE.ARGUMENT_COUNT_MISMATCH,
                coord=node.coord,
                name=method_name,
            )

            # Check each argument type matches the corresponding parameter type
            for index, (arg, param_type) in enumerate(zip(arg_exprs, param_types)):
                param_name = param_names[index] if index < len(param_names) else method_name
                assert_semantic(
                    condition=(arg.mj_type == param_type),
                    error_type=SE.PARAMETER_TYPE_MISMATCH,
                    coord=node.coord,
                    name=param_name,
                )

            node.mj_type = method_info[0]  # return type
        else:
            node.mj_type = ObjectType

    def visit_Length(self, node: Length):
        # Visit the target expression
        self.visit(node.expr)

        expr_type = node.expr.mj_type

        # The target must be an array or String type
        assert_semantic(
            condition=(expr_type in (IntArrayType, CharArrayType, StringType)),
            error_type=SE.INVALID_LENGTH_TARGET,
            coord=node.coord,
        )

        # .length always returns an int
        node.mj_type = IntType

    def visit_NewArray(self, node: NewArray):
        # Visit the size expression
        self.visit(node.size)

        size_type = node.size.mj_type

        # The size must be an integer
        assert_semantic(
            condition=(size_type == IntType),
            error_type=SE.ARRAY_DIMENSION_MISMATCH,
            coord=node.coord,
            ltype=size_type,
        )

        # Resolve the array type (e.g. "int[]" -> IntArrayType)
        array_type = self.typemap.get(_type_name(node.type))
        node.mj_type = array_type

    def visit_NewObject(self, node: NewObject):
        class_name = _type_name(node.type)

        # The type must be a declared class
        class_info = self.global_symtab.lookup(class_name)

        assert_semantic(
            condition=(class_info is not None),
            error_type=SE.UNDECLARED_CLASS,
            coord=node.coord,
            name=class_name,
        )

        # Store the class name for downstream FieldAccess/MethodCall resolution
        node.class_name = class_name
        node.mj_type = ObjectType

    def visit_Constant(self, node: Constant):
        node.mj_type = self.typemap.get(node.type)

    def visit_This(self, node: This):
        # 'this' refers to the current class instance
        node.class_name = self.current_class
        node.mj_type = ObjectType

    def visit_ID(self, node: ID):
        name = node.name
        var_type = self.scope.lookup(name)

        if var_type is None and self.current_class is not None:
            class_info = self.global_symtab.lookup(self.current_class)

            if class_info is not None and name in class_info.fields:
                field_type_name = class_info.fields[name]
                var_type = self.typemap.get(field_type_name, ObjectType)

        assert_semantic(
            condition=(var_type is not None),
            error_type=SE.UNDECLARED_NAME,
            coord=node.coord,
            name=name,
        )

        node.mj_type = var_type
        node.scope = True

        if var_type == ObjectType and node.name in self._var_class_names:
            node.class_name = self._var_class_names[node.name]

    def visit_Type(self, node: Type):
        type_name = _type_name(node)
        mj_type = self.typemap.get(type_name)

        if mj_type is None:
            class_info = self.global_symtab.lookup(type_name)

            assert_semantic(
                condition=(class_info is not None),
                error_type=SE.UNDECLARED_CLASS,
                coord=node.coord,
                name=type_name,
            )

            mj_type = ObjectType

        node.mj_type = mj_type

    def visit_Extends(self, node: Extends):
        parent_name = node.super.name

        # The parent class must have been declared
        assert_semantic(
            condition=(self.global_symtab.lookup(parent_name) is not None),
            error_type=SE.UNDECLARED_CLASS,
            coord=node.coord,
            name=parent_name,
        )

    def visit_ExprList(self, node: ExprList):
        for expr in node.exprs or []:
            self.visit(expr)

    def visit_InitList(self, node: InitList):
        # First, validate that all expressions are constants
        for expr in node.exprs or []:
            assert_semantic(
                condition=isinstance(expr, Constant),
                error_type=SE.NOT_A_CONSTANT,
                coord=expr.coord,
            )
            self.visit(expr)

        exprs = node.exprs or []
        if not exprs:
            return

        # All elements must have the same type
        expected_type = exprs[0].mj_type
        for expr in exprs[1:]:
            assert_semantic(
                condition=(expr.mj_type == expected_type),
                error_type=SE.ARRAY_ELEMENT_TYPE_MISMATCH,
                coord=expr.coord,
                name=getattr(expr, "value", ""),
                ltype=expected_type,
                rtype=expr.mj_type,
            )
        element_to_array_type = {
            IntType: IntArrayType,
            CharType: CharArrayType,
        }
        node.mj_type = element_to_array_type.get(expected_type, ObjectType)


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    p = MJParser()
    # open file and parse it
    with open(input_path) as f:
        # Parse the code to an AST
        ast = p.parse(f.read())

        # First, build the global symtab
        global_symtab_builder = SymbolTableBuilder()
        global_symtab = global_symtab_builder.visit(ast)

        # Then, execute the semantic analysis
        sema = SemanticAnalyzer(global_symtab=global_symtab)
        sema.visit(ast)


if __name__ == "__main__":
    main()
