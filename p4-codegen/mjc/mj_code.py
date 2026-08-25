import argparse
import pathlib
import sys
from typing import Dict, List, Tuple

from mjc.mj_ast import *
from mjc.mj_block import (
    CFG,
    BasicBlock,
    Block,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from mjc.mj_interpreter import MJIRInterpreter
from mjc.mj_parser import MJParser
from mjc.mj_sema import NodeVisitor, SemanticAnalyzer, SymbolTableBuilder
from mjc.mj_type import CharType, IntType, VoidType

import rich

class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool):
        self.viewcfg: bool = viewcfg
        self.current_block: Block = None
        self.current_class = None
        self.exit_block = None
        self.scope_stack = []
        self.if_count = 0
        self.while_count = 0
        self.for_count = 0
        self.assert_count = 0
        self.loop_exit_blocks = []
        self.method_locals = {}  # Local variable bindings for each method, keyed by method name

        # version dictionary for temporaries. We use the name as a Key
        self.fname: str = "_glob_"
        self.versions: Dict[str, int] = {self.fname: 0}

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code: List[Tuple[str]] = []

        # Used for global declarations & constants (list, strings)
        self.text: List[Tuple[str]] = []

    def show(self):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        rich.print(_str.strip())

    def new_temp(self) -> str:
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 0
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def get_typename(self, mj_type) -> str:
        """
        Get the clean string representation of a primitive or array type.
        """
        if hasattr(mj_type, "typename"):
            return mj_type.typename
        if hasattr(mj_type, "name"):
            return mj_type.name
        
        # Fallback to string extraction if it's an instance of ArrayType
        type_str = str(mj_type)
        if "IntArrayType" in type_str or "int[]" in type_str:
            return "int[]"
        if "CharArrayType" in type_str or "char[]" in type_str:
            return "char[]"
            
        return type_str

    def new_text(self, typename: str) -> str:
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        return name

    # Scope management methods for variable bindings.
    # We maintain a stack of scopes where each scope is a dictionary mapping variable names to their generated locations.
    def push_scope(self):
        self.scope_stack.append({})

    def pop_scope(self):
        if self.scope_stack:
            self.scope_stack.pop()

    def bind_name(self, name: str, gen_loc: str):
        if not self.scope_stack:
            self.push_scope()
        self.scope_stack[-1][name] = gen_loc

        # Track that this variable name maps to this register in the current method
        if self.fname not in self.method_locals:
            self.method_locals[self.fname] = {}
        self.method_locals[self.fname][name] = gen_loc

    def lookup_name(self, name: str):
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        return None

    # You must implement visit_Nodename methods for all of the AST nodes.
    # In your code, you will need to make instructions
    # and append them to the current block code list.
    #
    # A few sample methods follow. Do not hesitate to complete or change
    # them if needed.

    def visit_Program(self, node: Program):
        # First visit all of the Class Declarations
        for class_decl in node.class_decls:
            self.visit(class_decl)

        # At the end of codegen, first init the self.code with the list
        # of global instructions allocated in self.text
        self.code = self.text.copy()


        # After, visit all the class definitions and emit the
        # code stored inside basic blocks.
        for class_decl in node.class_decls:
            block_visitor = EmitBlocks()
            block_visitor.visit(class_decl.cfg)
            for code in block_visitor.code:
                self.code.append(code)

    def visit_ClassDecl(self, node: ClassDecl):
        # Create a cfg to hold the class context
        node.cfg = BasicBlock(label=None)

        #
        # Guideline
        #
        # Generate the class decl instruction
        parent_name = None
        if node.extends is not None:
            if hasattr(node.extends, "super") and node.extends.super is not None:
                parent_name = node.extends.super.name
            elif hasattr(node.extends, "name"):
                parent_attr = node.extends.name
                parent_name = parent_attr.name if hasattr(parent_attr, "name") else str(parent_attr)

            if parent_name is not None and not str(parent_name).startswith("@"):
                parent_name = f"@{parent_name}"
        class_inst = ("class", f"@{node.name.name}", parent_name)
        node.cfg.append(class_inst)

        # Save current class context
        self.current_class = node
    
        # Visit all the Field Declarations
        self.current_block = node.cfg
        for field_decl in node.var_decls:
            self.visit(field_decl)

        # Visit all the Method Declarations
        for method_decl in node.method_decls:
            self.visit(method_decl)

        # Finally, visit all the method definitions and emit the
        # code stored inside basic blocks.
        for method_decl in node.method_decls:
            block_visitor = EmitBlocks()
            block_visitor.visit(method_decl.cfg)
            for instruction in block_visitor.code:
                node.cfg.append(instruction)

        # If -cfg flag is present in command line
        if self.viewcfg:
            for method_decl in node.method_decls:
                method_name = getattr(method_decl, "name", None)
                if method_name is not None:
                    method_name = method_name.name
                else:
                    method_name = "main"

                dot = CFG(f"@{node.name.name}.{method_name}")
                dot.view(method_decl.cfg)

    def visit_VarDecl(self, node: VarDecl):
        # Extract the type name safely using the helper method
        type_name = self.get_typename(node.mj_type)

        # Class field
        if self.current_block == self.current_class.cfg:
            field_name = f"@{self.current_class.name.name}.{node.name.name}"
            node.gen_loc = field_name
            default_value = None

            if node.init is not None:
                # Visit initializer
                self.visit(node.init)
                # Constants / lists
                if isinstance(node.init, (Constant, InitList)):
                    default_value = node.init.value
                    # If it is a String constant in class scope, convert it to a list of characters
                    # to match the interpreter's expected representation for array-like structures
                    if type_name == "String" and isinstance(default_value, str):
                        default_value = list(default_value)
                # New array: pass the integer size so the interpreter allocates the right number of cells.
                # Using the register name (gen_loc) would cause _store_fields to treat it as a 2-char
                # string and allocate only 2 cells, corrupting adjacent memory at runtime.
                elif isinstance(node.init, NewArray) and isinstance(node.init.size, Constant):
                    default_value = int(node.init.size.value)
                # Objects / arrays
                else:
                    default_value = node.init.gen_loc

            field_inst = (f"field_{type_name}", field_name, default_value)
            self.current_block.append(field_inst)

        # Local variable
        else:
            # Allocate a new unique logical location for the variable
            node.gen_loc = self.new_temp()
            self.bind_name(node.name.name, node.gen_loc)

            # Use alloc_Object for arrays and object types to avoid interpreter alloc size issues
            if "[]" in type_name or type_name not in ("int", "char", "boolean"):
                alloc_inst = ("alloc_Object", node.gen_loc)
            else:
                alloc_inst = (f"alloc_{type_name}", node.gen_loc)
            self.current_block.append(alloc_inst)

            # Optional initializer
            if node.init is not None:
                if isinstance(node.init, InitList):
                    # Create a global array constant in the text section
                    global_name = self.new_text(type_name.replace("[]", "_array"))
                    
                    # Extract the constant values from the InitList elements
                    values = []
                    for expr in node.init.exprs:
                        if hasattr(expr, "value"):
                            values.append(expr.value)
                        else:
                            values.append(0)
                            
                    self.text.append((f"global_{type_name}", global_name, values))
                    
                    # Load the global constant array into the local variable register
                    load_inst = (f"load_{type_name}", global_name, node.gen_loc)
                    self.current_block.append(load_inst)
                    
                elif isinstance(node.init, Constant) and type_name in ("int[]", "char[]"):
                    # Create a global constant for literal array/string initialization
                    global_name = self.new_text(type_name.replace("[]", "_array"))
                    
                    value = node.init.value
                    if isinstance(value, str) and len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                        value = value[1:-1]
                        
                    if type_name == "char[]":
                        # Convert string literal to a list of characters for the global array descriptor
                        values = list(value)
                    else:
                        values = [value]
                        
                    self.text.append((f"global_{type_name}", global_name, values))
                    
                    # Load the global constant array into the local variable register
                    load_inst = (f"load_{type_name}", global_name, node.gen_loc)
                    self.current_block.append(load_inst)
                    
                else:
                    # General expression or object allocation initialization
                    self.visit(node.init)
                    store_inst = (f"store_{type_name}", node.init.gen_loc, node.gen_loc)
                    self.current_block.append(store_inst)

    def visit_MethodDecl(self, node: MethodDecl):
        # Create CFG for the method
        node.cfg = BasicBlock(label=f"{node.name.name}.entry")

        # Set current function scope
        self.fname = node.name.name
        self.method_locals[self.fname] = {}
        self.versions[self.fname] = 0

        # Set current block
        self.current_block = node.cfg
        self.push_scope()

        type_name = self.get_typename(node.mj_type)

        # Allocate registers for parameters
        params = []
        for param in node.param_list.params:
            param_type = param.mj_type.typename
            param_reg = self.new_temp()
            param.gen_loc = param_reg
            self.bind_name(param.name.name, param_reg)
            params.append((param_type, param_reg))

        # Define instruction
        define_inst = (
            f"define_{type_name}",
            f"@{self.current_class.name.name}.{node.name.name}",
            params
        )
        self.current_block.append(define_inst)

        # Create entry block
        entry_block = BasicBlock(label=f"{node.name.name}.body")
        self.current_block.next_block = entry_block
        self.current_block = entry_block

        # Create exit block
        exit_block = BasicBlock(label=f"{node.name.name}.exit")
        self.exit_block = exit_block

        # Allocate return register for non-void methods
        self.return_reg = None
        # if not isinstance(node.mj_type, VoidType):
        if node.mj_type != VoidType:
            self.return_reg = self.new_temp()
            alloc_inst = (
                f"alloc_{type_name}",
                self.return_reg
            )
            self.current_block.append(alloc_inst)

        # Visit method body
        self.visit(node.body)

        # Ensure jump to exit block exists
        if (
            len(self.current_block.instructions) == 0
            or self.current_block.instructions[-1][0]
            not in (
                "jump",
                "return_void",
                "return_int",
                "return_boolean",
                "return_char",
            )
        ):
            jump_inst = ("jump", f"%{exit_block.label}")
            self.current_block.append(jump_inst)
        self.current_block.next_block = exit_block

        # Emit exit block
        self.current_block = exit_block

        # Emit the exit label so the interpreter can resolve jumps
        self.current_block.append((f"{exit_block.label}:",))

        # Return for non-void methods
        if self.return_reg is not None:
            temp = self.new_temp()
            load_inst = (f"load_{type_name}", self.return_reg, temp)
            self.current_block.append(load_inst)
            return_inst = (f"return_{type_name}", temp)
            self.current_block.append(return_inst)

        # Return void
        else:
            return_inst = ("return_void",)
            self.current_block.append(return_inst)

        self.pop_scope()

    def visit_MainMethodDecl(self, node: MainMethodDecl):
        # Create CFG for the main method
        node.cfg = BasicBlock(label="main.entry")

        # Set current function scope
        self.fname = "main"
        self.method_locals[self.fname] = {}
        self.versions[self.fname] = 0

        # Set current block
        self.current_block = node.cfg

        # Main method parameter
        args_reg = self.new_temp()
        params = [("String[]", args_reg)]

        # Define instruction
        define_inst = (
            "define_void",
            f"@{self.current_class.name.name}.main",
            params
        )
        self.current_block.append(define_inst)

        # Create entry block
        entry_block = BasicBlock(label="main.body")
        self.current_block.next_block = entry_block
        self.current_block = entry_block

        # Create exit block
        exit_block = BasicBlock(label="main.exit")
        self.exit_block = exit_block

        # Store args register
        node.args.gen_loc = args_reg
        self.bind_name(node.args.name, args_reg)

        # Visit method body
        self.visit(node.body)

        # Ensure jump to exit block exists
        if (
            len(self.current_block.instructions) == 0
            or self.current_block.instructions[-1][0]
            not in (
                "jump",
                "return_void",
            )
        ):
            jump_inst = ("jump", f"%{exit_block.label}")
            self.current_block.append(jump_inst)
        self.current_block.next_block = exit_block

        # Emit exit block
        self.current_block = exit_block
        # Emit the exit label so the interpreter can resolve jumps
        self.current_block.append((f"{exit_block.label}:",))
        return_inst = ("return_void",)
        self.current_block.append(return_inst)

        self.pop_scope()

    def visit_ParamList(self, node: ParamList):
        pass

    def visit_ParamDecl(self, node: ParamDecl):
        pass

    def visit_Compound(self, node: Compound):
        # Visit the block items
        self.push_scope()
        for statement in node.statements:
            self.visit(statement)
        self.pop_scope()

    def visit_If(self, node: If):
        # Create unique labels for the if blocks
        self.if_count += 1
        idx = self.if_count

        cond_label = f"if.cond{idx}"
        then_label = f"if.then{idx}"
        else_label = f"if.else{idx}"
        end_label = f"if.end{idx}"

        # Keep the current block so we can link it to the condition block
        prev_block = self.current_block

        # Create the condition block and link the current block to it
        cond_block = ConditionBlock(label=cond_label)
        prev_block.next_block = cond_block
        cond_block.append((f"{cond_label}:",))
        self.current_block = cond_block

        # Evaluate the condition
        self.visit(node.cond)
        
        # Create the then and end blocks
        then_block = BasicBlock(label=then_label)
        end_block = BasicBlock(label=end_label)

        # Connect blocks based on the presence of an else branch
        if node.iffalse is not None:
            else_block = BasicBlock(label=else_label)
            cond_block.append(("cbranch", node.cond.gen_loc, f"%{then_label}", f"%{else_label}"))
            
            cond_block.next_block = then_block
            then_block.next_block = else_block
            else_block.next_block = end_block
        else:
            # If there is no else branch, cbranch goes directly to the end block
            cond_block.append(("cbranch", node.cond.gen_loc, f"%{then_label}", f"%{end_label}"))
            
            cond_block.next_block = then_block
            then_block.next_block = end_block

        # Emit the then label and generate the true branch
        then_block.append((f"{then_label}:",))
        self.current_block = then_block
        if node.iftrue is not None:
            self.visit(node.iftrue)
            
        # Ensure the end of the generated then branch links to the correct next block
        if self.current_block is not None:
            if (
                len(self.current_block.instructions) == 0
                or self.current_block.instructions[-1][0] 
                not in ("jump", "return_int", "return_void", "return_boolean", "return_char")
            ):
                self.current_block.append(("jump", f"%{end_label}"))
            
            # Reconnect to the original layout chain to avoid orphaning the else block
            if node.iffalse is not None:
                self.current_block.next_block = else_block
            else:
                self.current_block.next_block = end_block

        # Emit the else label and generate the false branch if it exists
        if node.iffalse is not None:
            else_block.append((f"{else_label}:",))
            self.current_block = else_block
            self.visit(node.iffalse)
                
            # Ensure the end of the generated else branch links to the end block
            if self.current_block is not None:
                if (
                    len(self.current_block.instructions) == 0
                    or self.current_block.instructions[-1][0] 
                    not in ("jump", "return_int", "return_void", "return_boolean", "return_char")
                ):
                    self.current_block.append(("jump", f"%{end_label}"))
                # Link the final block of else branch to the end block
                self.current_block.next_block = end_block

        # Emit the end label and continue after the if
        end_block.append((f"{end_label}:",))
        self.current_block = end_block

    def visit_While(self, node: While):
        # Create unique labels for the loop blocks
        self.while_count += 1
        idx = self.while_count

        cond_label = f"while.cond{idx}"
        body_label = f"while.body{idx}"
        exit_label = f"while.end{idx}"

        # Keep the current block so we can link it to the loop condition
        prev_block = self.current_block

        # Create the condition, body, and exit blocks
        cond_block = ConditionBlock(label=cond_label)
        body_block = BasicBlock(label=body_label)
        exit_block = BasicBlock(label=exit_label)

        # Connect the blocks in loop order
        prev_block.next_block = cond_block
        cond_block.next_block = body_block
        body_block.next_block = exit_block

        cond_block.append((f"{cond_label}:",))
        body_block.append((f"{body_label}:",))
        exit_block.append((f"{exit_label}:",))

        # Save the exit block so break statements can jump here
        self.loop_exit_blocks.append(exit_block)

        # Emit the condition label and evaluate the loop condition
        self.current_block = cond_block
        self.visit(node.cond)
        cond_block.append(("cbranch", node.cond.gen_loc, f"%{body_label}", f"%{exit_label}"))

        # Emit the body label and generate the loop body
        self.current_block = body_block
        if node.body is not None:
            self.visit(node.body)
        # Jump back to the condition if the body did not end control flow
        if (
            len(self.current_block.instructions) == 0
            or self.current_block.instructions[-1][0]
            not in ("jump", "return_int", "return_void", "return_boolean", "return_char")
        ):
            self.current_block.append(("jump", f"%{cond_label}"))

        # Emit the exit label and continue code generation after the loop
        self.current_block = exit_block
        self.loop_exit_blocks.pop()

    def visit_For(self, node: For):
        # Create unique labels for the for-loop blocks
        self.for_count += 1
        idx = self.for_count

        cond_label = f"for.cond{idx}"
        body_label = f"for.body{idx}"
        next_label = f"for.next{idx}"
        exit_label = f"for.end{idx}"

        # Keep the current block so we can link it to the loop condition
        prev_block = self.current_block

        # Open a new scope context for the loop variables
        self.push_scope()

        # Visit the initializer before entering the loop blocks
        if node.init is not None:
            self.visit(node.init)

        # Create the condition, body, increment, and exit blocks
        cond_block = ConditionBlock(label=cond_label)
        body_block = BasicBlock(label=body_label)
        next_block = BasicBlock(label=next_label)
        exit_block = BasicBlock(label=exit_label)

        # Connect the blocks in for-loop order
        prev_block.next_block = cond_block
        cond_block.next_block = body_block

        cond_block.append((f"{cond_label}:",))
        body_block.append((f"{body_label}:",))
        next_block.append((f"{next_label}:",))
        exit_block.append((f"{exit_label}:",))

        # Save the exit block so break statements can jump here
        self.loop_exit_blocks.append(exit_block)

        # Emit the condition label and evaluate the loop condition
        self.current_block = cond_block
        if node.cond is not None:
            self.visit(node.cond)
            cond_block.append(("cbranch", node.cond.gen_loc, f"%{body_label}", f"%{exit_label}"))
        else:
            cond_block.append(("jump", f"%{body_label}"))

        # Emit the body label and generate the loop body
        self.current_block = body_block
        if node.body is not None:
            self.visit(node.body)

        # Ensure the end of the generated body links to the increment block
        if self.current_block is not None:
            if (
                len(self.current_block.instructions) == 0
                or self.current_block.instructions[-1][0]
                not in ("jump", "return_int", "return_void", "return_boolean", "return_char")
            ):
                self.current_block.append(("jump", f"%{next_label}"))
            # Link the current block to the increment block for correct CFG structure
            self.current_block.next_block = next_block

        # Emit the increment label, generate the next expression, and loop back
        self.current_block = next_block
        if node.next is not None:
            self.visit(node.next)

        # Jump back to the condition if the increment did not end control flow
        if (
            len(self.current_block.instructions) == 0
            or self.current_block.instructions[-1][0]
            not in ("jump", "return_int", "return_void", "return_boolean", "return_char")
        ):
            self.current_block.append(("jump", f"%{cond_label}"))

        self.current_block.next_block = exit_block
        # Emit the exit label and continue code generation after the loop
        self.current_block = exit_block
        self.loop_exit_blocks.pop()
        
        # Close the loop scope context
        self.pop_scope()

    def visit_DeclList(self, node: DeclList):
        # Visit each declaration in the for-loop initializer
        for decl in node.decls:
            self.visit(decl)

    def visit_Print(self, node: Print):
        # Visit the expression
        if node.expr is None:
            self.current_block.append(("print_void",))
            return

        if isinstance(node.expr, ExprList):
            # Handle the cases when node.expr is ExprList
            for expr in node.expr.exprs:
                self.visit(expr)
                type_name = self.get_typename(expr.mj_type)
                inst = (f"print_{type_name}", expr.gen_loc)
                self.current_block.append(inst)
            return

        self.visit(node.expr)

        # Create the opcode using the safe helper method
        type_name = self.get_typename(node.expr.mj_type)
        inst = (f"print_{type_name}", node.expr.gen_loc)
        self.current_block.append(inst)

    def visit_Assert(self, node: Assert):
        # Create unique labels for the assert blocks
        self.assert_count += 1
        idx = self.assert_count

        cond_label = f"assert.cond{idx}"
        ok_label = f"assert.ok{idx}"
        fail_label = f"assert.fail{idx}"
        end_label = f"assert.end{idx}"

        # Keep the current block so we can link it to the assert condition
        prev_block = self.current_block

        # Create the condition block and link the current block to it
        cond_block = ConditionBlock(label=cond_label)
        prev_block.next_block = cond_block

        # Evaluate the condition and branch to success or failure
        self.current_block = cond_block
        self.visit(node.expr)
        
        # Ensure the condition expression value is fresh and correctly isolated
        cond_reg = node.expr.gen_loc
        cond_block.append(("cbranch", cond_reg, f"%{ok_label}", f"%{fail_label}"))

        # Create the success, failure, and end blocks
        ok_block = BasicBlock(label=ok_label)
        fail_block = BasicBlock(label=fail_label)
        end_block = BasicBlock(label=end_label)

        # Connect the blocks in execution order
        cond_block.next_block = ok_block
        ok_block.next_block = fail_block
        fail_block.next_block = end_block

        # Emit the success label and continue after the assert
        ok_block.append((f"{ok_label}:",))
        ok_block.append(("jump", f"%{end_label}"))

        # Emit the failure label, print the error message, and jump to exit
        fail_block.append((f"{fail_label}:",))
        
        # Extract line and column information from the AST node coordinates
        line = node.coord.line if node.coord and node.coord.line is not None else 0
        # Add 7 to the column number to account for the length of "assert " in the source code, so the error points to the condition expression
        col = (node.coord.column + 7) if node.coord and node.coord.column is not None else 0
        error_string = f"assertion_fail on {line}:{col}"
        
        assert_msg = self.new_text("str")
        self.text.append(("global_string", assert_msg, error_string))
        fail_block.append(("print_String", assert_msg))
        fail_block.append(("jump", f"%{self.exit_block.label}"))

        # Emit the end label and continue code generation after the assert
        end_block.append((f"{end_label}:",))
        self.current_block = end_block

    def visit_Break(self, node: Break):
        # Generate a jump instruction to the current exit label
        inst = ("jump", f"%{self.loop_exit_blocks[-1].label}")
        self.current_block.append(inst)

    def visit_Return(self, node: Return):
        # Visit the return value expression
        if node.expr is not None:
            self.visit(node.expr)

            # Emit the store instruction to save the return value
            if self.return_reg is not None:
                inst = ("store_" + node.expr.mj_type.typename, node.expr.gen_loc, self.return_reg)
                self.current_block.append(inst)

        # Emit a jump to the method exit block
        self.current_block.append(("jump", f"%{self.exit_block.label}"))

    def visit_Assignment(self, node: Assignment):
        # Visit the right side first to evaluate its value
        self.visit(node.rvalue)
        source = node.rvalue.gen_loc

        type_name = self.get_typename(node.lvalue.mj_type)

        # Check if the left side is a field access to generate the correct target format
        if isinstance(node.lvalue, FieldAccess):
            # Visit the object to ensure its register (e.g., %this or %1) is generated
            self.visit(node.lvalue.object)
            field_name = node.lvalue.field_name.name
            target = f"{node.lvalue.object.gen_loc}.{field_name}"
            is_array_ref = False
        elif isinstance(node.lvalue, ArrayRef):
            # For array elements on the left-hand side, compute their address cleanly
            self.visit(node.lvalue.subscript)
            target = self.new_temp()
            array_type = self.get_typename(node.lvalue.mj_type)
            
            if isinstance(node.lvalue.name, FieldAccess):
                self.visit(node.lvalue.name.object)
                array_source = f"{node.lvalue.name.object.gen_loc}.{node.lvalue.name.field_name.name}"
            else:
                self.visit(node.lvalue.name)
                array_source = node.lvalue.name.gen_loc
                
            elem_inst = (f"elem_{array_type}", array_source, node.lvalue.subscript.gen_loc, target)
            self.current_block.append(elem_inst)
            is_array_ref = True
        else:
            # For regular variables, visit as usual to get the gen_loc
            self.visit(node.lvalue)
            target = node.lvalue.gen_loc
            is_array_ref = False

        # Emit the code according to the assignment operator
        if node.op == "=":
            if is_array_ref:
                inst = (f"store_{type_name}_*", source, target)
            else:
                inst = (f"store_{type_name}", source, target)
            self.current_block.append(inst)
            node.gen_loc = source
            return

        # For compound assignments
        op_map = {
            "+=": "add",
            "-=": "sub",
            "*=": "mul",
            "/=": "div",
            "%=": "mod",
        }
        if node.op in op_map:
            loaded = self.new_temp()
            if is_array_ref:
                self.current_block.append((f"load_{type_name}_*", target, loaded))
            else:
                self.current_block.append((f"load_{type_name}", target, loaded))
                
            result = self.new_temp()
            inst = (f"{op_map[node.op]}_{type_name}", loaded, source, result)
            self.current_block.append(inst)
            
            if is_array_ref:
                self.current_block.append((f"store_{type_name}_*", result, target))
            else:
                self.current_block.append((f"store_{type_name}", result, target))
            node.gen_loc = result

    def visit_BinaryOp(self, node: BinaryOp):
        # Visit the left and right expressions to set their gen locations
        self.visit(node.lvalue)
        self.visit(node.rvalue)
        node.gen_loc = self.new_temp()
        op_map = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            "<": "lt",
            "<=": "le",
            ">": "gt",
            ">=": "ge",
            "==": "eq",
            "!=": "ne",
            "&&": "and",
            "||": "or",
        }
        opcode = op_map[node.op]
        type_name = self.get_typename(node.lvalue.mj_type)
        inst = (f"{opcode}_{type_name}", node.lvalue.gen_loc, node.rvalue.gen_loc, node.gen_loc)
        self.current_block.append(inst)

    def visit_UnaryOp(self, node: UnaryOp):
        # Visit the expression to set its gen location
        self.visit(node.expr)
        type_name = self.get_typename(node.expr.mj_type)
        if node.op == "+":
            node.gen_loc = node.expr.gen_loc
        elif node.op == "-":
            zero_temp = self.new_temp()
            self.current_block.append((f"literal_{type_name}", 0, zero_temp))
            node.gen_loc = self.new_temp()
            self.current_block.append((f"sub_{type_name}", zero_temp, node.expr.gen_loc, node.gen_loc))
        elif node.op == "!":
            node.gen_loc = self.new_temp()
            self.current_block.append((f"not_{type_name}", node.expr.gen_loc, node.gen_loc))

    def visit_ArrayRef(self, node: ArrayRef):
        # Visit the subscript expression
        self.visit(node.subscript)
        
        # Allocate a temporary register to hold the element address
        addr_temp = self.new_temp()
        type_name = self.get_typename(node.mj_type)
        
        # Check if the array itself is a field access (e.g., this.v[i])
        if isinstance(node.name, FieldAccess):
            self.visit(node.name.object)
            array_source = f"{node.name.object.gen_loc}.{node.name.field_name.name}"
        else:
            self.visit(node.name)
            array_source = node.name.gen_loc
        
        # Generate the elem instruction to compute the target cell memory address
        elem_inst = (f"elem_{type_name}", array_source, node.subscript.gen_loc, addr_temp)
        self.current_block.append(elem_inst)
        
        # Allocate the final temporary register to load the actual value from the address
        node.gen_loc = self.new_temp()
        load_inst = (f"load_{type_name}_*", addr_temp, node.gen_loc)
        self.current_block.append(load_inst)

    def visit_FieldAccess(self, node: FieldAccess):
        # Visit the object expression to set its gen location
        self.visit(node.object)
        node.gen_loc = self.new_temp()
        
        field_name = node.field_name.name
        source = f"{node.object.gen_loc}.{field_name}"
        type_name = self.get_typename(node.mj_type)
        
        # Only String types require load_ to fetch descriptors without structure loss in binary ops.
        # Native arrays (int[], char[]) must use get_field_ to modify the true heap reference.
        if type_name == "String":
            inst = (f"load_{type_name}", source, node.gen_loc)
        else:
            inst = (f"get_field_{type_name}", source, node.gen_loc)
            
        self.current_block.append(inst)

    def visit_MethodCall(self, node: MethodCall):
        self.visit(node.object)
        # Collect argument expressions from ExprList or a bare expression
        if node.args is not None:
            arg_exprs = node.args.exprs if hasattr(node.args, "exprs") else [node.args]
        else:
            arg_exprs = []

        # Emit param instructions for each argument
        for arg in arg_exprs:
            self.visit(arg)
            self.current_block.append((f"param_{arg.mj_type.typename}", arg.gen_loc))

        # Emit the call: source is "%obj.method" so the interpreter can split on "."
        node.gen_loc = self.new_temp()
        type_name = self.get_typename(node.mj_type)
        source = f"{node.object.gen_loc}.{node.method_name.name}"
        self.current_block.append((f"call_{type_name}", source, node.gen_loc))

    def visit_Length(self, node: Length):
        # Visit the expression to set its gen location
        self.visit(node.expr)
        # Alloc a register to store the length
        node.gen_loc = self.new_temp()
        # gen the length instruction
        length_inst = ("length", node.expr.gen_loc, node.gen_loc)
        # Store the length instruction
        self.current_block.append(length_inst)

    def visit_NewArray(self, node: NewArray):
        # Visit the size expression to set its gen location
        self.visit(node.size)
        node.gen_loc = self.new_temp()
        type_name = self.get_typename(node.mj_type)
        if isinstance(node.size, Constant):
            size = node.size.value
        else:
            size = node.size.gen_loc
        inst = (f"new_{type_name}_{size}", node.gen_loc)
        self.current_block.append(inst)

    def visit_NewObject(self, node: NewObject):
        node.gen_loc = self.new_temp()
        # class_name = None
        # if hasattr(node, "class_name"):
        #     class_attr = node.class_name
        #     class_name = class_attr.name if hasattr(class_attr, "name") else str(class_attr)
        # elif hasattr(node, "type") and node.type is not None:
        #     type_attr = node.type
        #     class_name = type_attr.name if hasattr(type_attr, "name") else str(type_attr)
        class_name = str(node.class_name)
        inst = (f"new_@{class_name}", node.gen_loc)
        self.current_block.append(inst)

    def visit_Constant(self, node: Constant):
        # Emit a literal into a temporary register and use it as gen_loc
        type_name = self.get_typename(node.mj_type)
        temp = self.new_temp()
        
        # Clean string literal bounds directly on the node value to ensure field declarations receive it correctly
        if type_name == "String" and isinstance(node.value, str) and len(node.value) >= 2 and node.value[0] == '"' and node.value[-1] == '"':
            node.value = node.value[1:-1]

        # For String constants, we must allocate them globally in the text section to create a proper array descriptor
        if type_name == "String":
            global_name = self.new_text("String")
            # Convert the raw python string into a list of characters for the interpreter array layout
            values = list(node.value)
            self.text.append((f"global_{type_name}", global_name, values))
            
            # Load the global string descriptor structure into the temporary register
            load_inst = (f"load_{type_name}", global_name, temp)
            self.current_block.append(load_inst)
        else:
            # Standard primitive constant emission
            literal_inst = (f"literal_{type_name}", node.value, temp)
            self.current_block.append(literal_inst)
            
        node.gen_loc = temp

    def visit_This(self, node: This):
        # visit_This now assigns "%this" to node.gen_loc, making the current object 
        # reference available for downstream use
        node.gen_loc = "%this"

    def visit_ID(self, node: ID):
        # visit_ID assigns the gen_loc of the variable's binding to node.gen_loc, allowing the 
        # variable's value to be accessed in subsequent code generation steps
        resolved = self.lookup_name(node.name)
        if resolved is not None:
            node.gen_loc = resolved
        else:
            node.gen_loc = f"%{node.name}"

    def visit_Type(self, node: Type):
        node.gen_loc = node.name

    def visit_Extends(self, node: Extends):
        node.gen_loc = node.name

    def visit_ExprList(self, node: ExprList):
        # initializes node.value as an empty list, visits each expression 
        # in node.exprs to evaluate it (setting its gen_loc), and appends each expression's gen_loc to node.value
        node.value = []
        for expr in node.exprs:
            self.visit(expr)
            node.value.append(expr.gen_loc)

    def visit_InitList(self, node: InitList):
        # Visit each expression to evaluate it and collect both registers and raw values
        node.value = []
        node.gen_loc = [] # Keep a copy of registers if needed elsewhere
        
        for expr in node.exprs:
            self.visit(expr)
            node.gen_loc.append(expr.gen_loc)
            
            # Extract the literal raw value (assuming it's a Constant)
            node.value.append(expr.value)

def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate MJIR. By default, this script only runs the interpreter on the MJIR. \
              Use the other options for printing the MJIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print MJIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--ir-pp",
        help="Print MJIR generated from input_file. (pretty print)",
        action="store_true",
    )
    parser.add_argument(
        "--cfg",
        help="Show the cfg of the input_file.",
        action="store_true",
    )

    args = parser.parse_args()

    print_ir = args.ir
    print_ir_pp = args.ir_pp
    create_cfg = args.cfg

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = MJParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())

    global_symtab_builder = SymbolTableBuilder()
    global_symtab = global_symtab_builder.visit(ast)
    sema = SemanticAnalyzer(global_symtab=global_symtab)
    sema.visit(ast)

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated MJIR: --------")
        rich.print(gencode)
        print("------------------------\n")

    elif print_ir_pp:
        print("Generated MJIR: --------")
        gen.show()
        print("------------------------\n")

    else:
        vm = MJIRInterpreter()
        # for i, inst in enumerate(gencode):
        #     print(i, inst)
        vm.run(gencode)


if __name__ == "__main__":
    main()
