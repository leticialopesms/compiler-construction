import argparse
import pathlib
import sys
from typing import List, Tuple

import rich

from mjc.mj_ast import ClassDecl, Program
from mjc.mj_block import (
    CFG,
    BasicBlock,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from mjc.mj_code import CodeGenerator
from mjc.mj_interpreter import MJIRInterpreter
from mjc.mj_parser import MJParser
from mjc.mj_sema import NodeVisitor, SemanticAnalyzer, SymbolTableBuilder


class DataFlow(NodeVisitor):
    def __init__(self, viewcfg: bool):
        # flag to show the optimized control flow graph
        self.viewcfg: bool = viewcfg

        # list of code instructions after optimizations
        self.code: List[Tuple[str]] = []

        self.current_func = None

        # Reaching Definitions (RD)
        self.rd_blocks = []
        self.rd_gen = {}
        self.rd_kill = {}
        self.rd_in = {}
        self.rd_out = {}

        # Auxiliary structures
        self.definitions = {}
        self.defs_by_var = {}
        self.definition_vars = {}

        # Live Variables (LV)
        self.lv_use = {}
        self.lv_def = {}
        self.lv_in = {}
        self.lv_out = {}

    def show(self):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        rich.print(_str.strip())

    def visit_Program(self, node: Program):
        # First, save the global instructions on code member
        self.code = node.text[:]  # [:] to do a copy

        # Visit all class declaration in the program
        for class_decl in node.class_decls:
            self.visit(class_decl)

    def visit_ClassDecl(self, node: ClassDecl):
        bb = EmitBlocks()
        bb.visit(node.cfg)
        for _code in bb.code:
            if "class" in _code[0] or "field" in _code[0]:
                self.code.append(_code)

        for method_decl in node.method_decls:

            # Build the control-flow edges (successors/predecessors) before any
            # dataflow analysis: the code generator only links blocks linearly.
            self.build_cfg_edges(method_decl.cfg)

            self.current_func = method_decl

            self.rd_blocks = []
            self.rd_gen = {}
            self.rd_kill = {}
            self.rd_in = {}
            self.rd_out = {}
            self.definitions = {}
            self.defs_by_var = {}
            self.definition_vars = {}
            self.lv_use = {}
            self.lv_def = {}
            self.lv_in = {}
            self.lv_out = {}
            # start with Reach Definitions Analysis
            self.buildRD_blocks(method_decl.cfg)
            self.computeRD_gen_kill()
            self.computeRD_in_out()

            # after do live variable analysis
            self.buildLV_blocks(method_decl.cfg)
            self.computeLV_use_def()
            self.computeLV_in_out()
            # and do dead code elimination
            self.deadcode_elimination()

            # after that do cfg simplify (optional)
            self.discard_unused_allocs(method_decl.cfg)
            self.short_circuit_jumps(method_decl.cfg)
            self.merge_blocks(method_decl.cfg)

            # finally save optimized instructions in self.code
            self.appendOptimizedCode(method_decl.cfg)

        if self.viewcfg:
            for method_decl in node.method_decls:
                method_name = getattr(method_decl, "name", None)
                if method_name is not None:
                    method_name = method_name.name
                else:
                    method_name = "main"

                dot = CFG(f"@{node.name.name}.{method_name}.opt")
                dot.view(method_decl.cfg)


    def build_cfg_edges(self, cfg: CFG):
        """
        Populate the successors and predecessors of every block in the CFG.

        The code generator only links blocks through `next_block` (a linear
        linked list); it never records the real control-flow edges. Here we
        rebuild them from each block's terminator instruction:
          - jump      -> single successor (the jump target label)
          - cbranch   -> two successors (taken / fall-through labels)
          - return_*  -> no successors
          - otherwise -> falls through to next_block
        """
        # Map every label to its block and reset edge sets.
        label_map = {}
        blocks = []
        block = cfg
        while block is not None:
            blocks.append(block)
            if block.label is not None:
                label_map[block.label] = block
            block.successors = []
            block.predecessors = []
            block = block.next_block

        # Compute successors from the terminator of each block.
        for block in blocks:
            terminator = block.instructions[-1] if block.instructions else None
            targets = []
            falls_through = True
            if terminator is not None:
                op = terminator[0]
                if op == "jump":
                    targets.append(terminator[1].lstrip("%"))
                    falls_through = False
                elif op == "cbranch":
                    targets.append(terminator[2].lstrip("%"))
                    targets.append(terminator[3].lstrip("%"))
                    falls_through = False
                elif op.startswith("return"):
                    falls_through = False

            for target in targets:
                succ = label_map.get(target)
                if succ is not None and succ not in block.successors:
                    block.successors.append(succ)

            if falls_through and block.next_block is not None:
                if block.next_block not in block.successors:
                    block.successors.append(block.next_block)

        # Derive predecessors from successors.
        for block in blocks:
            for succ in block.successors:
                if block not in succ.predecessors:
                    succ.predecessors.append(block)

    def buildRD_blocks(self, cfg: CFG):
        """
        Build the list of basic blocks that compose the current CFG.
        """
        block = cfg
        while block is not None:
            self.rd_blocks.append(block)
            # initialize RD structures
            self.rd_gen[block] = set()
            self.rd_kill[block] = set()
            self.rd_in[block] = set()
            self.rd_out[block] = set()
            # move to the next block
            block = block.next_block

    def find_definitions(self):
        """
        Find all reaching definitions in the current method.
        Creates two dictionaries:
        - self.definitions: definition_id -> (block, instruction)
        - self.defs_by_var: variable -> {definition_ids}
        """
        self.definitions = {}
        self.defs_by_var = {}
        self.definition_vars = {}
        def_id = 0

        # Instructions that do not define a new register
        ignore = {"define", "store", "jump", "cbranch", "print", "return", "param"}

        for block in self.rd_blocks:
            for inst in block.instructions:
                # ignore labels (entry:, if.then:, ...)
                if inst[0].endswith(":"):
                    continue
                op = inst[0].split("_")[0]
                if op in ignore:
                    continue
                # A definition is an instruction that always ends with a register (e.g., %1, %2, ...)
                if (
                    len(inst) >= 2
                    and isinstance(inst[-1], str)
                    and inst[-1].startswith("%")
                ):
                    var = inst[-1]
                    self.definitions[def_id] = (block, inst)
                    if var not in self.defs_by_var:
                        self.defs_by_var[var] = set()
                    self.defs_by_var[var].add(def_id)
                    self.definition_vars[def_id] = var
                    def_id += 1

    def computeRD_gen_kill(self):
        """
        Compute the GEN and KILL sets for each basic block.
        """
        if not self.definitions:
            self.find_definitions()

        for block in self.rd_blocks:
            self.rd_gen[block] = set()
            self.rd_kill[block] = set()

            for def_id, (def_block, inst) in self.definitions.items():
                if def_block != block:
                    continue
                var = inst[-1]
                self.rd_gen[block].add(def_id)
                self.rd_kill[block] |= self.defs_by_var[var] - {def_id}

    def computeRD_in_out(self):
        """
        Compute the IN and OUT sets for each basic block.
        """
        changed = True
        while changed:
            changed = False
            for block in self.rd_blocks:
                old_out = self.rd_out[block].copy()
                # IN = U of OUT of predecessors
                new_in = set()
                for pred in block.predecessors:
                    new_in |= self.rd_out[pred]
                self.rd_in[block] = new_in
                # OUT = GEN U (IN - KILL)
                self.rd_out[block] = (
                    self.rd_gen[block] | (self.rd_in[block] - self.rd_kill[block])
                )

                if old_out != self.rd_out[block]:
                    changed = True

    def constant_propagation(self):
        """
        Perform constant propagation optimization on the current method.
        """
        for block in self.rd_blocks:
            reaching = self.rd_in[block].copy()
            for i, inst in enumerate(block.instructions):
                # get the operation
                opcode = inst[0]
                if opcode.endswith(":"):
                    continue
                new_inst = list(inst)
                for pos in range(1, len(inst)-1):
                    operand = inst[pos]
                    if not (isinstance(operand, str) and operand.startswith("%")):
                        continue
                    # Find all definitions that use this operand
                    reaching_defs = [d for d in reaching if self.definition_vars[d] == operand]
                    # Only propagate if there is exactly one reaching definition and it is a literal
                    if len(reaching_defs) != 1:
                        continue
                    d = reaching_defs[0]
                    _, def_inst = self.definitions[d]
                    if not def_inst[0].startswith("literal_"):
                        continue
                    literal = def_inst[1]
                    new_inst[pos] = literal
                # Update the instruction in the block
                block.instructions[i] = tuple(new_inst)

                # Update reaching definitions after processing the instruction
                if (
                    len(inst) >= 2
                    and isinstance(inst[-1], str)
                    and inst[-1].startswith("%")
                ):
                    # The instruction defines a new variable, so we need to update the reaching definitions
                    var = inst[-1]
                    reaching -= {d for d in reaching if self.definition_vars[d] == var}
                    for d, (_, definition) in self.definitions.items():
                        if definition is inst:
                            reaching.add(d)
                            break

    def buildLV_blocks(self, cfg: CFG):
        """
        Collect all CFG blocks and initialize liveness sets.
        """
        block = cfg
        while block is not None:
            # initialize liveness sets for the block
            self.lv_use[block] = set()
            self.lv_def[block] = set()
            self.lv_in[block] = set()
            self.lv_out[block] = set()
            # move to the next block
            block = block.next_block

    def computeLV_use_def(self):
        """
        Compute the USE and DEF sets for each basic block.
        """
        for block in self.lv_use:
            use = set()
            defs = set()
            for inst in block.instructions:
                # ignore labels (entry:, if.then:, ...)
                if inst[0].endswith(":"):
                    continue

                # get the operation
                op = inst[0].split("_")[0]

                # Variables used in the instruction are all operands except the last one (which is the destination)
                for operand in inst[1:-1]:
                    if (
                        isinstance(operand, str)
                        and operand.startswith("%")
                        and operand not in defs
                    ):
                        use.add(operand)

                # These operations do not define a new variable, so we skip them
                if op not in {"jump", "cbranch", "print", "return", "param", "store"}:
                    if (
                        len(inst) >= 2
                        and isinstance(inst[-1], str)
                        and inst[-1].startswith("%")
                    ):
                        defs.add(inst[-1])

                # "store" operation defines a variable (the last operand), so we add it to defs
                if op == "store":
                    defs.add(inst[-1])

            self.lv_use[block] = use
            self.lv_def[block] = defs

    def computeLV_in_out(self):
        changed = True
        while changed:
            changed = False
            # Compute the IN and OUT sets for each basic block in reverse order
            for block in reversed(list(self.lv_use.keys())):
                old_in = self.lv_in[block].copy()
                self.lv_out[block] = set()
                for succ in block.successors:
                    self.lv_out[block] |= self.lv_in[succ]
                self.lv_in[block] = (
                    self.lv_use[block]
                    |
                    (self.lv_out[block] - self.lv_def[block])
                )
                if old_in != self.lv_in[block]:
                    changed = True

    def deadcode_elimination(self):
        """
        Remove dead assignments using liveness analysis.
        """
        for block in self.lv_use:
            live = self.lv_out[block].copy()
            new_instructions = []

            # Iterate over the instructions in reverse order to determine which ones are live
            for inst in reversed(block.instructions):
                # Ignore labels (entry:, if.then:, ...)
                if inst[0].endswith(":"):
                    new_instructions.append(inst)
                    continue

                op = inst[0].split("_")[0]

                # Instructions that are always live
                if op in {"jump","cbranch","return","print","call","store"}:
                    new_instructions.append(inst)
                    # add operands to live set
                    for operand in inst[1:-1]:
                        if isinstance(operand, str) and operand.startswith("%"):
                            live.add(operand)
                    continue

                # Check if the instruction defines a variable that is not live
                defines = (
                    len(inst) >= 2
                    and isinstance(inst[-1], str)
                    and inst[-1].startswith("%")
                )

                if defines:
                    dest = inst[-1]
                    # dead definition, skip it
                    if dest not in live:
                        continue
                    live.discard(dest)

                # Add operands to live set
                for operand in inst[1:-1]:
                    if isinstance(operand, str) and operand.startswith("%"):
                        live.add(operand)
                # Add the instruction to the new list
                new_instructions.append(inst)

            # Reverse the new instructions to restore original order
            # NOTE: Necessary?
            # block.instructions = list(reversed(new_instructions))

    def merge_blocks(self, cfg: CFG):
        """
        Merge basic blocks in the CFG when possible.
        NOTE: not sure if this analysis is correct. Maybe implement a way that do not use predecessors and successors?
        """
        block = cfg
        while block is not None:
            if (
                len(block.successors) == 1
                and len(block.successors[0].predecessors) == 1
                and block.successors[0] is not block
            ):
                # Merge the successor block into the current block
                succ = block.successors[0]
                # drop the jump that connected block to succ: control now
                # falls through into the merged instructions.
                if (
                    block.instructions
                    and block.instructions[-1][0] == "jump"
                    and block.instructions[-1][1].lstrip("%") == succ.label
                ):
                    block.instructions.pop()
                block.instructions.extend(succ.instructions)
                block.successors = succ.successors
                for s in succ.successors:
                    s.predecessors.remove(succ)
                    s.predecessors.append(block)
                # unlink succ from the next_block chain so EmitBlocks does
                # not emit its instructions a second time.
                prev = cfg
                while prev is not None and prev.next_block is not succ:
                    prev = prev.next_block
                if prev is not None:
                    prev.next_block = succ.next_block
            else:
                block = block.next_block

    def short_circuit_jumps(self, cfg: CFG):
        """
        Short-circuit jumps in the CFG when possible.
        """
        block = cfg
        while block is not None:
            if len(block.instructions) > 0 and block.instructions[-1][0] == "jump":
                target_label = block.instructions[-1][1]
                target_block = block.next_block
                while target_block is not None and target_block.label != target_label:
                    target_block = target_block.next_block
                if (
                    target_block is not None
                    and len(target_block.instructions) > 0
                    and target_block.instructions[0][0] == "jump"
                ):
                    # Short-circuit the jump to the next jump's target
                    new_target_label = target_block.instructions[0][1]
                    block.instructions[-1] = ("jump", new_target_label)
            block = block.next_block

    def discard_unused_allocs(self, cfg: CFG):
        """
        Discard unused allocations in the CFG.

        An alloc is only safe to remove when its register is never referenced
        anywhere else (no store, load or operand uses it). Using the block's
        lv_out is wrong: a variable can be used within a block yet dead at its
        exit, and dropping its alloc while a store to it remains makes the
        interpreter resolve the target address to None.
        """
        # Collect every register referenced by a non-alloc instruction.
        referenced = set()
        block = cfg
        while block is not None:
            for inst in block.instructions:
                if inst[0].startswith("alloc_"):
                    continue
                for operand in inst[1:]:
                    if isinstance(operand, str) and operand.startswith("%"):
                        referenced.add(operand)
            block = block.next_block

        block = cfg
        while block is not None:
            new_instructions = []
            for inst in block.instructions:
                if inst[0].startswith("alloc_"):
                    var = inst[-1]
                    if var not in referenced:
                        continue
                new_instructions.append(inst)
            block.instructions = new_instructions
            block = block.next_block

    def appendOptimizedCode(self, cfg: CFG):
        """
        Append the optimized instructions of a CFG to self.code.
        """
        bb = EmitBlocks()
        bb.visit(cfg)
        self.code.extend(bb.code)


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate MJIR. By default, this script runs the interpreter on the optimized MJIR \
              and shows the speedup obtained from comparing original MJIR with its optimized version.",
        type=str,
    )
    parser.add_argument(
        "--opt",
        help="Print optimized MJIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--opt-pp",
        help="Print optimized MJIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--speedup",
        help="Show speedup from comparing original MJIR with its optimized version.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized MJIR for each function in pdf format",
        action="store_true",
    )
    args = parser.parse_args()

    speedup = args.speedup
    print_opt_ir = args.opt
    print_opt_ir_pp = args.opt_pp
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

    gen = CodeGenerator(False)
    gen.visit(ast)
    gencode = gen.code
    ast.text = gen.text

    opt = DataFlow(create_cfg)
    opt.visit(ast)
    optcode = opt.code
    if print_opt_ir:
        print("Optimized MJIR: --------")
        rich.print(optcode)
        print("------------------------\n")

    elif print_opt_ir_pp:
        print("Optimized MJIR: --------")
        opt.show()
        print("------------------------\n")

    speedup = len(gencode) / len(optcode)
    sys.stderr.write(
        "[SPEEDUP] Default: %d Optimized: %d Speedup: %.2f\n\n"
        % (len(gencode), len(optcode), speedup)
    )

    if not (print_opt_ir or print_opt_ir_pp):
        vm = MJIRInterpreter()
        vm.run(optcode)


if __name__ == "__main__":
    main()
