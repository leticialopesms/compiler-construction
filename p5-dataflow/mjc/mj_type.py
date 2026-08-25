class MJType:
    """
    Class that represents a type in the MiniJava language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops

    def __str__(self):
        return f"type({self.typename})"


# Array & Object types need to be instantiated for each declaration
class ArrayType(MJType):
    def __init__(self, array_type: str, element_type: str, size=None):
        """
        :param array_type: Type of the array (int[] or char[])
        :param element_type: Type of the array elements (int or char)
        :param size: Integer with the length of the array.
        """
        self.size = size
        self.element_type = element_type
        super().__init__(name=array_type, rel_ops={"==", "!="}, assign_ops={"="})

class IntArrayType(ArrayType):
    typename = "int[]"

    def __init__(self, size=None):
        super().__init__(array_type="int[]", element_type="int", size=size)

class CharArrayType(ArrayType):
    typename = "char[]"

    def __init__(self, size=None):
        super().__init__(array_type="char[]", element_type="char", size=size)


# Create specific instances of basic types. You will need to add
# appropriate arguments depending on your definition of MJType
IntType = MJType(
    name="int",
    binary_ops={"+", "-", "*", "/", "%"},
    unary_ops={"-", "+"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

# In MiniJava, boolean is a primitive type that can only take the values true and false
BooleanType = MJType(
    name="boolean",
    binary_ops={"&&", "||"},
    unary_ops={"!"},
    rel_ops={"==", "!="},
    assign_ops={"="},
)

# In MiniJava, char is a 16-bit unsigned integer type
CharType = MJType(
    name="char",
    binary_ops={"*", "/", "%"},
    unary_ops={"-", "+"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="} ,
    assign_ops={"="},
)

StringType = MJType(
    name="String",
    binary_ops=set(),
    unary_ops=set(),
    rel_ops={"==", "!="},
    assign_ops={"="},
)

ObjectType = MJType(
    name="Object",
    binary_ops=set(),
    unary_ops=set(),
    rel_ops={"==", "!="},
    assign_ops={"="},
)

VoidType = MJType(
    name="void",
    binary_ops=set(),
    unary_ops=set(),
    rel_ops=set(),
    assign_ops=set(),
)