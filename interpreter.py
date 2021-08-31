#############################################################
#                                                           #
#                          Lexer                            #
#                                                           #
#############################################################

# 标识符类型
from os import name
from typing import List


REAL_CONST = 'REAL_CONST'
INTEGER_CONST = 'INTEGER_CONST'
INTEGER = 'INTEGER'
REAL = 'REAL'
PROGRAM = 'PROGRAM'
PROCEDURE = 'PROCEDURE'
VAR = 'VAR'
EOF = 'EOF'
PLUS = 'PLUS'
MINUS = 'MINUS'
MUL = 'MUL'
INTEGER_DIV = 'INTEGER_DIV'
FLOAT_DIV = 'FLOAT_DIV'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
BEGIN = 'BEGIN'
END = 'END'
DOT = 'DOT'
COMMA = 'COMMA'
COLON = 'COLON'
SEMI = 'SEMI'
ID = 'ID'
ASSIGN = 'ASSIGN'

# 标识符
class Token(object):
    def __init__(self, type, value):
        # token type: INTEGER_CONST, MUL, DIV, or EOF
        self.type = type
        # token value: non-negative INTEGER_CONST value, '*', '/', or None
        self.value = value

    def __str__(self):
        """String representation of the class instance.
        Examples:
            Token(INTEGER_CONST, 3)
            Token(MUL, '*')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()

# 词法分析器
class Lexer(object):
    RESERVED_KEYWORDS = {
        'BEGIN': Token('BEGIN', 'BEGIN'),
        'END': Token('END', 'END'),
        'PROGRAM': Token('PROGRAM', 'PROGRAM'),
        'VAR': Token('VAR', 'VAR'),
        'INTEGER': Token('INTEGER', 'INTEGER'),
        'REAL': Token('REAL', 'REAL'),
        'DIV' : Token('INTEGER_DIV', 'DIV'),
        'PROCEDURE' : Token('PROCEDURE', 'PROCEDURE')
    }

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def _id(self):
        '''If it's not a reserved keyword, it returns a new ID token whose value is the character string.'''
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()
        token = self.RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token

    def error(self):
        raise Exception('Invalid character')

    def get_next_token(self):
        while(self.current_char is not None):
            if self.current_char == '{':
                self.advance()
                self.skip_comment()
                continue

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(FLOAT_DIV, '/')

            if self.current_char == 'DIV':
                self.advance()
                return Token(INTEGER_DIV, 'DIV')
            
            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char.isalpha():
                return self._id()
            
            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(ASSIGN, ':=')

            if self.current_char == ':':
                self.advance()
                return Token(COLON, ':')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            self.error()

        return Token(EOF, None)
         
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != '}':
            self.advance()
        self.advance()

    def number(self):
        """Return a (multidigit) INTEGER_CONST or REAL_CONST consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        if self.current_char == '.':
            result += '.'
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            return Token(REAL_CONST, float(result))
        else:
            return Token(INTEGER_CONST, int(result))

    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
    
    def peek(self):
        """peek into the input buffer without consuming the next character."""
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else :
            return self.text[peek_pos]

#############################################################
#                                                           #
#                          Parser                           #
#                                                           #
#############################################################

class AST(object):
    pass

class BinOp(AST):
    """Binary operator node"""
    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.token = self.op = op

class Num(AST):
    """Number node"""
    def __init__(self, token:Token):
        self.token = token
        self.value = token.value

class UnaryOp(AST):
    """Unary operator node"""
    def __init__(self, op, right):
        self.op = op
        self.right = right

class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self):
        self.children = []

class Assign(AST):
    def __init__(self, left, assign_token:Token, right):
        self.left = left
        self.assign_token = assign_token
        self.right = right

class Var(AST):
    def __init__(self, var_token:Token):
        self.var_token = var_token
        self.value = var_token.value

class NoOp(AST):
    def __init__(self):
        pass

class Program(AST):
    def __init__(self, name:str, block_node):
        self.name = name
        self.block = block_node

class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement

class Type(AST):
    def __init__(self, token:Token):
        self.token = token
        self.value = token.value

class VarDecl(AST):
    def __init__(self, var_node:Var, type_node:Type):
        self.var_node = var_node
        self.type_node = type_node

class Param(AST):
    def __init__(self, var_node:Var, type_node:Type) -> None:
        self.var_node = var_node
        self.type_node = type_node

    def __str__(self) -> str:
        return ' {name}:{type} '.format(
            name = self.var_node.value,
            type = self.type_node.value,
        )

class ProcedureDecl(AST):
    def __init__(self, name:str, params, block_node:Block) -> None:
        self.name = name
        self.params = params  # A list of Param nodes
        self.block_node = block_node


# 语法分析器
class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def expr(self):
        """expr   : term ((PLUS | MINUS) term)*"""
        node = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self):
        """term : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            op_token = self.current_token
            if op_token.type == MUL:
                self.eat(MUL)
            elif op_token.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif op_token.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)
            node = BinOp(left=node, op=op_token, right=self.factor())

        return node

    def factor(self):
        """factor : INTEGER_CONST | LPAREN expr RPAREN | (PLUS | MINUS) factor | variable"""
        token = self.current_token
        if token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        elif token.type in (PLUS, MINUS):
            if token.type == PLUS:
                self.eat(PLUS)
                node = self.factor()
                return UnaryOp(op=token, right=node)
            elif token.type == MINUS:
                self.eat(MINUS)
                node = self.factor()
                return UnaryOp(op=token, right=node)
        elif token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return Num(token)
        elif token.type == ID:
            return self.variable()
        self.error()

    def program(self):
        node = self.compound_statement()
        self.eat(DOT)
        return node

    def compound_statement(self) -> Compound:
        """
        compound_statement: BEGIN statement_list END
        """
        self.eat(BEGIN)
        nodes = self.statement_list()
        self.eat(END)

        root = Compound()
        for node in nodes:
            root.children.append(node)
        return root

    def statement_list(self):
        """
        statement_list : statement
                    | statement SEMI statement_list
        """
        node = self.statement()
        result = [node]
        while self.current_token.type == SEMI:
            self.eat(SEMI)
            result.append(self.statement())

        if self.current_token.type == ID:
            self.error()

        return result

    def statement(self):
        if self.current_token.type == BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == ID:
            node = self.assignment_statement()
        else:
            node = self.empty()
        return node

    def assignment_statement(self):
        """
        assignment_statement : variable ASSIGN expr
        """
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr()
        return Assign(left=left, assign_token=token, right=right)

    def variable(self) -> Var:
        """
        variable : ID
        """
        token = self.current_token
        self.eat(ID)
        return Var(token)

    def empty(self):
        """An empty production"""
        return NoOp()

    def program(self) -> Program:
        self.eat(PROGRAM)
        var_node = self.variable()
        self.eat(SEMI)
        node = self.block()
        self.eat(DOT)
        return Program(name=var_node.value, block_node=node)

    def block(self) -> Block:
        decl_node = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declarations=decl_node, compound_statement=compound_statement_node)

    def declarations(self) -> List:
        """declarations : (VAR (variable_declaration SEMI)+)*
                    | (PROCEDURE ID (LPAREN formal_parameter_list RPAREN)? SEMI block SEMI)*
                    | empty
        """
        declarations = []
        # peocess varialble declaration
        if self.current_token.type == VAR:
            self.eat(VAR)
            while self.current_token.type == ID:
                declarations.extend(self.variable_declaration())
                self.eat(SEMI)
        # process procedure delaration and definition
        while self.current_token.type == PROCEDURE:
            self.eat(PROCEDURE)
            proc_name = self.current_token.value
            self.eat(ID)
            params = []
            if self.current_token.type == LPAREN:  # If formal params exists, get them
                self.eat(LPAREN)
                params.extend(self.formal_parameter_list())
                self.eat(RPAREN)
            self.eat(SEMI)
            block_node = self.block()
            declarations.append(ProcedureDecl(name=proc_name, params=params, block_node=block_node))
            self.eat(SEMI)
        return declarations


    def formal_parameter_list(self) -> list:
        """ formal_parameter_list : formal_parameters
                              | formal_parameters SEMI formal_parameter_list
        """
        params = []
        params.extend(self.formal_parameters())
        while self.current_token.type == SEMI:
            self.eat(SEMI)
            params.extend(self.formal_parameters())
        return params


    def formal_parameters(self) -> list:
        """formal_parameters: ID (COMMA ID)* COLON type_spec"""
        params = []
        if self.current_token.type == ID:
            var_list = []
            var_node_temp = Var(self.current_token)
            var_list.append(var_node_temp)
            self.eat(ID)
            while self.current_token.type == COMMA: # If exists more than two params, save them though List
                self.eat(COMMA)
                var_node_temp = Var(self.current_token)
                self.eat(ID)
                var_list.append(var_node_temp)
            self.eat(COLON)
            type_node = self.type_spec()
            params = [
                Param(var_node=var_node, type_node=type_node)
                for var_node in var_list
            ]
        return params


    def variable_declaration(self) -> list:
        var_nodes = [self.variable()]
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(self.variable())
        self.eat(COLON)
        type_node = self.type_spec()
        var_declarations = [
            VarDecl(var_node=var_node, type_node=type_node)
            for var_node in var_nodes
        ]
        return var_declarations


    def type_spec(self) -> Type:
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
        else:
            self.eat(REAL)
        return Type(token=token)

    def parse(self):
        node = self.program()
        if self.current_token.type != EOF:
            self.error()

        return node



##################### AST node visitor ######################
class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


#############################################################
#                                                           #
#                Symbol, Scope Symbol Table                 #
#                                                           #
#############################################################
# Symbol
class Symbol(object):
    def __init__(self, name, type=None):
        self.name = name
        self.type = type
    

class BuildinTypeSymbol(Symbol):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<{class_name}(name='{name}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
        )


class VarSymbol(Symbol):
    def __init__(self, name:str, type:BuildinTypeSymbol):
        super().__init__(name, type=type)

    def __str__(self):
        return "<{class_name}(name='{name}', type='{type}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            type=self.type,
        )

    __repr__ = __str__


class ProcedureSymbol(Symbol):
    def __init__(self, name, params=None):
        super().__init__(name)
        self.params = params if params is not None else []

    def __str__(self) -> str:
        return '<{class_name}(name={name}, parameters={params})>'.format(
            class_name = self.__class__.__name__,
            name = self.name,
            params = self.params,
        )

    __repr__ = __str__


# Symbol Table
class ScopedSymbolTable(object):
    def __init__(self, scope_level, scope_name, enclosing_scope=None) -> None:
        self._symbols = {}
        self._init_buildins()
        self.scope_level = scope_level
        self.scope_name = scope_name
        self.enclosing_scope = enclosing_scope

    def _init_buildins(self) -> None:
        """Initialize build-in symbols"""
        self.insert(BuildinTypeSymbol(INTEGER))
        self.insert(BuildinTypeSymbol(REAL))

    def insert(self, symbol) -> None:
        print('Insert: %s' % symbol)
        self._symbols[symbol.name] = symbol

    def lookup(self, name) -> Symbol:
        print('Lookup: %s' % name)
        return self._symbols.get(name)
    
    def __str__(self):
        symtab_header = 'Scope Symbol Table Contents'
        lines = ['\n', symtab_header, '=' * len(symtab_header)]
        lines.append('%-15s: %s' % ('Scope Name', self.scope_name))
        lines.append('%-15s: %s' % ('Scope Level', self.scope_level))
        lines.append('%-15s: %s' % ('Enclosing Scope', self.enclosing_scope.scope_name if self.enclosing_scope else None))
        lines.append('-' * len(symtab_header))
        lines.extend(
            ('%-15s: %r' % (key, value))
            for key, value in self._symbols.items()
        )
        lines.append('=' * len(symtab_header))
        s = '\n'.join(lines)
        return s

    __repr__ = __str__



#############################################################
#                                                           #
#                       Interpreter                         #
#                                                           #
#############################################################
class Interpreter(NodeVisitor):
    GLOBAL_SCOPE = {}

    def __init__(self, parser):
        self.parser = parser
        self.symtab = None

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == FLOAT_DIV:
            return float(self.visit(node.left) / self.visit(node.right))

    def visit_Num(self, node):
        return node.value


    def visit_UnaryOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.right)
        elif node.op.type == MINUS:
            return 0 - self.visit(node.right)


    def visit_Compound(self, nodes):
        for node in nodes.children:
            self.visit(node)


    def visit_Assign(self, node):
        left = node.left
        left_var_symbol = self.symtab.lookup(left.value)
        if left_var_symbol is None:
            raise NameError(repr(left.value))
        else:
            right = node.right
            self.GLOBAL_SCOPE[left.value] = self.visit(right)


    def visit_Var(self, node:Var):
        var_name = node.value
        # Verify variable wether is declared before
        var_symbol = self.symtab.lookup(var_name)
        if var_symbol is None:
            raise NameError(repr(var_name))
        

    def visit_NoOp(self, node):
        pass


    def visit_Program(self, node:Program):
        print('ENTER scope: global')
        # Create scope table
        global_scope = ScopedSymbolTable(
            scope_name='global',
            scope_level=1,
            enclosing_scope=self.symtab,
        )
        self.symtab = global_scope

        # Visit subtree
        self.visit(node.block)

        print(global_scope)
        # The scope tree constructions will be
        # broken when we have more than
        # two scopes in our program
        self.symtab = self.symtab.enclosing_scope
        print('LEAVE scope: global')


    def visit_Block(self, node:Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)


    def visit_VarDecl(self, node:VarDecl):
        """Put the variable into symbol table."""
        var_name = node.var_node.value
        var_type_symbol = self.symtab.lookup(node.type_node.value)
        var_symbol = VarSymbol(var_name, var_type_symbol)
        self.symtab.insert(var_symbol)


    def visit_Type(self, node):
        pass


    def visit_ProcedureDecl(self, node:ProcedureDecl):
        print('ENTER scope: %s' % node.name)

        # Insert procedure name as parameter into symbol table of Main scope 
        proc_symbol = ProcedureSymbol(name=node.name)
        self.symtab.insert(proc_symbol)

        # Create scope table for current procedure
        proc_scope = ScopedSymbolTable(
            scope_name=node.name,
            scope_level=self.symtab.scope_level + 1,
            enclosing_scope=self.symtab,
        )
        self.symtab = proc_scope

        # Visit every param node
        params = node.params
        for param in params:
            self.visit(param)
          
        # Visit block node
        self.visit(node.block_node)

        print(proc_scope)
        
        # reset the value of current scope
        self.symtab = self.symtab.enclosing_scope
        print('LEAVE scope: %s' % node.name)


    def visit_Param(self, node:Param):
        # Insert parameters into the procedure scope
        var_name = node.var_node.value
        var_type_name = node.type_node.value
        symbol = VarSymbol(name=var_name, type=self.symtab.lookup(var_type_name))
        self.symtab.insert(symbol)




def main():
    # url = input('simple> ')
    url = '/Users/lishanqiu/vscode-projects/vscode-python/simple-interpreter/test.pas'
    with open(url, 'r', encoding='utf-8') as f:
        text = f.read()
        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        interpreter.interpret()
        

if __name__ == '__main__':
    main()