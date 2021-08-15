#############################################################
#                                                           #
#                          Lexer                            #
#                                                           #
#############################################################

# 标识符类型
INTEGER_CONST = 'INTEGER_CONST'
INTEGER, REAL = 'INTEGER', 'REAL'
PROGRAM, VAR, EOF = 'PROGRAM', 'VAR', 'EOF'
PLUS, MINUS, MUL, DIV = '+', '-', '*', '/'
LPAREN, RPAREN = '(', ')'
BEGIN, END, DOT, COMMA, COLON, SEMI ,ID = 'BEGIN', "END", '.', ',', ':', ';', 'ID'
ASSIGN = ':='


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
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(INTEGER_CONST, self.integer_const())

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')
            
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

    def integer_const(self):
        """Return a (multidigit) INTEGER_CONST consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

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
    def __init__(self, token):
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
    def __init__(self, left, assign_token, right):
        self.left = left
        self.assign_token = assign_token
        self.right = right

class Var(AST):
    def __init__(self, var_token):
        self.var_token = var_token
        self.value = var_token.value

class NoOp(AST):
    def __init__(self):
        pass

class Program(AST):
    def __init__(self, name, block_node):
        self.name = name
        self.block_node = block_node

class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement

class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

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
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (MUL, DIV):
            op_token = self.current_token
            if op_token.type == MUL:
                self.eat(MUL)
            elif op_token.type == DIV:
                self.eat(DIV)
            node = BinOp(left=node, op=op_token, right=self.factor())

        return node

    def factor(self):
        """factor : INTEGER_CONST | LPAREN expr RPAREN | (MUL | DIV) factor"""
        token = self.current_token
        if token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == LPAREN:
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
        elif token.type == ID:
            return self.variable()
        self.error()

    def program(self):
        node = self.compound_statement()
        self.eat(DOT)
        return node

    def compound_statement(self):
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

    def variable(self):
        """
        variable : ID
        """
        token = self.current_token
        self.eat(ID)
        return Var(token)

    def empty(self):
        """An empty production"""
        return NoOp()

    def program(self):
        self.eat(PROGRAM)
        var_node = self.variable()
        self.eat(SEMI)
        node = self.block()
        self.eat(DOT)
        return Program(name=var_node.value, block_node=node)

    def block(self):
        decl_node = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declarations=decl_node, compound_statement=compound_statement_node)

    def declarations(self):
        declarations = []
        if self.current_token.type == VAR:
            self.eat(VAR)
            while self.current_token.type == ID:
                declarations.extend(self.variable_declaration())
                self.eat(SEMI)
            return declarations

    def variable_declaration(self):
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

    def type_spec(self):
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

#############################################################
#                                                           #
#                       Interpreter                         #
#                                                           #
#############################################################
class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    GLOBAL_SCOPE = {}

    def __init__(self, parser):
        self.parser = parser

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
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

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
        right = node.right
        self.GLOBAL_SCOPE[left.value] = self.visit(right)

    def visit_Var(self, node):
        var_name = node.value
        var_value = self.GLOBAL_SCOPE.get(var_name)
        if var_value is None:
            raise NameError(repr(var_name))
        else:
            return var_value

    def visit_NoOp(self, node):
        pass

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Block(self, node):
        for var_decl_node in node.declarations:
            self.visit(var_decl_node)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        pass

    def visit_Type(self, node):
        pass


def main():
    url = input('simple> ')
    url = '/Users/lishanqiu/vscode-projects/vscode-python/simple-interpreter/test.pas'
    with open(url, 'r', encoding='utf-8') as f:
        text = f.read()
        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        interpreter.interpret()
        print(interpreter.GLOBAL_SCOPE)



if __name__ == '__main__':
    main()