#############################################################
#                                                           #
#                          Lexer                            #
#                                                           #
#############################################################

# 标识符类型
INTEGER = 'INTEGER'
EOF = 'EOF'
PLUS, MINUS, MUL, DIV = '+', '-', '*', '/'
LPAREN, RPAREN = '(', ')'
BEGIN, END, DOT, ID = 'BEGIN', "END", '.', 'ID'
ASSIGN = ':='
SEMI = ';'

# 标识符
class Token(object):
    def __init__(self, type, value):
        # token type: INTEGER, MUL, DIV, or EOF
        self.type = type
        # token value: non-negative integer value, '*', '/', or None
        self.value = value

    def __str__(self):
        """String representation of the class instance.
        Examples:
            Token(INTEGER, 3)
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
                return Token(INTEGER, self.integer())

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

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            self.error()

        return Token(EOF, None)
            

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()


    def integer(self):
        """Return a (multidigit) integer consumed from the input."""
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
    def __init__(self, left, op, right):
        self.left = left
        self.right = right
        self.token = self.op = op

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class UnaryOp(AST):
    def __init__(self, op, right):
        self.op = op
        self.right = right

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
        """factor : INTEGER | LPAREN expr RPAREN | (MUL | DIV) factor"""
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
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
        self.error()


    def parse(self):
        return self.expr()


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
        

def main():
    while True:
        try:
            try:
                text = input('spi> ')
            except NameError:  # Python3
                text = input('spi> ')
        except EOFError:
            break
        if not text:
            continue

        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        result = interpreter.interpret()
        print(result)



if __name__ == '__main__':
    main()