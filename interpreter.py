import argparse
from enum import Enum
import sys
# enums allow us to access enum members by values

_SHOULD_LOG_SCOPE = False

#############################################################
#                                                           #
#                          Error                            #
#                                                           #
#############################################################
class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    ID_NOT_FOUND = 'Identifier not found'
    DUPLICATE_ID = 'Duplicate id found'

class Error(Exception):
    def __init__(self, error_code: ErrorCode=None, token=None, message: str= None) -> None:
        self.error_code = error_code
        self.token = token
        self.message = f'{self.__class__.__name__}: {message}'

class LexerError(Error):
    """Use to indicate an error encountered in the Lexer"""
    pass

class ParserError(Error):
    """ParserError is for syntax related error during the parsing phrase"""
    pass

class SemanticError(Error):
    """SemanticError is for semantic errors"""
    pass


#############################################################
#                                                           #
#                          Token                            #
#                                                           #
#############################################################
from os import error, name
from typing import List

class TokenType(Enum):
    # single-character token types
    PLUS          = '+'
    MINUS         = '-'
    MUL           = '*'
    FLOAT_DIV     = '/'
    LPAREN        = '('
    RPAREN        = ')'
    SEMI          = ';'
    DOT           = '.'
    COLON         = ':'
    COMMA         = ','
    # block of reserved words
    PROGRAM       = 'PROGRAM'  # marks the beginning of the block
    INTEGER       = 'INTEGER'
    REAL          = 'REAL'
    INTEGER_DIV   = 'DIV'
    VAR           = 'VAR'
    PROCEDURE     = 'PROCEDURE'
    BEGIN         = 'BEGIN'
    END           = 'END'      # marks the end of the block
    # misc
    ID            = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST    = 'REAL_CONST'
    ASSIGN        = ':='
    EOF           = 'EOF'


def _build_reserved_keywords() -> dict:
    token_type_list = list(TokenType)
    start_index = token_type_list.index(TokenType.PROGRAM)
    end_index = token_type_list.index(TokenType.END)
    reserved_keywords_list = {
        token_type.name: token_type
        for token_type in token_type_list[start_index: end_index+1]
    }
    return reserved_keywords_list


# Key: token name, Value: TokenType enum
RESERVED_KEYWORDS = _build_reserved_keywords()


class Token(object):
    def __init__(self, type: TokenType, value, lineno=None, column=None):
        # token type: INTEGER_CONST, MUL, DIV, or EOF
        self.type = type
        # token value: non-negative INTEGER_CONST value, '*', '/', or None
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        """String representation of the class instance.
        Examples:
            Token(INTEGER_CONST, 3), Position(5,10)
            Token(MUL, '*'), Position(3,6)
        """
        return 'Token({type}, {value}), Position({lineno}:{column})'.format(
            type=self.type.name,
            value=repr(self.value),
            lineno=self.lineno,
            column=self.column
        )

    def __repr__(self):
        return self.__str__()

#############################################################
#                                                           #
#                          Lexer                            #
#                                                           #
#############################################################
class Lexer(object):

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]
        # token line number and column number
        self.lineno = 1
        self.column = 1
    

    def _id(self):
        """
        If it's not a reserved keyword,
        it returns a new ID token whose value is the character string.
        """
        # Create a new token with current line and column number
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()
        
        token_type = RESERVED_KEYWORDS.get(result.upper()) # Use upper method to ignore case
        if token_type is None:
            # return id token
            token.type = TokenType.ID
            token.value = result
        else:
            # return reserved keyword
            token.type = token_type
            token.value = result.upper()

        return token


    def error(self):
        s = "Lexer error on '{lexme}' line: {lineno}, column: {column}".format(
            lexme = self.current_char,
            lineno = self.lineno,
            column = self.column,
        )
        raise LexerError(message=s)
    

    def advance(self):
        if self.current_char == '\n':
            self.lineno += 1
            self.column = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
            self.column += 1


    def get_next_token(self):
        """
        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
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
            
            if self.current_char.isalpha():
                return self._id()
            
            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.ASSIGN, ':=')

            # Single character token
            try:
                token_type = TokenType(self.current_char)
            except:
                self.error()
            else:
                token = Token(
                        type=token_type,
                        value=token_type.value,
                        lineno=self.lineno,
                        column=self.column
                    )
                self.advance()
                return token

        return Token(TokenType.EOF, None)
       
        
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()


    def skip_comment(self):
        while self.current_char != '}':
            self.advance()
        self.advance()


    def number(self):
        """
        Return a (multidigit) INTEGER_CONST or REAL_CONST consumed from the input.
        """
        # Token start position
        lineno = self.lineno
        column = self.column

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        # Token value's type is float
        if self.current_char == '.':
            result += '.'
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            return Token(
                    token = TokenType.REAL_CONST,
                    value = float(result),
                    lineno = lineno,
                    column = column
                )
        else:
            return Token(
                    type = TokenType.INTEGER_CONST, 
                    value = int(result),
                    lineno = lineno,
                    column = column
                )

    
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
        self.name = self.var_node.value
        self.type_node = type_node

    def __str__(self) -> str:
        return ' {name}:{type} '.format(
            name = self.var_node.value,
            type = self.type_node.value,
        )

    def __repr__(self) -> str:
        return self.__str__()


class ProcedureDecl(AST):
    def __init__(self, name:str, formal_params, block_node:Block) -> None:
        self.name = name
        self.formal_params = formal_params  # A list of Param nodes(formal paramters)
        self.block_node = block_node


class ProcedureCall(AST):
    def __init__(self, proc_name, actual_params, token, proc_symbol=None) -> None:
        self.proc_name = proc_name
        self.actual_params = actual_params  # a list of BinOp node(actual parameters)
        self.token = token
        # a reference to procedure declaration symbol
        self.proc_symbol = proc_symbol



# 语法分析器
class Parser(object):
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()


    def error(self, error_code:ErrorCode, token: Token):
        raise ParserError(
            error_code=error_code, 
            token=token, 
            message=f'{error_code.value} -> {token}',
        )


    def eat(self, token_type: TokenType):
        """
        compare the current token type with the passed token
        type and if they match then "eat" the current token
        and assign the next token to the self.current_token,
        otherwise raise an exception.
        """
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )


    def expr(self):
        """expr   : term ((PLUS | MINUS) term)*"""
        node = self.term()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)
            node = BinOp(left=node, op=token, right=self.term())

        return node


    def term(self):
        """term : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (TokenType.MUL, TokenType.INTEGER_DIV, TokenType.FLOAT_DIV):
            op_token = self.current_token
            if op_token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif op_token.type == TokenType.INTEGER_DIV:
                self.eat(TokenType.INTEGER_DIV)
            elif op_token.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)
            node = BinOp(left=node, op=op_token, right=self.factor())

        return node


    def factor(self):
        """factor : INTEGER_CONST | LPAREN expr RPAREN | (PLUS | MINUS) factor | variable"""
        token = self.current_token
        if token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        elif token.type in (TokenType.PLUS, TokenType.MINUS):
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
                node = self.factor()
                return UnaryOp(op=token, right=node)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)
                node = self.factor()
                return UnaryOp(op=token, right=node)
        elif token.type == TokenType.INTEGER_CONST:
            self.eat(TokenType.INTEGER_CONST)
            return Num(token)
        elif token.type == TokenType.REAL_CONST:
            self.eat(TokenType.REAL_CONST)
            return Num(token)
        elif token.type == TokenType.ID:
            return self.variable()
        self.error()


    def program(self):
        node = self.compound_statement()
        self.eat(TokenType.DOT)
        return node


    def compound_statement(self) -> Compound:
        """
        compound_statement: BEGIN statement_list END
        """
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        self.eat(TokenType.END)

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
        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            result.append(self.statement())

        if self.current_token.type == TokenType.ID:
            self.error()

        return result


    def statement(self):
        """
        statement : compound_statement
                   | procedure_call_statement
                   | assignment_statement
                   | empty
        """
        if self.current_token.type == TokenType.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == TokenType.ID and self.lexer.current_char == '(':
            node = self.procedure_call_statement()
        elif self.current_token.type == TokenType.ID:
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
        self.eat(TokenType.ASSIGN)
        right = self.expr()
        return Assign(left=left, assign_token=token, right=right)


    def variable(self) -> Var:
        """
        variable : ID
        """
        token = self.current_token
        self.eat(TokenType.ID)
        return Var(token)


    def empty(self):
        """An empty production"""
        return NoOp()


    def program(self) -> Program:
        self.eat(TokenType.PROGRAM)
        var_node = self.variable()
        self.eat(TokenType.SEMI)
        node = self.block()
        self.eat(TokenType.DOT)
        return Program(name=var_node.value, block_node=node)


    def block(self) -> Block:
        decl_node = self.declarations()
        compound_statement_node = self.compound_statement()
        return Block(declarations=decl_node, compound_statement=compound_statement_node)


    def declarations(self) -> list:
        """declarations : (VAR (variable_declaration SEMI)+) *
                    | procedure_declaration *
                    | empty
        """
        declarations = []

        # peocess varialble declaration
        if self.current_token.type == TokenType.VAR:
            self.eat(TokenType.VAR)
            while self.current_token.type == TokenType.ID:
                declarations.extend(self.variable_declaration())
                self.eat(TokenType.SEMI)

        # process procedure delaration and definition
        while self.current_token.type == TokenType.PROCEDURE:
            procedure_declaration_node = self.procedure_declaration()
            declarations.append(procedure_declaration_node)

        return declarations


    def procedure_declaration(self) -> ProcedureDecl:
        """
        procedure_declaration : (PROCEDURE ID (LPAREN formal_parameter_list RPAREN)? SEMI block SEMI)
        """
        self.eat(TokenType.PROCEDURE)
        procedure_name = self.current_token.value
        self.eat(TokenType.ID)

        params = []
        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            params.extend(self.formal_parameter_list())
            self.eat(TokenType.RPAREN)

        self.eat(TokenType.SEMI)
        block_node = self.block()
        self.eat(TokenType.SEMI)

        return ProcedureDecl(
                name=procedure_name, 
                formal_params=params, 
                block_node=block_node,
            )


    def formal_parameter_list(self) -> list:
        """ formal_parameter_list : formal_parameters
                              | formal_parameters SEMI formal_parameter_list
        """
        params = []
        params.extend(self.formal_parameters())
        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            params.extend(self.formal_parameters())
        return params


    def formal_parameters(self) -> list:
        """formal_parameters: ID (COMMA ID)* COLON type_spec"""
        params = []
        if self.current_token.type == TokenType.ID:
            var_list = []

            var_node_temp = Var(self.current_token)
            var_list.append(var_node_temp)
            self.eat(TokenType.ID)

            while self.current_token.type == TokenType.COMMA: # If exists more than two params, save them though List
                self.eat(TokenType.COMMA)
                var_node_temp = Var(self.current_token)
                self.eat(TokenType.ID)
                var_list.append(var_node_temp)
            
            self.eat(TokenType.COLON)
            type_node = self.type_spec()
            params = [
                Param(var_node=var_node, type_node=type_node)
                for var_node in var_list
            ]

        return params


    def variable_declaration(self) -> list:
        var_nodes = [self.variable()]

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            var_nodes.append(self.variable())
        
        self.eat(TokenType.COLON)
        
        type_node = self.type_spec()
        var_declarations = [
            VarDecl(var_node=var_node, type_node=type_node)
            for var_node in var_nodes
        ]
        
        return var_declarations


    def type_spec(self) -> Type:
        token = self.current_token
        if token.type == TokenType.INTEGER:
            self.eat(TokenType.INTEGER)
        else:
            self.eat(TokenType.REAL)
        return Type(token=token)


    def procedure_call_statement(self) -> ProcedureCall:
        """
        procedure_call_statement : ID LPAREN (expr (COMMA expr)*)* RPAREN
        """
        proc_token = self.current_token
        proc_name = self.current_token.value
        self.eat(TokenType.ID)
        self.eat(TokenType.LPAREN)
        actual_params = []

        while self.current_token.type != TokenType.RPAREN:
            binop_node = self.expr()
            actual_params.append(binop_node)
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                binop_node = self.expr()
                actual_params.append(binop_node)

        self.eat(TokenType.RPAREN)

        proc_call_node = ProcedureCall(
            proc_name=proc_name,
            actual_params=actual_params,
            token=proc_token,
        )
        return proc_call_node


    def parse(self):
        node = self.program()
        if self.current_token.type != TokenType.EOF:
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
#             Symbol, Table, SEMANTIC ANALYZER              #
#                                                           #
#############################################################

class Symbol(object):
    def __init__(self, name, type=None):
        self.name = name
        self.type = type
        self.scope_level = 0
    

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

    def __repr__(self) -> str:
        return self.__str__()


class ProcedureSymbol(Symbol):
    def __init__(self, name, formal_params=None):
        super().__init__(name)
        self.formal_params = formal_params if formal_params is not None else []
        # a reference to procedure's body (AST sub-tree)
        self.block_ast = None


    def __str__(self) -> str:
        return '<{class_name}(name={name}, parameters={params})>'.format(
            class_name = self.__class__.__name__,
            name = self.name,
            params = self.formal_params,
        )

    def __repr__(self) -> str:
        return self.__str__()


class ScopedSymbolTable(object):
    def __init__(self, scope_level, scope_name, enclosing_scope=None, block_ast=None) -> None:
        #key: symbol name, value: symbol object
        self._symbols = {}
        self.scope_level = scope_level
        self.scope_name = scope_name
        self.enclosing_scope = enclosing_scope
        # a reference to procedure's body (AST sub-tree)
        self.block_ast = block_ast
        self._init_buildins()


    def _init_buildins(self) -> None:
        """Initialize build-in symbols"""
        self.insert(BuildinTypeSymbol(TokenType.INTEGER))
        self.insert(BuildinTypeSymbol(TokenType.REAL))


    def insert(self, symbol:Symbol) -> None:
        self.log(f'insert: {symbol.name:10} (Scope name: {self.scope_name})')
        symbol.scope_level = self.scope_level
        self._symbols[symbol.name] = symbol


    def lookup(self, name, current_scope_only=False) -> Symbol:
        self.log(f'lookup: {name:10} (Scope name: {self.scope_name})')

        # Find in the current scope symbol table
        symbol = self._symbols.get(name)
        if symbol is not None:
            return symbol
        else:
            if self.enclosing_scope is not None and current_scope_only == False:
                # Find in parant scope symbol table
                return self.enclosing_scope.lookup(name)
            elif current_scope_only == True:
                return symbol
    

    def log(self, msg):
        """
        The message will be printed only if the global 
        variable _SHOULD_LOG_SCOPE is set to true
        """
        if _SHOULD_LOG_SCOPE:
            print(msg)


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
        lines.append('\n')
        s = '\n'.join(lines)
        return s


    def __repr__(self) -> str:
        return self.__str__()


class SemanticAnalyzer(NodeVisitor):
    def __init__(self) -> None:
        self.current_scope = None


    def analyze(self, tree):
        self.visit(tree)
    

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)


    def error(self, error_code, token):
        raise SemanticError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}',
        )

    def visit_Program(self, node: Program):
        program_name = node.name
        self.log(f'ENTER SCOPE: {program_name}')

        program_scope = ScopedSymbolTable(
            scope_level=1,
            scope_name=program_name,
            enclosing_scope=None,
        )
        self.current_scope = program_scope

        self.visit(node.block)

        self.current_scope = None
        self.log(f'LEAVE SCOPE: {program_name}')



    def visit_Block(self, node: Block):
        for decl in node.declarations:
            self.visit(decl)
        self.visit(node.compound_statement)

    
    def visit_VarDecl(self, node: VarDecl):
        var_name = node.var_node.value
        var_type_name = node.type_node.value
        var_type_symbol = self.current_scope.lookup(var_type_name, current_scope_only=True)

        if self.current_scope.lookup(var_name, current_scope_only=True):
            self.error(
                error_code=ErrorCode.DUPLICATE_ID,
                token=node.var_node.var_token
            )
        else:
            var_symbol = VarSymbol(
                            name=var_name,
                            type=var_type_symbol,
                        )
            self.current_scope.insert(var_symbol)


    def visit_ProcedureDecl(self, node: ProcedureDecl):
        proc_name = node.name

        # Create Procedure Symbol
        params = node.formal_params
        proc_symbol = ProcedureSymbol(proc_name, params)
        self.current_scope.insert(proc_symbol)

        self.log(f'ENTER SCOPE {proc_name}')

        # Create Procedure Scope Table
        proc_scope = ScopedSymbolTable(
            scope_level=self.current_scope.scope_level + 1,
            scope_name=proc_name,
            enclosing_scope=self.current_scope,
            block_ast=None,
        )

        # Push procedure formal paramters in Procedure Scope Table
        for param in params:
            name = param.var_node.value
            type_name = param.type_node.value
            param_symbol = VarSymbol(name, self.current_scope.lookup(type_name))
            proc_scope.insert(param_symbol)

        self.log(self.current_scope.__str__())
        self.current_scope = proc_scope

        self.visit(node.block_node)
        # A reference to procedure's block body
        proc_symbol.block_ast = node.block_node

        self.log(self.current_scope.__str__())
        self.current_scope = self.current_scope.enclosing_scope

        self.log(f'LEAVE SCOPE : {proc_name}')


    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)


    def visit_Assign(self, node: Assign):
        self.visit(node.left)
        self.visit(node.right)


    def visit_Var(self, node: Var):
        var_name = node.value
        var_symbol = self.current_scope.lookup(var_name)
        if var_symbol is None:
            self.error(
                error_code=ErrorCode.ID_NOT_FOUND,
                token=node.var_token
            )

    
    def visit_BinOp(self, node: BinOp):
        pass


    def visit_Num(self, node: Num):
        pass


    def visit_UnaryOp(self, node: UnaryOp):
        pass


    def visit_ProcedureCall(self, node: ProcedureCall):
        for param in node.actual_params:
            self.visit(param)

        # Linking to the procedure block node.
        node.proc_symbol = self.current_scope.lookup(node.proc_name)


    def visit_NoOp(self, node):
        pass



#############################################################
#                                                           #
#                       Interpreter                         #
#                                                           #
#############################################################
class ARType(Enum):
    PROGRAM = 'PROGRAM'
    PEOCEDURE = 'PROCEDURE'


class ActivationRecord:
    '''
    For only one activation record
    '''
    def __init__(self, name: str, type: ARType, nesting_level) -> None:
        self.name = name  # the name of the AR
        self.type = type  # the type of the AR(for example, PROGRAM)
        self.nesting_level = nesting_level  # corresponding to the scope level of the repective procedure or function
        self.members = {}  # Used for keeping info about a particular invocation of a routine

    def __setitem__(self, key, value):
        """
        to give activation record objects a dictionary-like 
        interface for storing key-value pairs
        """
        self.members[key] = value

    def __getitem__(self, key):
        return self.members[key]

    def get(self, key):
        """
        Method will return None if the key doesn't exist in members dict
        """
        return self.members.get(key)

    def __str__(self) -> str:
        lines = [
            '{level}: {type} {name}'.format(
                level=self.nesting_level,
                type=self.type.value,
                name=self.name,
            )
        ]
        for name, value in self.members.items():
            lines.append(f'    {name:<20}: {value}')

        s = '\n'.join(lines)
        return s

    def __repr__(self) -> str:
        return self.__str__()
        

class CallStack:
    def __init__(self) -> None:
        self._records = []

    def push(self, ar:ActivationRecord):
        # ar means activation record
        self._records.append(ar)

    def pop(self):
        return self._records.pop()

    def peek(self):
        return self._records[-1]

    def __str__(self):
        s = '\n'.join(repr(ar) for ar in reversed(self._records))
        s = f'CALL STACK\n{s}\n'
        return s

    def __repr__(self) -> str:
        return self.__str__()


class Interpreter(NodeVisitor):

    def __init__(self, parser):
        self.parser = parser
        self.call_stack = CallStack()


    def interpret(self, tree):
        return self.visit(tree)


    def error(self, error_code: ErrorCode, token: Token):
        raise SemanticError(
            error_code=error_code, 
            token=token,
            message=f'{error_code.value} -> {token}',
        )


    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)


    def visit_BinOp(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == TokenType.FLOAT_DIV:
            return float(self.visit(node.left) / self.visit(node.right))


    def visit_Num(self, node):
        return node.value


    def visit_UnaryOp(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return 0 - self.visit(node.right)


    def visit_Compound(self, nodes):
        for node in nodes.children:
            self.visit(node)


    def visit_Assign(self, node):
        left = node.left
        right = node.right
        ar = self.call_stack.peek()
        ar[left.value] = self.visit(right)


    def visit_Var(self, node:Var):
        var_name = node.value
        ar = self.call_stack.peek()
        var_value = ar.get(var_name)
        return var_value
        

    def visit_NoOp(self, node):
        pass


    def visit_Program(self, node:Program):
        program_name = node.name
        self.log(f'ENTER: PROGRAM {program_name}')

        ar = ActivationRecord(
            name = program_name,
            type = ARType.PROGRAM,
            nesting_level = 1,
        )
        self.call_stack.push(ar=ar)
        self.visit(node.block)
        self.call_stack.pop()

        self.log(f'LEAVE: PROGRAM {program_name}')


    def visit_Block(self, node:Block):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)


    def visit_VarDecl(self, node:VarDecl):
        pass
    

    def visit_Type(self, node):
        pass


    def visit_ProcedureDecl(self, node:ProcedureDecl):
        pass


    def visit_ProcedureCall(self, node: ProcedureCall):
        proc_name = node.proc_name
        self.log(f'ENTER PROCEDURE: {proc_name}')

        ar = ActivationRecord(
            name=proc_name,
            type=ARType.PEOCEDURE,
            nesting_level=self.call_stack.peek().nesting_level + 1,
        )

        formal_params = node.proc_symbol.formal_params
        actual_params = node.actual_params
        for param_symbol, argument_node in zip(formal_params, actual_params):
            ar[param_symbol.name] = self.visit(argument_node)

        # Push activity record of precedure in call stack.
        self.call_stack.push(ar)


        self.log(str(self.call_stack))
        # Execute the body of procedure.

        self.visit(node.proc_symbol.block_ast)

        # Pop stack and exit this procedure.
        self.call_stack.pop()

        self.log(f'LEAVE PROCEDURE: {proc_name}')
        



def main():
    cmd_parser = argparse.ArgumentParser(
        description='Simple Pascal Interpreter'
    )
    cmd_parser.add_argument(
        'inputfile',
        help='Pascal source file',
    )
    cmd_parser.add_argument(
        '--scope',
        help='Print scope information',
        action='store_true',
    )

    args = cmd_parser.parse_args(['test.pas', '--scope'])

    global _SHOULD_LOG_SCOPE
    _SHOULD_LOG_SCOPE = args.scope

    # args.inputfile
    with open(args.inputfile, 'r', encoding='utf-8') as f:
        text = f.read()
        try:
            lexer = Lexer(text)
            parser = Parser(lexer)
            tree = parser.parse()

            analyzer = SemanticAnalyzer()
            analyzer.analyze(tree)
        except (LexerError, ParserError, SemanticError) as e:
            print(e.message)
            sys.exit(1)
        
    interpreter = Interpreter(parser)
    interpreter.interpret(tree)


if __name__ == '__main__':
    main()