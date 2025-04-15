import sys

# Represents a term which can be a constant, variable, or function
class Term:
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []

    def is_variable(self):
        # Checks if the term is a variable (lowercase and no arguments)
        return self.args == [] and self.name[0].islower()

    def __eq__(self, other):
        return isinstance(other, Term) and self.name == other.name and self.args == other.args

    def __hash__(self):
        return hash((self.name, tuple(self.args)))

    def __repr__(self):
        # String representation of the term
        if self.args:
            return f"{self.name}({', '.join(map(str, self.args))})"
        else:
            return self.name

# Represents a literal, possibly negated, with a predicate and terms
class Literal:
    def __init__(self, predicate, terms, negated=False):
        self.predicate = predicate
        self.terms = terms
        self.negated = negated

    def __eq__(self, other):
        return (self.predicate == other.predicate and
                self.terms == other.terms and
                self.negated == other.negated)

    def __hash__(self):
        return hash((self.predicate, tuple(self.terms), self.negated))

    def __repr__(self):
        # String representation of the literal
        neg = '!' if self.negated else ''
        if self.terms:
            return f"{neg}{self.predicate}({', '.join(map(str, self.terms))})"
        else:
            return f"{neg}{self.predicate}"

# Represents a clause, which is a disjunction of literals
class Clause:
    def __init__(self, literals):
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return self.literals == other.literals

    def __hash__(self):
        return hash(self.literals)

    def __repr__(self):
        # String representation of the clause
        if not self.literals:
            return 'Empty Clause'
        return ' | '.join(map(str, self.literals))

# Parses the input cnf and returns clauses
def parseCNF(k):
    lines = k.strip().split('\n')
    data = {}
    key = None
    value = ''
    for line in lines + ['']:
        if ':' in line:
            if key is not None:
                data[key.strip()] = value.strip()
            key, val = line.split(':', 1)
            value = val.strip() + '\n'
        else:
            value += line.strip() + '\n'
    if key is not None:
        data[key.strip()] = value.strip()

    # Parse clauses into Clause objects
    input = data.get('Clauses', '').strip().split('\n')
    clauses = []
    for clause in input:
        literals = []
        tokens = clause.strip().split()
        for token in tokens:
            if token == '|':
                continue
            negated = False
            if token.startswith('!'):
                negated = True
                token = token[1:]
            predicate, args = parsePredicate(token)
            literals.append(Literal(predicate, args, negated))
        if literals:
            clauses.append(Clause(literals))
    return clauses

# Parses a predicate token into its name and list of term arguments
def parsePredicate(token):
    if '(' in token:
        predicate, rest = token.split('(', 1)
        args = rest[:-1].split(',')
        terms = [parseTerm(arg.strip()) for arg in args]
    else:
        predicate = token
        terms = []
    return predicate, terms

# Parses a term token into a Term object
def parseTerm(token):
    if '(' in token:
        functionName, rest = token.split('(', 1)
        args = rest[:-1].split(',')
        terms = [parseTerm(arg.strip()) for arg in args]
        return Term(functionName, terms)
    else:
        return Term(token)

# Attempts to unify two terms x and y with a given substitute
# Returns the updated substitution if successful, else None
def unify(x, y, substitute):
    if substitute is None:
        return None
    elif x == y:
        return substitute
    elif isinstance(x, Term) and x.is_variable():
        return unifyVariable(x, y, substitute)
    elif isinstance(y, Term) and y.is_variable():
        return unifyVariable(y, x, substitute)
    elif isinstance(x, Term) and isinstance(y, Term):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        for a, b in zip(x.args, y.args):
            substitute = unify(a, b, substitute)
            if substitute is None:
                return None
        return substitute
    else:
        return None

# Unifies a variable with term x under the substitute
def unifyVariable(variable, x, substitute):
    if variable.name in substitute:
        return unify(substitute[variable.name], x, substitute)
    elif isinstance(x, Term) and x.is_variable() and x.name in substitute:
        return unify(variable, substitute[x.name], substitute)
    elif occurCheck(variable, x, substitute):
        return None
    else:
        substitute[variable.name] = x
        return substitute

# Attempts to unify two lists of terms
def unifyTerms(terms1, terms2, substitute):
    if len(terms1) != len(terms2):
        return None
    for term1, term2 in zip(terms1, terms2):
        substitute = unify(term1, term2, substitute)
        if substitute is None:
            return None
    return substitute

# Checks for circular references to prevent infinite loops during unification
def occurCheck(variable, x, substitute):
    if variable == x:
        return True
    elif isinstance(x, Term) and x.is_variable() and x.name in substitute:
        return occurCheck(variable, substitute[x.name], substitute)
    elif isinstance(x, Term):
        return any(occurCheck(variable, arg, substitute) for arg in x.args)
    else:
        return False

# Applies the substitution substitute to all literals in a clause
def substitute(clause, substitute):
    new_literals = set()
    for literal in clause.literals:
        new_terms = [substituteTerm(term, substitute) for term in literal.terms]
        new_literals.add(Literal(literal.predicate, new_terms, literal.negated))
    return Clause(new_literals)

# Recursively applies the substitution substitute to a term
def substituteTerm(term, substitute):
    if term.is_variable():
        if term.name in substitute:
            return substituteTerm(substitute[term.name], substitute)
        else:
            return term
    elif term.args:
        return Term(term.name, [substituteTerm(arg, substitute) for arg in term.args])
    else:
        return term

# Applies substitution substitute to a literal
def substituteLiterals(literal, substitute):
    new_terms = [substituteTerm(term, substitute) for term in literal.terms]
    return Literal(literal.predicate, new_terms, literal.negated)

# Checks if a set of literals contains both a literal and its negation
def complementaryLiterals(literals):
    literals_set = set(literals)
    for literal in literals:
        complementary = Literal(literal.predicate, literal.terms, not literal.negated)
        if complementary in literals_set:
            return True
    return False

# Resolves two clauses clause1 and clause2, returning a set of resolvent clauses
def resolve(clause1, clause2):
    resolvents = set()
    for literal1 in clause1.literals:
        for literal2 in clause2.literals:
            # Look for complementary literals
            if literal1.predicate == literal2.predicate and literal1.negated != literal2.negated:
                substitute = {}
                # Unify the terms directly
                substitute = unifyTerms(literal1.terms, literal2.terms, substitute)
                if substitute is not None:
                    # Apply substitutions to clauses
                    clause1Substitute = substitute(clause1, substitute)
                    clause2Substitute = substitute(clause2, substitute)
                    # Remove the resolved literals
                    literal1Substitute = substituteLiterals(literal1, substitute)
                    literal2Substitute = substituteLiterals(literal2, substitute)
                    new_literals = (clause1Substitute.literals | clause2Substitute.literals) - {literal1Substitute, literal2Substitute}
                    # Avoid tautologies
                    if complementaryLiterals(new_literals):
                        continue
                    resolvents.add(Clause(new_literals))
    return resolvents

# Performs the resolution algorithm on a set of clauses
def resolution(clauses):
    clauses_set = set(clauses)
    new = set()
    while True:
        pairs = []
        clauses_list = list(clauses_set)
        n = len(clauses_list)
        # Generate all possible pairs of clauses
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((clauses_list[i], clauses_list[j]))
        # Attempt to resolve each pair
        for (clause1, clause2) in pairs:
            resolvents = resolve(clause1, clause2)
            if Clause(frozenset()) in resolvents:
                return 'no'
            new.update(resolvents)
        # Check if new clauses are already known
        if new.issubset(clauses_set):
            return 'yes'
        clauses_set.update(new)
        new.clear()

# Main function to read input, parse clauses, and perform resolution
def main():
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        content = f.read()
    # Split input into individual cases
    input_cases = content.strip().split('\n\n\n')
    for k in input_cases:
        clauses = parseCNF(k)
        result = resolution(clauses)
        print(result)

if __name__ == '__main__':
    main()
