import numbers

def generate_equation(model):
    """
    Generates the equation of a given PDE-Net by algebraic explansion of the SymNet multi-layer perceptron.
    Uses a combination of classes included in this file.

    Parameters
    ----------
      model : PDE_NET
          An instance of a trained PDE_NET model

    """
    params = list(model.parameters())
    variables = model.gradient_generator.variables
    base_terms = [Term(1,[var]) for var in variables]

    for param in params[:-2]:
        result0 = []
        result1 = []
        for i in range(len(param[0])):
            result0.append(TermUtil.mult(param[0][i].item(), base_terms[i]))
            result1.append(TermUtil.mult(param[1][i].item(), base_terms[i]))
        col = TermUtil.mult(TermCollection.from_list(result0),TermCollection.from_list(result1))
        base_terms.append(col)

    coefs = params[-2]
    result = []
    for i in range(len(coefs[0])):
        result.append(TermUtil.mult(coefs[0][i].item(), base_terms[i]))
    result.append(Term(params[-1][0]))
    final_col = TermCollection.from_list(result)
    return final_col

class Term:
    def __init__(self,coef,var = []):
        try: 
            self.coef = coef.item()
        except:
            self.coef = coef
        var.sort()
        self.var = var
        
    def __str__(self):
        return str("%.4f" % self.coef) + '*' + str(self.var)

    def __repr__(self):
        return str(self)
    
    def __gt__(self, term2):
        return abs(self.coef) > abs(term2.coef)
    
    def __lt__(self, term2):
        return abs(self.coef) < abs(term2.coef)
    
    def __eq__(self, term2):
        return abs(self.coef) == abs(term2.coef)
    
class TermCollection:
    def __init__(self, term_list):
        self.term_list = term_list
        self.simplify()

    def from_list(some_list):
        new_col = []
        for elem in some_list:
            if isinstance(elem,Term): 
                new_col.append(elem)
            elif isinstance(elem, TermCollection):
                new_col += elem.term_list
        
        return TermCollection(new_col)
        
    def __str__(self):
        return ' + '.join([str(x) for x in self.term_list])
    
    def __repr__(self):
        return str(self)    
    
    def length(self):
        return len(self.term_list)
    
    def print_collection(self):
        for x in self.term_list: print(x)
    
    def simplify(self):
        result = []
        visited = [0]*len(self.term_list)
        for i in range(len(self.term_list)):
            if visited[i]:
                continue
            visited[i] = 1
            term1 = self.term_list[i]
            for j in range(i+1, len(self.term_list)):
                term2 =  self.term_list[j]
                if term1.var == term2.var:
                    visited[j] = 1
                    term1 = TermUtil.add_tt(term1,term2)
            result.append(term1)
        self.term_list = result
    
    def show_topk(self,k):
        return TermCollection(sorted(self.term_list, reverse=True)[:k])
        

class TermUtil:        
    def mult_nt(number, term):
        return Term(term.coef * number, term.var)
    
    def mult_nl(number, col):
        return TermCollection([TermUtil.mult_nt(number, term) for term in col.term_list])
    
    def mult_tt(term1, term2):
        var = term1.var + term2.var
        var.sort()
        return Term(term1.coef * term2.coef, var)

    def add_tt(term1, term2):
        assert(term1.var == term2.var)
        return Term(term1.coef + term2.coef, term1.var)
    
    def mult_ll(col1, col2):
        result = []
        for i in range(col1.length()):
            for j in range(col2.length()):
                term = TermUtil.mult_tt(col1.term_list[i], col2.term_list[j])
                result.append(term)
        return TermCollection(result)
    
    def mult(a,b):
        if isinstance(a,numbers.Number):
            if isinstance(b, Term):
                return TermUtil.mult_nt(a,b)
            elif isinstance(b, TermCollection):
                return TermUtil.mult_nl(a,b)   
            
        elif isinstance(a, Term):
            if isinstance(b, Term):
                return TermUtil.mult_tt(a,b)
            elif isinstance(b, TermCollection):
                return TermUtil.mult_tl(a,b)
            
            
        elif isinstance(a, TermCollection):
            if isinstance(b, TermCollection):
                return TermUtil.mult_ll(a,b)
            else:
                return TermUtil.mult(b,a)