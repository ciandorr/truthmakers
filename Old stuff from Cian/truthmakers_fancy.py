import sys
from functools import reduce
from tqdm import tqdm
import networkx as nx

#printout to stderr - I added since it was slow and I wanted feedback while running

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
from itertools import *

# In this code, *states* are frozensets, fusion is union, and parthood is subset.  (We have to use frozensets, not sets, because we want them to be included in tuples which need to be hashable to be included in further sets.)  So:

def fuse(pair):
    return pair[0] | pair[1] # we take a tuple as argument rather than two arguments so that we can use 'map', as in the definition of bottomwedge below.

def isproperpart(first, second):
    return first.issubset(second) and (first != second)

# This code is about *regular, bilateral propositions* - ordered pairs of regular sets of states.  But instead of actually working with the regular sets of states, we will represent each regular set X in turn as an ordered pair of a set of states X- and a single state X+.  X- is the set of states in X that have no proper part in X (so it's an 'antichain' in mathematical parlance); X+ is just a state, the top element in X.  

# To use this coding, we'll need the following funciton that takes a set of states X and returns X-, the subset containing just the minimal elements of X.

def minimize(stateset):
    listset = sorted(stateset, key = len, reverse = True)
    n = 0
    while n < len(listset):
        for test in listset:
            if isproperpart(test,listset[n]):
                listset.pop(n)
                n = n-1
                continue
        n = n+1
    return frozenset(listset)

# This is code I was using before that takes an arbitrary set of states and returns the smallest regular set containing that set.  I don't think I'm actually using it in this particular code.  

def regclose(prop):
    listset = sorted(prop)
    maxstate = frozenset([])
    for state in listset:
        maxstate = maxstate | state
    atoms = sorted(maxstate)
    for atom in atoms:
        n = 0
        while n < len(listset):
            new = fuse([listset[n], frozenset([atom])])
            if new in listset:
                n = n+1
            else:
                listset.append(new)
    return frozenset(listset)

# Since we are coding regular sets of states as ordered pairs of an antichain and a state, the operations of "wedge" and "vee" on regular sets need to be transferred to this coding.  The following operation takes two antichains X- and Y- and spits back the set of minimal elements in X ∧ Y (where X and Y could be any regular sets that have X- and Y- as their minimal elements).
    
def bottomwedge(first, second):
    return minimize(map(fuse, product(first, second)))

# Less complicatedly, disjunction corresponds to the operation on antichains that takes the union and then minimizes:

def bottomvee(first, second):
    return minimize(first | second) 
    
# We can use this to define the wedge and vee operations on pairs-coding-regular-sets:

def wedge(first, second):
    return (bottomwedge(first[0],second[0]),fuse([first[1],second[1]]))
    
def vee(first,second):
    return (bottomvee(first[0],second[0]),fuse([first[1],second[1]]))

# Finally, since we're dealing with *bilateral* propositions (pairs of [codes for] regular propositions), we define the conjunction and disjunction operations on them.  

# Conjunction wedges the verifier sets and vees the falsifier sets:
        
def conjoin(first, second):
    return (wedge(first[0], second[0]), vee(first[1], second[1]))

# Disjunction does the opposite:

def disjoin(first, second):
    return (vee(first[0], second[0]), wedge(first[1], second[1]))

# And finally, negation just swaps the two elements.      

def negate(x):
    return (x[1], x[0])

# OK.  Now, what we're actually going to do in this particular file is to start with some small set of bilateral propositions and use it to generate a bigger set containing all of the bilateral propositions we can get from the starting set by successively applying the three operations of conjunction, disjunction, and negation as defined above.  So, we define three functions that take a list of propositions as input and output a bigger list containing everything we can make from the input using one of these three operations.

# First, negation:

def negclose(c):
    n = 0
    while n < len(c):
        new = negate(c[n])
        if not (new in c):
            c.append(new)
            eprint(f"{len(c)}: {propasstring(new)}= ~{n}", end="\r")  # comment this out if you don't want a bunch of stuff spewing out on stderr
        n = n+1
    return c

# Next, conjunction, which is just slightly more complicated since conjunction is an operation on two (bilateral) propositions:
    
def conjclose(c):
    n = 0
    while n < len(c):
        m = 0
        while m < n:
            new = conjoin(c[m], c[n])
            if not (new in c):
                c.append(new)
                #print(f"{len(c)} = {n} ^ {m}")#, propasstring(new)+' = '+propasstring(c[m])+' ^ '+propasstring(c[n])
                eprint(f"{len(c)}: {propasstring(new)} = {n} ^ {m}", end="\r")
            m = m+1
        n = n+1
    return c

# Finally disjunction, which is just the same:

def disjclose(c):
    n = 0
    while n < len(c):
        m = 0
        while m < n:
            new = disjoin(c[m], c[n])
            if not (new in c):
                c.append(new)
                #eprint(f"{len(c)}={n}v{m}")
                eprint(f"{len(c)}: {propasstring(new)} = {n} v {m}", end="\r")
            m = m+1
        n = n+1
    return c

# Now, what we really want is everything we can get by applying these operations in any order.  But as Fine proves in 'A Theory of Truth-Condiitonal COntent Part I', regular nonempty bilateral propositions obey the distributive laws and De Morgan's laws.  This means that we can always change the order of operations to do negation first, then conjunction, and finally disjunction, so we use the following definition of the 'closure' of a set of bilateral propositions.  

def close(c):
    return(disjclose(conjclose(negclose(c))))

# We could also close under all three operations simultaneously (looping through as in the definition of conjclose(c), but adding disjunctions and negations as well as conjunctions when they aren't already in c); I don't think this would be much slower, and in some other settings (e.g. if we were working with arbitrary sets of states, or pairs of sets of states, rather than regular sets), we'd need to do it that way.  The only reason I did 'conjunctions and negations first' was because I was interested in seeing how many of the final list of propositions could be made just from those two operations. In the next version, I'd like to have a flexible 'close' function that takes a list c, a list of unary operations, and a list of binary functions, and closes c under all of the given unary operations and binary operations.  

# Next, we have some utility functions for helping to analyse the rather large list of generated propositions.  This one sifts out the special bilateral propositions of the form <{x},Y>, i.e. where the verifier set is a singleton.  Because we are coding regular sets as pairs of an antichain and a state that's above everything in the antichain, this is equivalent to the second element being in the first element.

def definites(props):        
    result = []
    for prop in props:
        verifiers = prop[0]
        falsifiers = prop[1]
        if verifiers[1] in verifiers[0]:  # i.e. state is in the antichain, which can only happen if it's the only member of the antichain
            result.append(prop)
    return result

# this one sifts out special bilateral propositions where the regular proposition consists of everything between a certain bottom state and the top state - i.e. where the length of the antichain is 1:

def diamonds(props):
    return [prop for prop in props if len(prop[0][0]) == 1]

# This is another function I cobbled together as a way of making the list of bilateral propositions output by close(c) easier to grok.  This one takes in a list 'props' of bilateral propositions (pairs of unilateral propositions), and spits back a list of length-two lists, where the first element of each list is a verifier set V, and the second element is the list of all falsifier sets F such that <V,F> belongs to props. 

def reorganise(props):        
    result = []
    for prop in props:
        verifiers = prop[0]
        falsifiers = prop[1]
        n = 0
        found = False
        while n < len(result) and found == False:
            if result[n][0] == verifiers:
                result[n][1] = result[n][1] + [falsifiers]
                found = True
            else:
                n = n+1
        if found == False:
            result = result + [[verifiers, [falsifiers]]]
#    result = result.sort(key = first)
    return result

# THe next two groups the given list of regaulr bilateral props in a different way.  We associate each proposition <V,F>, coded as <<V-,v>,<F-,f>>, with its 'L equivalence class' <V-,F-> and its 'M equivalence class' <v,f>.  Then we either list all the L equivalence classes with each one followed by a list of all the M-equivalence classees that are paired with it in props, or list the M equivalence classes with each one followed by a list of all the L-equivalence classes.  

def Lreorganise(props):        
    result = []
    for prop in props:
        Lclass = [prop[0][0], prop[1][0]]
        Mclass = [prop[0][1], prop[1][1]]
        n = 0
        found = False
        while n < len(result) and found == False:
            if result[n][0] == Lclass:
                result[n][1] = result[n][1] + [Mclass]
                found = True
            else:
                n = n+1
        if found == False:
            result = result + [[Lclass, [Mclass]]]
#    result = result.sort(key = first)
    return result

def Mreorganise(props):        
    result = []
    for prop in props:
        Lclass = [prop[0][0], prop[1][0]]
        Mclass = [prop[0][1], prop[1][1]]
        n = 0
        found = False
        while n < len(result) and found == False:
            if result[n][0] == Mclass:
                result[n][1] = result[n][1] + [Lclass]
                found = True
            else:
                n = n+1
        if found == False:
            result = result + [[Mclass, [Lclass]]]
#    result = result.sort(key = first)
    return result

# Here are a few more operations used for probing the final list of bilateral propositions.  This one takes a list of propositions and spits back the results of conjoining all of them with a given proposition:

def closewithconj(props,conjunct):
    return set([conjoin(prop,conjunct) for prop in props])

# These give the disjunction and conjunction of all the propositions in props:
    
def top(props):
    return reduce(disjoin, props)

def bottom(props):
    return reduce(conjoin, props)

# These give two other interesting propositions: the conjunction of all propositions of the form p∨¬p, and the disjunction of all propositions of the form p∧¬p.  

def fulltop(props):
    return reduce(conjoin, [disjoin(prop,negate(prop)) for prop in props])

def fullbottom(props):
    return reduce(disjoin, [conjoin(prop,negate(prop)) for prop in props])

# Now we have a bunch of functions which we use to output things as strings.  If we were redoing this code in a more object-oriented way, I guess that these would be 'repr' methods on the various sorts of objects.

# A state is a frozenset of letters: we output it as a string (e.g. 'r' or 'rgb'), sorting the letters in reverse order so we get 'rgb' not 'bgr'.

def stateasstring(state):
    return "".join(sorted(state, reverse = True))

# We output a set of states as a comma-separated string.  There's a bit of tinkering here to get the sets to come out in order like 'r,gb,rgb':
    
def mysortkey(state):
    return (10-len(state), stateasstring(state))

def statesetasstring(stateset):
    thelist = sorted(stateset, key = mysortkey, reverse = True)
    return ",".join([stateasstring(state) for state in thelist])

# Recall that we are coding regular sets of strings as a pair of a set of states (the antichain of bottom elements) and a single state.  We join these with a semicolon.  

def unipropasstring(prop):
    return statesetasstring(prop[0])+";"+stateasstring(prop[1])

# Finally, a bilateral proposition is output as the pair of two unilateral propositions, separated with | and surrounded by parentheses.

def propasstring(prop):
    return "(" + unipropasstring(prop[0])+" | "+unipropasstring(prop[1])+")"

# We're also interested in these 'L equivalence classes', which are ordered pairs of sets of states; we also separate these with a |:

def Lpropasstring(prop):
    return statesetasstring(prop[0][0])+'|'+statesetasstring(prop[1][0])

# This one just spits out a list of bilateral propositions, one per line:
           
def flatformat(mylist):
    return "\n".join([propasstring(prop) for prop in mylist])

# This one is for giving a string representation of the more useful lists outputted by the 'reorganize' functions above.  The first one is for Lreorganize, the second for Mreorganize.  
           
def formatbilist(mylist):
    result = ""
    for ver in mylist:
        result = result + statesetasstring(ver[0][0]) +"|" + statesetasstring(ver[0][1]) + ", "
        result = result + ", ".join(sorted([stateasstring(prop[0])+"|"+stateasstring(prop[1]) for prop in ver[1]], key=len)) +"\n" 
    return result

def formatbilistII(mylist):
    result = ""
    for ver in mylist:
        result = result + stateasstring(ver[0][0]) +"|" + stateasstring(ver[0][1]) + ", "
        result = result + ", ".join(sorted([statesetasstring(prop[0])+"|"+statesetasstring(prop[1]) for prop in ver[1]], key=len)) +"\n" 
    return result

# we already had this as 'flatformat' but for some reason here it is again.

def formatproplist(mylist):
    return "\n".join([propasstring(prop) for prop in mylist])

# Next, we have some functions for *inputting* things as strings.  If we did all this in an object-oriented way, I guess these would all be coded into the __init__ method of the objects.

# Right now there's some inconsistency in the inputs and outputs which seems bad - better to set things up so that we can go back and forth freely between each kind of thing and its string representation.

# a state is given just by a string, e.g. 'rgb', which we turn into the frozenset of its letters:

def statefromstring(string):
    return frozenset(string)

# a set of states is given by a space-separated list of strings.  Here, for some reason, I 'minimize' the result to make sure that it's an antichain - this shouldn't really be in this function.

def statesetfromstring(string):
    states = string.split(' ')
    return minimize([statefromstring(state) for state in states])

# this is for inputting the <antichain, top element> pair that codes up a regular proposition: we separate them with a semicolon:

def unipropfromstring(string):
    bits = string.split(';')
    return (statesetfromstring(bits[0]),statefromstring(bits[1]))

# a regular proposition is intput as a pair of two coded-regular-propositions, separated by |:

def propfromstring(string):
    components = string.split('|')
    return tuple([unipropfromstring(prop) for prop in components])

# if we want to specify a list of propositions we just give it as a list of strings rather than one big string:

def propsfromlist(props):
    return [propfromstring(prop) for prop in props]

# Now we have some functions whose purpose in this code is to help us output things as nice looking Hasse diagrams.  For this, we don't just need a list/set of propositions (or whatevers), we need some relation ≤ on the elements of the list - e.g. ≤_∧ or ≤_∨.  We represent this relation as a Digraph.  It would probably be more efficient to construct this in tandem with the list of propositions rather than afterwards.  Also, the approach here where we first code the entire transitive relation ≤ and then take its transitive reduction (which is what's actually used to display the Hasse diagram) seems a bit wasteful - it might be better to work with the transitive reduction from the get go.  

# Here are the two tests we'll use, depending on whether we're interested in ≤_∧ or ≤_∨:

def isconjunct(first,second):
    return conjoin(first,second) == second

def isdisjunct(first,second):
    return disjoin(first,second) == second

# This is the funciton that makes the Digraph representing the poset.  

def posetfromlist(list, test, wrapper = lambda x: x):
    poset = nx.DiGraph()
    poset.add_nodes_from([wrapper(item) for item in list])
    for x in tqdm(list):  # tqdm displays a progress bar, which is helpful here since this computatation takes a while.
        #eprint(f"processing {propasstring(x)}")
        for y in list:
            if x!=y and test(x,y):
                poset.add_edge(wrapper(x),wrapper(y))
    eprint('Done making poset')
    return poset

# Networkx turned out to have a built in function for taking transitive reductions.  If we decide we'll be using transitive reductions from the beginning, we won't need this step.  

def detransitivize(poset):
    return nx.transitive_reduction(poset)

# This is some code for outputting the transitively-reduce Digraph as a Graphviz .dot file, based on something I found online (https://gist.github.com/mapio/c44c029a1c1a5ff1ab59).  I didn't know about the pydot package which has a built in facility for this. 

def stringify(graph):
    return [(Lpropasstring(edge[0]),Lpropasstring(edge[1])) for edge in graph.edges] #the particular graph I was trying to make here was just showing the L-equivalence classes, hence the Lpropasstring.  But in our code we'll also be interested in graphs that show the full proposiiton, graphs of states, etc.  

# this function outputs a valid Graphviz .dot file based on the list of edges output by stringify.  

def to_dot( E ):
	res = [ 'digraph G { rankdir = BT; ' ]
	res.extend( [ '"{}"->"{}";'.format( *e ) for e in E ] )
	res.append( '}' )
	return '\n'.join( res )


# INPUT

# the next part of the file is where I was playing around with different starting sets of bilateral propositions, where in each case I was interested in seeing what would fall out when I took the closure of the starting set using the `close`  function (see above).  Because I didn't know about Jupyter notebooks, I was just editing this file and commenting different things out.  I'm leaving it in here to give a sense of the kind of exploration we're interested in:

#the original - this generates the set of bilateral propositions whose L-equivalence classes are graphed in the Hasse diagram that says 'original' in the slides.  This example is important for Fine: we start with three bilateral propositions, where each of them has a single verifier (r, g, b).  
atoms1 = propsfromlist(['r;r|g b;gb','g;g|r b;rb','b;b|r g;rg'])

#A variant where the starting propositions are upward closed on both sides.  If all we're interested in is the L-equivalence classes, this is equivalent to the previous.  
atoms2 = propsfromlist(['r;rgb|g b;rgb','g;rgb|r b;rgb','b;rgb|r g;rgb'])

#The 'alternative' version, where the starting propositions have singleton verifiers - the L-equivalence clases are graphed as 'alternate' in the slides.
atoms3 = propsfromlist(['r;r|gb;gb','g;g|rb;rb','b;b|rg;rg'])  

#A variant of the previous where everything upward closed
atoms4 = propsfromlist(['r;rgb|gb;rgb','g;rgb|rb;rgb','b;rgb|rg;rgb'])

#Here's asymmetric variant of the original where two of the three atoms are compossible
atoms5 = propsfromlist(['r;r|b;b','g;g|b;b','b;b|r g;rg'])

#Now we see what happens when we have 4 atomic states r, g, b, y rather than 3.  (These get very slow.).  First we have an analogue of atoms1 where the four atomic states are "mutually exclusive":
atoms6 = propsfromlist(['r;r|g b y;gby','g;g|r b y;rby','b;b|r g y;rgy','y;y|r g b;rgb'])

#4 atoms; any two are compossible
atoms7 = propsfromlist(['r;r|gb gy by;gby','g;g|rb ry by;rby','b;b|rg ry gy;rgy','y;y|rg rb gb;rgb'])

#4 atoms; any two compossible; everything upward-closed
atoms8 = propsfromlist(['r;rgby|gb gy by;rgby','g;rgby|rb ry by;rgby','b;rgby|rg ry gy;rgby','y;yrgb|rg rb gb;yrgb'])

#Some weird asymmetric variant
atoms9 = propsfromlist(['y;y|r g;rg', 'r;r|g y;gy','g;g|r b;rb','b;b|r g y;rgy'])        

# Now I start to get really crazy and have 5 atoms, which is going to be unbearably slow.
#
# 5 mutually exclusive atoms
atoms10 = propsfromlist(['r;r|g b y z;gbyz','g;g|r b y z;rbyz','b;b|r g y z;rgyz','y;y|r g b z;rgbz','z;z|r g b y;rgby'])

# 5 atoms; at most two compossible
atoms11 = propsfromlist(['r;r|gb gy gz by bz yz;gbyz','g;g|rb ry rz by bz yz;rbyz','b;b|rg ry rz gy gz yz;rgyz','y;y|rg rb rz gb gz bz;rgbz','z;z|rg rb ry gb gy by;rgby'])

# 5 atoms; at most three compossible
atoms12 = propsfromlist(['r;r|gby gbz gyz byz;gbyz','g;g|rby rbz ryz byz;rbyz','b;b|rgy rgz ryz gyz yz;rgyz','y;y|rgb rgz rbz gbz bz;rgbz','z;z|rgb rgy rby gby;rgby'])

atoms = atoms1  # edit this line depending on what starting atoms you want - or add new ones....

#Lastly, we have two different functions which I wanted to be able to rename to __main__ so I could call them by running this file in the terminal.  The first one is for generating a Graphviz file showing the ≤_∧ relation on the generated set of propositions close(atoms):

if __name__ == '__main1__':
    propositions = close(atoms)
    eprint(f"{len(propositions)} propositions") # comment out if you don't want stuff on stderr
    pairs = posetfromlist(propositions,isconjunct) # as mentioned above, would be more efficient to construct the poset as we go
    hasse = detransitivize(pairs) # also as mentioned above, would probably be more efficient to keep it transitively-reduced as we construct it
    has = stringify(hasse) 
    dot = to_dot(has)
    print(dot)  # I was just printing the dot to the terminal and then cutting-and-pasting it into Graphviz because I couldn't remember how to output files to disk in Python and I couldn't be bothered learning.  We'll instead want to show our graphs in our Jupyter notebook.  
    
# This one is just spitting out a big lot of string output showing the generated set of propositions, presented in different ways.  My original MO was to copy the bit of the string output I was interested, paste it into a spreadsheet, and just sort of stare at it.  This was slow and not very illuminating, which is why I started making the graphs; still, it's good to have the possibility of staring at the lists.  

if __name__ == '__main2__':
    states = conjclose(atoms) # first we make all the "state propositions", i.e. the ones we can get from the input 'atoms' by conjunction:
    eprint(f"{len(states)} states")
    literals = negclose(states)
    eprint(f"{len(literals)} literals") # literals are state propositions and negations of state propositions.  
    # eprint flatformat(literals)  # uncomment if you want a big list of the propositions to be spit out to stderr..
    conjunctions = conjclose(literals) # conjunctions of literals
    eprint(f"{len(conjunctions)} conjunctions")
    # eprint flatformat(conjunctions)
    propositions = disjclose(conjunctions) # disjuncitons of conjunctions of literals.  I could got the same result by just writing propositions = close(atoms), and cutting all the preceding lines; I guess I did it this way because I was interested in seeing the intermediate steps, or maybe I just hadn't written the close function when I wrote this bit....
    eprint(f"{len(propositions)} propositions")    
    # reorganised = reorganise(propositions)
    # eprint(f"{len(reorganised)} verifier-sets")
    Lclasses = Lreorganise(propositions) # the two different useful ways of organizing a list of bilateral propositions.  
    Mclasses = Mreorganise(propositions)
    print(f"\n{len(Lclasses)} L-equivalence classes")
    print(formatbilist(Lclasses)) # prints the L-equivalence classes
    print(f"{len(Mclasses)} M-equivalence classes")
    print(formatbilistII(Mreorganise(propositions))) # prints the M-equivalence classes
    top = top(propositions)
    bottom = bottom(propositions)
    fulltop = fulltop(propositions)
    fullbottom = fullbottom(propositions) # the last 4 lines aren't actually generating output, but I guess I was querying the variables in an interactive shell.  
    
    


                
        

# for prop in propositions:
#     verifiers = prop[0]
#     falsifiers = prop[1]
#     str = ""
#     for v in verifiers:
#         str = str+" "+("".join(v))
#     str = str+" | "
#     for f in falsifiers:
#         str = str+" "+("".join(f))
#     print str