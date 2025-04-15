% childOf(X,Y) - X is a child of Y
% Facts for both parents
childOf(andrew, elizabeth).
childOf(andrew, philip).
childOf(anne, elizabeth).
childOf(anne, philip).
childOf(beatrice, andrew).
childOf(beatrice, sarah).
childOf(charles, elizabeth).
childOf(charles, philip).
childOf(diana, kydd).
childOf(diana, spencer).
childOf(edward, elizabeth).
childOf(edward, philip).
childOf(elizabeth, george).
childOf(elizabeth, mum).
childOf(eugenie, andrew).
childOf(eugenie, sarah).
childOf(harry, charles).
childOf(harry, diana).
childOf(james, edward).
childOf(james, sophie).
childOf(louise, edward).
childOf(louise, sophie).
childOf(margaret, george).
childOf(margaret, mum).
childOf(peter, anne).
childOf(peter, mark).
childOf(william, charles).
childOf(william, diana).
childOf(zara, anne).
childOf(zara, mark).

% Gender Facts
female(anne).
female(beatrice).
female(diana).
female(elizabeth).
female(eugenie).
female(kydd).
female(louise).
female(margaret).
female(mum).
female(sarah).
female(sophie).
female(zara).

male(andrew).
male(charles).
male(edward).
male(george).
male(harry).
male(james).
male(mark).
male(peter).
male(philip).
male(spencer).
male(william).

% Married relationships
married(anne, mark).
married(diana, charles).
married(elizabeth, philip).
married(kydd, spencer).
married(mum, george).
married(sarah, andrew).
married(sophie, edward).

% Helper predicate: parentOf(P, C) - P is a parent of C
parentOf(P, C) :- childOf(C, P).

% spouse(X,Y) - Symmetric version of married
spouse(X, Y) :- married(X, Y).
spouse(X, Y) :- married(Y, X).

% daughterOf(X, Y) - X is the female child of Y
daughterOf(X, Y) :-
    childOf(X, Y),
    female(X).

% sonOf(X, Y) - X is the male child of Y
sonOf(X, Y) :-
    childOf(X, Y),
    male(X).

% sibling(X, Y) - X and Y share at least one parent
sibling(X, Y) :-
    parentOf(P, X),
    parentOf(P, Y),
    X \= Y.

% brotherOf(X, Y) - X is the male sibling of Y
brotherOf(X, Y) :-
    male(X),
    sibling(X, Y).

% sisterOf(X, Y) - X is the female sibling of Y
sisterOf(X, Y) :-
    female(X),
    sibling(X, Y).

% grandchildOf(X, Y) - X is a grandchild of Y
grandchildOf(X, Y) :-
    childOf(X, P),
    childOf(P, Y).

% ancestorOf(X, Y) - X is an ancestor of Y
ancestorOf(X, Y) :-
    parentOf(X, Y).
ancestorOf(X, Y) :-
    parentOf(X, Z),
    ancestorOf(Z, Y).

% auntOf(X, Y) - X is the aunt of Y
auntOf(X, Y) :-
    female(X),
    sibling(X, P),
    parentOf(P, Y).

% uncleOf(X, Y) - X is the uncle of Y
uncleOf(X, Y) :-
    male(X),
    sibling(X, P),
    parentOf(P, Y).

% firstCousinOf(X, Y) - X is the first cousin of Y
firstCousinOf(X, Y) :-
    parentOf(PX, X),
    parentOf(PY, Y),
    sibling(PX, PY),
    X \= Y.

% brotherInLawOf(X, Y) - X is the brother of Y's spouse or the male spouse of Y's sibling
brotherInLawOf(X, Y) :-
    male(X),
    spouse(Y, Spouse),
    sibling(X, Spouse).
brotherInLawOf(X, Y) :-
    male(X),
    sibling(Sibling, Y),
    spouse(X, Sibling).

% sisterInLawOf(X, Y) - X is the sister of Y's spouse or the female spouse of Y's sibling
sisterInLawOf(X, Y) :-
    female(X),
    spouse(Y, Spouse),
    sibling(X, Spouse).
sisterInLawOf(X, Y) :-
    female(X),
    sibling(Sibling, Y),
    spouse(X, Sibling).
