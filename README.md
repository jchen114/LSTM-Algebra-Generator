# LSTM Algebraic Expressions Generator
A demonstration of using an LSTM network to generate syntactically correct Algebraic sequences. This demonstration hopes to showcase the ability of using LSTM networks to discover dependencies within a sequence and act accordingly.

## Input
The sequence follows these set of rules:

1. An Expression (E) can transition into a terminal character (I); E -> I
2. An Expression (E) can transition into a multiplicative expression; E -> M * M
3. An Expression (E) can transition into an additive expression; E -> E + E
4. A Multiplicative expression M can transition into a terminal character (I); M -> I
5. A Multiplicative expression M can transition into another multiplicative expression; M -> M * M
6. A Multiplicative expression M can transition into a parenthetical additive expression M -> ( E + E )

Using these sets of rules, we can generate sequences like:
I + I * (I + I * (I + I)) + I * (I + I) + I

## Output
It is the goal of the network such that when given a seed it should continue to generate a syntactically correct sequence by recursively feeding the new sequence back into itself until it hits an end character, namely the '\n'.
For example, on a seed of [(, (, (] it should generate a sequence that closes these parenthesis.

## Architecture
I use a two layer LSTM network with 128 hidden units each. There are 6 classes: 'I', '+', '\*', '(', ')', '\n'. The network is trained using a softmax output with categorical cross entropy loss. The network is asked to predict on every label during training. I found that this results in faster convergence. Sequences are presented in batches of 15 time sequences. Each time sequence entry has 1 feature.

## Here are some test cases.
Seed: '('

Produces:
`
['(',
  'I', '\*', 'I', '\*',
  '(',
   'I', '+',
   '(',
    'I', '\*', 'I', '+', 'I',
   ')', '\*', 'I', '+', 'I', '+', 'I',
  ')', '\*', 'I', '\*', 'I', '\*', 
  '(',
   'I', '+', 'I', '\*',
   '(',
      'I', '+', 'I', '\*', 'I',
   ')', '+', 'I', '+', 'I', '+', 'I', '+', 'I', '\*', 'I', '\*', 'I',
  ')', '\*', 'I', '\*', 'I', '\*', 'I',
 ')', '\*', 'I', '\*', 'I', '\n']
`

 The paranthesis are balanced and accounted for.

 Seed: '(', '(', '('

 Produces:
 `
 ['(', '(', '(', 'I', '+', 'I', '+', 'I', '\*', 'I', ')', '\*', 'I', '+', 'I', '\*', 'I', ')', '\*', 'I', '+', 'I', ')', '\*', 'I', '\*', 'I', '\*', 'I', '\*', '(', '(', 'I', '+', 'I', ')', '\*', 'I', '\*', 'I', '+', '(', '(', 'I', '+', 'I', '\*', '(', 'I', '+', 'I', ')', '\*', 'I', '+', 'I', '\*', 'I', '\*', '(', '(', 'I', '+', 'I', '+', 'I', '+', 'I', ')', '\*', '(', '(', 'I', '+', 'I', ')', '\*', 'I', '+', 'I', ')', '\*', 'I', '\*', 'I', '+', 'I', ')', '+', 'I', '\*', 'I', ')', '\*', 'I', '+', 'I', '\*', 'I', '\*', '(', 'I', '+', 'I', '\*', 'I', '+', 'I', '\*', 'I', '+', 'I', ')', ')', '\*', '(', 'I', '\*', 'I', '\*', 'I', '\*', 'I', '\*', '(', 'I', '+', 'I', ')', '+', 'I', '+', 'I', '+', 'I', '\*', '(', 'I', '+', 'I', '+', 'I', '+', 'I', '\*', '(', 'I', '\*', 'I', '\*', 'I', '\*', '(', 'I', '+', 'I', ')', '\*', 'I', '+', 'I', ')', '+', 'I', '\*', 'I', ')', ')', '\*', '(', 'I', '\*', 'I', '+', '(', 'I', '+', 'I', '\*', 'I', '+', 'I', ')', '\*', 'I', '+', 'I', ')', '+', 'I', '\*', 'I', ')', '\*', '(', 'I', '+', 'I', '\*', '(', 'I', '+', 'I', '+', 'I', ')', ')', '\*', 'I', '\*', 'I', '\*', 'I', '\n']`
