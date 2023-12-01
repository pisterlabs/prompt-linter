"""Generated summarized paragraph for input text"""
import cohere
import pandas as pd

api_key = 'MklIKiJvqX1nFagSi1jRU4k9YxoxfLwZvRG6xIUJ'
co = cohere.Client(api_key)


def summarize(prompt: str):
    """summarizes the input prompt

    prompt: string containing the info that needs to be summarized"""

    sample_text = '''Passage: " Combinational logic is often grouped into larger building blocks to build more complex systems. This is an application of the principle of abstraction, hiding the unnecessary gate-level details to emphasize the function of the building block. We have already studied three such building blocks: full adders, priority circuits, and seven-segment display decoders. This section introduces two more commonly used building blocks: multiplexers and decoders. Chapter 5 covers other combinational building blocks. Multiplexers are among the most commonly used combinational circuits. They choose an output from among several possible inputs based on the value of a select signal. A multiplexer is sometimes affectionately called a mux. Figure 2.54 shows the schematic and truth table for a 2:1 multiplexer with two data inputs, D0 and D1, a select input, S, and one output, Y. The multiplexer chooses between the two data inputs based on the select. A 2:1 multiplexer can be built from sum-of-products logic as shown in Figure 2.55. The Boolean equation for the multiplexer may be derived with a Karnaugh map or read off by inspection. Alternatively, multiplexers can be built from tristate buffers, as shown in Figure 2.56. The tristate enables are arranged such that, at all times, exactly one tristate buffer is active. A 4:1 multiplexer has four data inputs and one output, as shown in Figure 2.57. Two select signals are needed to choose among the four data inputs. The 4:1 multiplexer can be built using sum-of-products logic, tristates, or multiple 2:1 multiplexers, as shown in Figure 2.58. The product terms enabling the tristates can be formed using AND gates and inverters. They can also be formed using a decoder, which we will introduce in Section 2.8.2. Wider multiplexers, such as 8:1 and 16:1 multiplexers, can be built by expanding the methods shown in Figure 2.58. In general, an N:1 multiplexer needs log2N select lines. Again, the best implementation choice depends on the target technology. Multiplexers can be used as lookup tables to perform logic functions. Figure 2.59 shows a 4:1 multiplexer used to implement a two-input AND gate. The inputs, A and B, serve as select lines. The multiplexer data inputs are connected to 0 or 1 according to the corresponding row of the truth table. In general, a 2N-input multiplexer can be programmed to perform any N-input logic function by applying 0’s and 1’s to the appropriate data inputs. Indeed, by changing the data inputs, the multiplexer can be reprogrammed to perform a different function. With a little cleverness, we can cut the multiplexer size in half, using only a 2N1-input multiplexer to perform any N-input logic function. The strategy is to provide one of the literals, as well as 0’s and 1’s, to the multiplexer data inputs. To illustrate this principle, Figure 2.60 shows two-input AND and XOR functions implemented with 2:1 multiplexers. We start with an ordinary truth table, and then combine pairs of rows to eliminate the rightmost input variable by expressing the output in terms of this variable. We then use the multiplexer as a lookup table according to the new, smaller truth table. A decoder has N inputs and 2N outputs. It asserts exactly one of its outputs depending on the input combination. Figure 2.63 shows a 2:4 decoder. The outputs are called one-hot, because exactly one is “hot” (HIGH) at a given time. Decoders can be combined with OR gates to build logic functions. Figure 2.65 shows the two-input XNOR function using a 2:4 decoder and a single OR gate. Because each output of a decoder represents a single minterm, the function is built as the OR of all the minterms in the function. When using decoders to build logic, it is easiest to express functions as a truth table or in canonical sum-of-products form. An N-input function with M 1’s in the truth table can be built with an N:2N decoder and an M-input OR gate attached to all of the minterms containing 1’s in the truth table. This concept will be applied to the building of Read Only Memories (ROMs) in Section 5.5.6."
    Summary of Passage: "Logic gates are combined to produce larger circuits such as multiplexers, decoders, and priority circuits. A multiplexer chooses one of the data inputs based on the select input. A decoder sets one of the outputs HIGH according to the input. A priority circuit produces an output indicating the highest priority input."

    ---

    Passage: "**********"
    Summary of Passage:"'''

    summary_input = sample_text.replace("**********", prompt)

    # print(summary_input)

    response = co.generate(
        model='xlarge',
        prompt=summary_input,
        return_likelihoods='GENERATION',
        stop_sequences=['"'],
        max_tokens=200,
        temperature=0.7,
        num_generations=5,
        k=0,
        p=0.75)

    gens = []
    likelihoods = []

    for gen in response.generations:
        gens.append(gen.text)
        sum_likelihood = 0
        for t in gen.token_likelihoods:
            sum_likelihood += t.likelihood

        likelihoods.append(sum_likelihood)

    pd.options.display.max_colwidth = 200
    df = pd.DataFrame({'generation': gens, 'likelihood': likelihoods})
    df = df.drop_duplicates(subset=['generation'])
    df = df.sort_values('likelihood', ascending=False, ignore_index=True)
    # print(df)

    return df["generation"].iloc[0]


# humble = "The Humble Object pattern is a design pattern that was originally identified as a way to help unit testers to separate behaviors that are hard to test from behaviors that are easy to test. The idea is very simple: Split the behaviors into two modules or classes. One of those modules is humble; it contains all the hard-to-test behaviors stripped down to their barest essence. The other module contains all the testable behaviors that were stripped out of the humble object. For example, GUIs are hard to unit test because it is very difficult to write tests that can see the screen and check that the appropriate elements are displayed there. However, most of the behavior of a GUI is, in fact, easy to test. Using the Humble Object pattern, we can separate these two kinds of behaviors into two different classes called the Presenter and the View."
# my_text2 = "It is possible that a single input transition can cause multiple output transitions. These are called glitches or hazards. Although glitches usually don’t cause problems, it is important to realize that they exist and recognize them when looking at timing diagrams."
# my_text = "The test for single-gene inheritance is to mate individuals showing the mutant property with wild-type and then analyze the first and second generation of descendants. As an example, a mutant plant with white flowers would be crossed to the wild type showing red flowers. The progeny of this cross are analyzed, and then they themselves are interbred to produce a second generation of descendants. In each generation, the diagnostic ratios of plants with red flowers to those with white flowers will reveal whether a single gene controls flower color. If so, then by inference, the wild type would be encoded by the wild-type form of the gene and the mutant would be encoded by a form of the same gene in which a mutation event has altered the DNA sequence in some way. Other mutations affecting flower color (perhaps mauve, blotched, striped, and so on) would be analyzed in the same way, resulting overall in a set of defined “flower-color genes.” The use of mutants in this way is sometimes called genetic dissection, because the biological property in question (flower color in this case) is picked apart to reveal its underlying genetic program, not with a scalpel but with mutants. Each mutant potentially identifies a separate gene affecting that property. After a set of key genes has been defined in this way, several different molecular methods can be used to establish the functions of each of the genes. These methods will be covered in later chapters. Hence, genetics has been used to define the set of gene functions that interact to produce the property we call flower color (in this example). This type of approach to gene discovery is sometimes called forward genetics, a strategy to understanding biological function starting with random single-gene mutants and ending with their DNA sequence and biochemical function."
# bio = "What kinds of research do biologists do? One central area of research in the biology of all organisms is the attempt to understand how an organism develops from a fertilized egg into an adult—in other words, what makes an organism the way it is. Usually, this overall goal is broken down into the study of individual biological properties such as the development of plant flower color, or animal locomotion, or nutrient uptake, although biologists also study some general areas such as how a cell works. How do geneticists analyze biological properties? The genetic approach to understanding any biological property is to find the subset of genes in the genome that influence that property, a process sometimes referred to as gene discovery. After these genes have been identified, their cellular functions can be elucidated through further research. There are several different types of analytical approaches to gene discovery, but one widely used method relies on the detection of single-gene inheritance patterns, and that is the topic of this chapter. All of genetics, in one aspect or another, is based on heritable variants. The basic approach of genetics is to compare and contrast the properties of variants, and from these comparisons make deductions about genetic function. It is similar to the way in which you could make inferences about how an unfamiliar machine works by changing the composition or positions of the working parts, or even by removing parts one at a time. Each variant represents a 'tweak”'of the biological machine, from which its function can be deduced."



# print(summarize(my_text))
